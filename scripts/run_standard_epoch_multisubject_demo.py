"""Run a dataset-neutral multi-subject Agent demo from a validated MAT epoch contract."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bci_autodiscovery.agents.audit import JsonlAuditSink
from bci_autodiscovery.agents.research_design_agent import ResearchDesignAgent
from bci_autodiscovery.demo.cli import (
    PRICING,
    _progress,
    _raw_provider,
    _run_subject,
    _summarize_completed_subject,
)
from bci_autodiscovery.demo.contracts import write_json
from bci_autodiscovery.pipelines import DeterministicPipelineExecutor, PipelineSpec
from bci_autodiscovery.profiling import MatEpochSource
from bci_autodiscovery.search import build_dataset_incumbent
from bci_autodiscovery.workflow.autonomy import (
    load_autonomy_envelope,
    load_json_object,
    sha256_path,
)
from bci_autodiscovery.workflow.budget import BudgetLedger
from bci_autodiscovery.workflow.dataset_contract import load_dataset_level_contract


def _limits(value: dict[str, Any]) -> dict[str, float]:
    return {key: float(item) for key, item in value.items()}


def _allocate_budget(
    *, envelope: dict[str, Any], subjects: list[str], candidate_count: int
) -> dict[str, Any]:
    budget = envelope["resource_budget"]
    design_tokens = min(350_000, int(budget["max_api_tokens"]) // 4)
    remaining_tokens = int(budget["max_api_tokens"]) - design_tokens
    per_subject_tokens = remaining_tokens // len(subjects)
    design_cost = min(1.2, float(budget["max_paid_cost"]) / 4)
    remaining_cost = float(budget["max_paid_cost"]) - design_cost
    per_subject_cost = remaining_cost / len(subjects)
    candidate_executions = candidate_count * len(subjects)
    maximum_candidates = int(budget["max_candidate_executions"])
    remaining_candidate_executions = maximum_candidates - candidate_executions
    remaining_research_cycles = int(budget["max_research_cycles"]) - candidate_count
    per_subject_candidate_budget = min(
        remaining_candidate_executions // len(subjects),
        remaining_research_cycles // len(subjects),
    )
    if per_subject_candidate_budget < 2:
        raise ValueError(
            "Envelope cannot fund at least two individualized candidates per subject "
            "after the dataset-incumbent grid"
        )
    design_retries = min(4, int(budget["max_api_retries"]))
    remaining_retries = int(budget["max_api_retries"]) - design_retries
    per_subject_retries = remaining_retries // len(subjects)
    allocation = {
        "schema_version": "1.0",
        "status": "frozen_before_research_design",
        "envelope_id": envelope["envelope_id"],
        "subjects": subjects,
        "interpretation": {
            "confirmation_max_access_count": "one access per subject",
            "shared_confirmation_data": False,
            "per_subject_candidate_budget": (
                "maximum equal allocation remaining after the complete dataset-incumbent "
                "grid, bounded by both global candidate-execution and research-cycle caps"
            ),
        },
        "design": {
            "research_cycles": 0,
            "candidate_executions": 0,
            "compute_seconds": 120,
            "api_total_tokens": design_tokens,
            "paid_cost": design_cost,
            "provider_retries": design_retries,
            "recovery_attempts": 1,
            "confirmation_accesses": 0,
        },
        "dataset_incumbent": {
            "research_cycles": candidate_count,
            "candidate_executions": candidate_executions,
            "compute_seconds": min(1200, float(budget["max_compute_seconds"]) / 3),
            "api_total_tokens": 0,
            "paid_cost": 0,
            "provider_retries": 0,
            "recovery_attempts": 0,
            "confirmation_accesses": 0,
        },
        "per_subject": {
            subject_id: {
                "research_cycles": per_subject_candidate_budget,
                "candidate_executions": per_subject_candidate_budget,
                "compute_seconds": min(1200, float(budget["max_compute_seconds"]) / len(subjects)),
                "api_total_tokens": per_subject_tokens,
                "paid_cost": per_subject_cost,
                "provider_retries": per_subject_retries,
                "recovery_attempts": 1,
                "confirmation_accesses": 1,
            }
            for subject_id in subjects
        },
    }
    return allocation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--dataset-level-contract", type=Path, required=True)
    parser.add_argument("--autonomy-envelope", type=Path, required=True)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.run_dir.expanduser().resolve()
    if not args.resume and root.exists() and any(root.iterdir()):
        raise SystemExit(f"Refusing to append to an existing run: {root}")
    if args.resume and not (root / "research_design" / "research_design_state.json").is_file():
        raise SystemExit(f"Cannot resume without Research Design state: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if args.resume:
        design_state = load_json_object(
            root / "research_design" / "research_design_state.json"
        )
        design_run_id = str(design_state["run_id"])
        run_id = design_run_id.removesuffix(":research-design")
    else:
        run_id = f"standard-epoch-multisubject-{uuid.uuid4().hex[:10]}"
    contract_path = args.dataset_level_contract.expanduser().resolve()
    envelope_path = args.autonomy_envelope.expanduser().resolve()
    plan_path = args.candidate_plan.expanduser().resolve()
    validation_path = args.validation.expanduser().resolve()
    contract = load_dataset_level_contract(contract_path)
    envelope = load_autonomy_envelope(
        envelope_path,
        expected_dataset_id=str(contract["dataset_id"]),
        expected_dataset_contract_path=contract_path,
    )
    plan = load_json_object(plan_path)
    if (
        plan.get("schema_version") != "1.0"
        or plan.get("status") != "frozen_before_execution"
        or plan.get("dataset_id") != contract["dataset_id"]
    ):
        raise SystemExit("Candidate plan is not a compatible frozen plan")
    candidates = [PipelineSpec.from_dict(item) for item in plan.get("candidates") or []]
    if not candidates or len(candidates) > int(plan["maximum_candidate_configurations"]):
        raise SystemExit("Candidate plan is empty or exceeds its frozen budget")
    subjects = [str(item) for item in args.subjects]
    if len(subjects) != len(set(subjects)) or len(subjects) < 3:
        raise SystemExit("A multi-subject demo requires at least three unique subjects")

    allocation_path = root / "authorities" / "budget_allocation.json"
    if args.resume:
        allocation = load_json_object(allocation_path)
        if allocation.get("subjects") != subjects:
            raise SystemExit("Resume subject list differs from frozen budget allocation")
    else:
        allocation = _allocate_budget(
            envelope=envelope,
            subjects=subjects,
            candidate_count=len(candidates),
        )
        write_json(allocation_path, allocation, refuse_overwrite=True)
    audit = JsonlAuditSink(root / "audit.jsonl", run_id=run_id, resume=args.resume)

    research_design_root = root / "research_design"
    continuation_paths = sorted(research_design_root.glob("budget_continuation-*.json"))
    legacy_extension_path = research_design_root / "budget_extension.json"
    continuation_path = continuation_paths[-1] if continuation_paths else None
    if continuation_path is None and legacy_extension_path.is_file():
        continuation_path = legacy_extension_path
    continuation: dict[str, Any] | None = None
    if args.resume and continuation_path is not None:
        continuation = load_json_object(continuation_path)
        continuation_id = str(
            continuation.get("extension_id")
            or continuation.get("continuation_id")
            or continuation_path.stem
        )
        if continuation_path == legacy_extension_path:
            continuation_ledger_path = research_design_root / "budget_extension_ledger.jsonl"
        else:
            continuation_ledger_path = continuation_path.with_name(
                f"{continuation_path.stem}_ledger.jsonl"
            )
        design_ledger = BudgetLedger(
            continuation_ledger_path,
            run_id=f"{run_id}:research-design:{continuation_id}",
            limits=_limits(continuation["additional_limits"]),
            authority_sha256=sha256_path(continuation_path),
            create=not continuation_ledger_path.exists(),
        )
    else:
        design_ledger = BudgetLedger(
            root / "research_design" / "budget_ledger.jsonl",
            run_id=f"{run_id}:research-design",
            limits=_limits(allocation["design"]),
            authority_sha256=sha256_path(allocation_path),
            create=not args.resume,
        )

    def provider_factory(_stage: str, _cycle: int):
        return _raw_provider(max_output_tokens=6144)

    design = ResearchDesignAgent(
        run_id=f"{run_id}:research-design",
        run_dir=root / "research_design",
        dataset_level_contract_path=contract_path,
        autonomy_envelope_path=envelope_path,
        provider_factory=provider_factory,
        budget_ledger=design_ledger,
        pricing=PRICING,
        audit=audit,
        audit_path=audit.path,
        max_revision_cycles=2,
    ).run(resume=args.resume)
    if design.status != "completed" or "frozen_protocol" not in design.artifacts:
        raise RuntimeError(f"Research Design failed: {design.error or design.status}")
    design_ledger.close("completed")
    protocol_path = Path(design.artifacts["frozen_protocol"]["path"])
    protocol = load_json_object(protocol_path)
    search_ids = [str(item) for item in protocol["data_roles"]["pipeline_search_and_lock"]]
    if len(search_ids) != 1:
        raise RuntimeError("Current dataset incumbent executor requires one search session")

    source = MatEpochSource(
        dataset_root=args.dataset_root,
        validation_path=validation_path,
    )
    incumbent_path = root / "dataset_incumbent" / "dataset_pipeline_incumbent.json"
    if incumbent_path.is_file():
        incumbent = load_json_object(incumbent_path)
    else:
        incumbent_ledger = BudgetLedger(
            root / "dataset_incumbent" / "budget_ledger.jsonl",
            run_id=f"{run_id}:dataset-incumbent",
            limits=_limits(allocation["dataset_incumbent"]),
            authority_sha256=sha256_path(allocation_path),
            create=True,
        )
        incumbent = build_dataset_incumbent(
            dataset_id=str(contract["dataset_id"]),
            executors={
                subject_id: DeterministicPipelineExecutor(
                    sessions=[
                        source.load_session(
                            subject_id=subject_id,
                            session_id=search_ids[0],
                        )
                    ]
                )
                for subject_id in subjects
            },
            candidates=candidates,
            primary_metric=str(plan.get("primary_metric", "balanced_accuracy")),
            stability_penalty=float(plan.get("stability_penalty", 0.25)),
            minimum_subjects=int(plan.get("minimum_subjects", 3)),
            minimum_candidates=int(plan.get("minimum_candidates", 2)),
            minimum_personalization_gain=float(
                plan.get("minimum_personalization_gain", 0.03)
            ),
            source_contracts={
                "dataset_level_contract": {
                    "path": str(contract_path),
                    "sha256": sha256_path(contract_path),
                },
                "candidate_plan": {
                    "path": str(plan_path),
                    "sha256": sha256_path(plan_path),
                },
                "frozen_protocol": {
                    "path": str(protocol_path),
                    "sha256": sha256_path(protocol_path),
                },
                "semantic_validation": {
                    "path": str(validation_path),
                    "sha256": sha256_path(validation_path),
                },
            },
            budget_ledger=incumbent_ledger,
        )
        write_json(incumbent_path, incumbent, refuse_overwrite=True)
        incumbent_ledger.close("completed")

    dataset_profile_path = Path(contract["provenance"]["dataset_profile"]["path"])
    capability_path = Path("configs/executable_pipeline_capabilities.v0.json").resolve()
    rows: list[dict[str, Any]] = []
    for subject_id in subjects:
        subject_root = root / "subjects" / subject_id
        if (subject_root / "final_internal_evidence_report.json").is_file():
            rows.append(_summarize_completed_subject(subject_root, subject_id))
            continue
        subject_limits = dict(allocation["per_subject"][subject_id])
        subject_authority_path = allocation_path
        if continuation is not None:
            overrides = continuation.get("per_subject_limit_overrides") or {}
            if subject_id in overrides:
                subject_limits.update(overrides[subject_id])
                subject_authority_path = continuation_path
        ledger = BudgetLedger(
            root / "budgets" / f"subject-{subject_id}.jsonl",
            run_id=f"{run_id}:subject:{subject_id}",
            limits=_limits(subject_limits),
            authority_sha256=sha256_path(subject_authority_path),
            create=True,
        )
        rows.append(
            _run_subject(
                run_id=run_id,
                subject_id=subject_id,
                root=root,
                source=source,
                dataset_profile_path=dataset_profile_path,
                protocol_path=protocol_path,
                envelope_path=envelope_path,
                capability_path=capability_path,
                audit=audit,
                ledger=ledger,
                dataset_incumbent_path=incumbent_path,
            )
        )
        ledger.close("completed")

    summary = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "completed",
        "demo_scope": "real_multisubject_internal_evidence",
        "external_scientific_claim_authorized": False,
        "dataset_pipeline_incumbent": {
            "path": str(incumbent_path),
            "sha256": sha256_path(incumbent_path),
            "selected_pipeline": incumbent["selected_pipeline"],
            "score_summary": incumbent["selected_score_summary"],
        },
        "subjects": rows,
        "route_counts": {
            mode: sum(
                1
                for subject_id in subjects
                if (
                    load_json_object(root / "subjects" / subject_id / "pipeline_lock.json").get(
                        "route_decision"
                    )
                    or {}
                ).get("mode")
                == mode
            )
            for mode in ("personalized", "fallback_to_dataset_incumbent")
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = root / "multi_subject_summary.json"
    write_json(summary_path, summary, refuse_overwrite=True)
    write_json(
        root / "run_manifest.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "completed",
            "dataset_level_contract": {
                "path": str(contract_path),
                "sha256": sha256_path(contract_path),
            },
            "autonomy_envelope": {
                "path": str(envelope_path),
                "sha256": sha256_path(envelope_path),
            },
            "frozen_protocol": {
                "path": str(protocol_path),
                "sha256": sha256_path(protocol_path),
            },
            "dataset_incumbent": {
                "path": str(incumbent_path),
                "sha256": sha256_path(incumbent_path),
            },
            "summary": {"path": str(summary_path), "sha256": sha256_path(summary_path)},
            "audit": {"path": str(audit.path), "sha256": sha256_path(audit.path)},
            "external_scientific_claim_authorized": False,
        },
        refuse_overwrite=True,
    )
    print(f"completed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
