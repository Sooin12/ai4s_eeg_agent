"""Run the polished multi-subject autonomous BCI engineering demonstration."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bci_autodiscovery.agents.audit import JsonlAuditSink
from bci_autodiscovery.agents.budgeted_provider import BudgetedProvider, TokenPricing
from bci_autodiscovery.agents.evidence_reporter import (
    EvidenceReporterAgent,
    create_evidence_reporter_tools,
)
from bci_autodiscovery.agents.pipeline_lock_critic import (
    PipelineLockCriticAgent,
    create_pipeline_lock_critic_tools,
)
from bci_autodiscovery.agents.pipeline_search import (
    PipelineSearchAgent,
    create_pipeline_search_tools,
)
from bci_autodiscovery.agents.providers import OpenAICompatibleProvider
from bci_autodiscovery.agents.research_design_agent import ResearchDesignAgent
from bci_autodiscovery.agents.runtime import AgentRuntime, RuntimeLimits
from bci_autodiscovery.agents.scientific_critic import (
    ScientificCriticAgent,
    create_scientific_critic_tools,
)
from bci_autodiscovery.agents.subject_profiler import (
    SubjectProfilerAgent,
    create_subject_profiler_tools,
)
from bci_autodiscovery.demo.contracts import build_demo_dataset_contract, write_json
from bci_autodiscovery.demo.synthetic import (
    DemoNpzEpochSource,
    SUBJECT_PHENOTYPES,
    write_synthetic_sessions,
)
from bci_autodiscovery.demo.presentation import build_presentation_bundle
from bci_autodiscovery.evaluation.confirmation import OneShotConfirmationController
from bci_autodiscovery.pipelines import DeterministicPipelineExecutor
from bci_autodiscovery.profiling.subject_measurements import (
    SubjectEpochSource,
    SubjectMeasurementEngine,
)
from bci_autodiscovery.reporting import finalize_internal_evidence_report
from bci_autodiscovery.workflow.autonomy import load_json_object, sha256_path
from bci_autodiscovery.workflow.budget import BudgetLedger, limits_from_envelope


PRICING = TokenPricing(
    currency="USD",
    prompt_per_million=3.0,
    completion_per_million=15.0,
    cached_prompt_per_million=0.3,
    source="https://www.kimi.com/help/kimi-api/api-pricing (checked 2026-08-15)",
)

_LAST_PROVIDER_REQUEST_AT = 0.0
_MINIMUM_PROVIDER_REQUEST_INTERVAL_SECONDS = 21.0


def _progress(event: str, payload: dict[str, Any]) -> None:
    if event in {"model_request", "tool_requested", "tool_completed", "tool_failed", "run_completed", "run_failed"}:
        detail = payload.get("tool_name") or payload.get("iteration") or payload.get("error") or ""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {event}: {detail}", flush=True)


def _raw_provider(*, max_output_tokens: int) -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider.kimi(
        model="kimi-k3",
        max_output_tokens=max_output_tokens,
        reasoning_effort="low",
    )
    provider.timeout_seconds = 240.0
    provider.progress_callback = _progress
    original_complete = provider.complete

    def rate_limited_complete(*, messages: Any, tools: Any) -> Any:
        global _LAST_PROVIDER_REQUEST_AT
        wait_seconds = (
            _MINIMUM_PROVIDER_REQUEST_INTERVAL_SECONDS
            - (time.monotonic() - _LAST_PROVIDER_REQUEST_AT)
        )
        if wait_seconds > 0:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] provider_rate_limit_wait: "
                f"{wait_seconds:.1f}s",
                flush=True,
            )
            time.sleep(wait_seconds)
        _LAST_PROVIDER_REQUEST_AT = time.monotonic()
        return original_complete(messages=messages, tools=tools)

    provider.complete = rate_limited_complete  # type: ignore[method-assign]
    return provider


def _runtime(
    *,
    run_id: str,
    stage: str,
    tools: Any,
    audit: JsonlAuditSink,
    ledger: BudgetLedger,
    max_iterations: int,
    max_tool_calls: int,
    max_output_tokens: int = 6144,
) -> AgentRuntime:
    provider = BudgetedProvider(
        provider=_raw_provider(max_output_tokens=max_output_tokens),
        ledger=ledger,
        pricing=PRICING,
        stage=stage,
    )
    return AgentRuntime(
        provider=provider,
        tools=tools,
        audit=audit,
        limits=RuntimeLimits(
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            max_repeated_identical_calls=2,
        ),
        run_id=f"{run_id}:{stage}",
        audit_path=str(audit.path),
        progress_callback=_progress,
    )


def _require_artifact(result: Any, tool_name: str, *, stage: str) -> dict[str, Any]:
    artifact = result.latest_tool_result(tool_name)
    if artifact is None:
        raise RuntimeError(f"{stage} failed: {result.error or 'required artifact missing'}")
    if result.status != "completed":
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] terminal_tool_artifact_recovered: "
            f"{stage} ({result.status})",
            flush=True,
        )
    return artifact


def _load_sessions(source: SubjectEpochSource, subject_id: str, ids: list[str]) -> list[Any]:
    return [source.load_session(subject_id=subject_id, session_id=item) for item in ids]


def _run_subject(
    *,
    run_id: str,
    subject_id: str,
    root: Path,
    source: SubjectEpochSource,
    dataset_profile_path: Path,
    protocol_path: Path,
    envelope_path: Path,
    capability_path: Path,
    audit: JsonlAuditSink,
    ledger: BudgetLedger,
    dataset_incumbent_path: Path | None = None,
) -> dict[str, Any]:
    subject_root = root / "subjects" / subject_id
    subject_root.mkdir(parents=True, exist_ok=False)
    protocol = load_json_object(protocol_path)
    profiling_ids = [str(item) for item in protocol["data_roles"]["profiling_and_calibration"]]
    search_ids = [str(item) for item in protocol["data_roles"]["pipeline_search_and_lock"]]
    confirmation_ids = [str(item) for item in protocol["data_roles"]["frozen_confirmation"]]

    print(f"\n=== {subject_id}: Subject Profiler ===", flush=True)
    engine = SubjectMeasurementEngine(
        source=source,
        subject_id=subject_id,
        allowed_session_ids=profiling_ids,
    )
    tools, context = create_subject_profiler_tools(
        engine=engine,
        dataset_profile_path=dataset_profile_path,
        frozen_protocol_path=protocol_path,
    )
    profile_result = SubjectProfilerAgent(
        runtime=_runtime(
            run_id=run_id,
            stage=f"{subject_id}:subject_profiler",
            tools=tools,
            audit=audit,
            ledger=ledger,
            max_iterations=9,
            max_tool_calls=10,
        ),
        context=context,
    ).run()
    subject_profile = _require_artifact(
        profile_result, "record_subject_profile", stage=f"{subject_id} Subject Profiler"
    )
    subject_profile_path = subject_root / "subject_profile.json"
    write_json(subject_profile_path, subject_profile, refuse_overwrite=True)

    print(f"\n=== {subject_id}: literature + individualized pipeline search ===", flush=True)
    executor = DeterministicPipelineExecutor(
        sessions=_load_sessions(source, subject_id, search_ids)
    )
    literature_store_path = subject_root / "literature_evidence.sqlite"
    tools, context = create_pipeline_search_tools(
        executor=executor,
        subject_profile_path=subject_profile_path,
        frozen_protocol_path=protocol_path,
        autonomy_envelope_path=envelope_path,
        capability_registry_path=capability_path,
        literature_store_path=literature_store_path,
        literature_search_run_id=f"{run_id}-{subject_id}-method-evidence",
        budget_ledger=ledger,
        dataset_incumbent_path=dataset_incumbent_path,
    )
    search_result = PipelineSearchAgent(
        runtime=_runtime(
            run_id=run_id,
            stage=f"{subject_id}:pipeline_search",
            tools=tools,
            audit=audit,
            ledger=ledger,
            max_iterations=14,
            max_tool_calls=20,
        ),
        context=context,
    ).run()
    pipeline_lock = _require_artifact(
        search_result, "lock_pipeline", stage=f"{subject_id} Pipeline Search"
    )
    lock_path = subject_root / "pipeline_lock.json"
    write_json(lock_path, pipeline_lock, refuse_overwrite=True)

    print(f"\n=== {subject_id}: outcome-blind Pipeline Lock Critic ===", flush=True)
    tools, context = create_pipeline_lock_critic_tools(
        pipeline_lock_path=lock_path,
        subject_profile_path=subject_profile_path,
        frozen_protocol_path=protocol_path,
        autonomy_envelope_path=envelope_path,
    )
    critic_result = PipelineLockCriticAgent(
        runtime=_runtime(
            run_id=run_id,
            stage=f"{subject_id}:pipeline_lock_critic",
            tools=tools,
            audit=audit,
            ledger=ledger,
            max_iterations=5,
            max_tool_calls=4,
            max_output_tokens=4096,
        ),
        context=context,
    ).run()
    lock_critique = _require_artifact(
        critic_result,
        "record_pipeline_lock_critique",
        stage=f"{subject_id} Pipeline Lock Critic",
    )
    if lock_critique.get("verdict") != "pass":
        raise RuntimeError(f"{subject_id} lock critic verdict: {lock_critique.get('verdict')}")
    lock_critique_path = subject_root / "pipeline_lock_critique.json"
    write_json(lock_critique_path, lock_critique, refuse_overwrite=True)

    print(f"\n=== {subject_id}: one-shot frozen confirmation ===", flush=True)
    confirmation_controller = OneShotConfirmationController(
        search_executor=executor,
        confirmation_loader=lambda: _load_sessions(source, subject_id, confirmation_ids),
        pipeline_lock_path=lock_path,
        lock_critique_path=lock_critique_path,
        frozen_protocol_path=protocol_path,
        autonomy_envelope_path=envelope_path,
        access_record_path=subject_root / "confirmation_access.json",
        confirmation_result_path=subject_root / "confirmation_result.json",
        budget_ledger=ledger,
    )
    confirmation_controller.confirm()
    return _complete_subject_reporting(
        run_id=run_id,
        subject_id=subject_id,
        subject_root=subject_root,
        protocol_path=protocol_path,
        envelope_path=envelope_path,
        audit=audit,
        ledger=ledger,
    )


def _complete_subject_reporting(
    *,
    run_id: str,
    subject_id: str,
    subject_root: Path,
    protocol_path: Path,
    envelope_path: Path,
    audit: JsonlAuditSink,
    ledger: BudgetLedger,
) -> dict[str, Any]:
    print(f"\n=== {subject_id}: evidence report + independent Scientific Critic ===", flush=True)
    subject_profile_path = subject_root / "subject_profile.json"
    lock_path = subject_root / "pipeline_lock.json"
    lock_critique_path = subject_root / "pipeline_lock_critique.json"
    tools, context = create_evidence_reporter_tools(
        subject_profile_path=subject_profile_path,
        pipeline_lock_path=lock_path,
        lock_critique_path=lock_critique_path,
        confirmation_result_path=subject_root / "confirmation_result.json",
        frozen_protocol_path=protocol_path,
        autonomy_envelope_path=envelope_path,
    )
    report_result = EvidenceReporterAgent(
        runtime=_runtime(
            run_id=run_id,
            stage=f"{subject_id}:evidence_reporter",
            tools=tools,
            audit=audit,
            ledger=ledger,
            max_iterations=6,
            max_tool_calls=6,
        ),
        context=context,
    ).run()
    report = _require_artifact(
        report_result, "record_evidence_report", stage=f"{subject_id} Evidence Reporter"
    )
    report_path = subject_root / "evidence_report.json"
    write_json(report_path, report, refuse_overwrite=True)

    tools, context = create_scientific_critic_tools(evidence_report_path=report_path)
    scientific_result = ScientificCriticAgent(
        runtime=_runtime(
            run_id=run_id,
            stage=f"{subject_id}:scientific_critic",
            tools=tools,
            audit=audit,
            ledger=ledger,
            max_iterations=5,
            max_tool_calls=4,
            max_output_tokens=4096,
        ),
        context=context,
    ).run()
    scientific_critique = _require_artifact(
        scientific_result,
        "record_scientific_critique",
        stage=f"{subject_id} Scientific Critic",
    )
    if scientific_critique.get("verdict") != "pass":
        raise RuntimeError(
            f"{subject_id} scientific critic verdict: {scientific_critique.get('verdict')}"
        )
    scientific_path = subject_root / "scientific_critique.json"
    write_json(scientific_path, scientific_critique, refuse_overwrite=True)
    finalize_internal_evidence_report(
        evidence_report_path=report_path,
        scientific_critique_path=scientific_path,
        final_report_path=subject_root / "final_internal_evidence_report.json",
    )
    return _summarize_completed_subject(subject_root, subject_id)


def _write_summary(root: Path, run_id: str, rows: list[dict[str, Any]], ledger: BudgetLedger) -> None:
    families = sorted({row["family"] for row in rows})
    configurations = {
        (row["family"], tuple(row["bandpass_hz"]), tuple(row["selected_channels"]))
        for row in rows
    }
    summary = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "completed",
        "demo_scope": "synthetic_engineering_validation_only",
        "external_scientific_claim_authorized": False,
        "subjects": rows,
        "personalization": {
            "distinct_families": families,
            "distinct_locked_configurations": len(configurations),
            "all_locks_independently_passed": all(row["lock_critic"] == "pass" for row in rows),
            "all_reports_independently_passed": all(row["scientific_critic"] == "pass" for row in rows),
            "all_confirmations_one_shot": all(row["confirmation_access_count"] == 1 for row in rows),
        },
        "budget": ledger.snapshot(),
    }
    write_json(root / "demo_summary.json", summary, refuse_overwrite=True)
    lines = [
        "# 全线路自动化个体化 BCI 科研 Agent Demo",
        "",
        "> 这是确定性合成 EEG 上的工程验证，不是对真实受试者或算法疗效的外部科学声明。",
        "",
        "| 被试 | 隐含工程表型 | Agent 锁定 pipeline | 频带 (Hz) | 通道 | 搜索 BA | 冻结确认 BA | 结论 |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        channels = ", ".join(row["selected_channels"]) or "all"
        lines.append(
            f"| {row['subject_id']} | {row['synthetic_phenotype']} | {row['family']} | "
            f"{row['bandpass_hz']} | {channels} | {row['search_score']:.3f} | "
            f"{row['confirmation_score']:.3f} | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            "## 可审计闭环",
            "",
            "每名被试均依次完成：无原始数组暴露的画像测量 → Agent 自主联网查文献 → "
            "预算内完整 pipeline 实验 → 独立盲审锁定 → 一次性冻结确认 → 证据报告 → 独立科学审查。",
            "",
            f"- 独立锁定配置数：{len(configurations)}",
            f"- Pipeline Lock Critic 全部通过：{all(row['lock_critic'] == 'pass' for row in rows)}",
            f"- Scientific Critic 全部通过：{all(row['scientific_critic'] == 'pass' for row in rows)}",
            "- 确认数据访问次数：每名被试恰好 1 次；确认后未重拟合、未重开搜索。",
            "- 所有工具参数、模型调用、错误和 token/费用均写入 append-only 审计与预算账本。",
        ]
    )
    (root / "DEMO_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summarize_completed_subject(subject_root: Path, subject_id: str) -> dict[str, Any]:
    lock = load_json_object(subject_root / "pipeline_lock.json")
    confirmation = load_json_object(subject_root / "confirmation_result.json")
    access = load_json_object(subject_root / "confirmation_access.json")
    lock_critique = load_json_object(subject_root / "pipeline_lock_critique.json")
    scientific = load_json_object(subject_root / "scientific_critique.json")
    final = load_json_object(subject_root / "final_internal_evidence_report.json")
    selected = lock["selected_pipeline"]
    return {
        "subject_id": subject_id,
        "synthetic_phenotype": SUBJECT_PHENOTYPES.get(subject_id, "real_subject"),
        "selected_pipeline_id": selected["pipeline_id"],
        "family": selected["family"],
        "bandpass_hz": selected["bandpass_hz"],
        "channel_strategy": selected.get("channel_strategy", "all"),
        "selected_channels": selected.get("selected_channels", []),
        "search_score": lock["selected_search_score"],
        "confirmation_score": confirmation["confirmation_score"],
        "confirmation_minus_search": confirmation["confirmation_minus_search"],
        "decision": final["deterministic_decision"]["outcome"],
        "research_cycles": lock["budget_usage"]["research_cycles"],
        "literature_paper_ids": lock.get("evidence_literature_paper_ids", []),
        "lock_critic": lock_critique["verdict"],
        "scientific_critic": scientific["verdict"],
        "confirmation_access_count": access["access_count"],
        "artifact_root": str(subject_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--run-dir", type=Path)
    destination.add_argument("--resume-run-dir", type=Path)
    parser.add_argument(
        "--authorized-api-token-limit",
        type=int,
        help="User-authorized total API token ceiling for a scoped budget extension.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_id = "synthetic-heterogeneous-mi-demo"
    resume = args.resume_run_dir is not None
    if resume:
        root = args.resume_run_dir.expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"Resume run directory does not exist: {root}")
        first_event = json.loads(
            (root / "budget_ledger.jsonl").read_text(encoding="utf-8").splitlines()[0]
        )
        run_id = str(first_event["run_id"])
        contract_path = root / "authorities" / "dataset_level_contract.json"
        envelope_path = root / "authorities" / "autonomy_envelope.json"
    else:
        run_id = f"individualized-demo-{uuid.uuid4().hex[:10]}"
        root = (args.run_dir or Path("artifacts") / "runs" / run_id).expanduser().resolve()
        if root.exists() and any(root.iterdir()):
            raise SystemExit(f"Refusing to append to existing demo run: {root}")
        root.mkdir(parents=True, exist_ok=True)
        contract_path, envelope_path = build_demo_dataset_contract(
            root / "authorities", dataset_id=dataset_id
        )
    envelope = load_json_object(envelope_path)
    if not resume:
        write_synthetic_sessions(root / "synthetic_data")
    source = DemoNpzEpochSource(root / "synthetic_data")
    audit = JsonlAuditSink(root / "audit.jsonl", run_id=run_id, resume=resume)
    base_ledger = BudgetLedger(
        root / "budget_ledger.jsonl",
        run_id=run_id,
        limits=limits_from_envelope(envelope),
        authority_sha256=sha256_path(envelope_path),
        create=not resume,
    )
    ledger = base_ledger
    extension_path = root / "budget_extension.json"
    extension_ledger_path = root / "budget_extension_ledger.jsonl"
    requested_total = args.authorized_api_token_limit
    original_total = int(envelope["resource_budget"]["max_api_tokens"])
    if requested_total is not None and requested_total < original_total:
        raise SystemExit("Authorized API token limit cannot reduce the frozen original limit")
    if resume and (extension_path.exists() or (requested_total or 0) > original_total):
        if not extension_path.exists():
            base_snapshot = base_ledger.snapshot()
            extension_total = int(requested_total) - original_total
            if extension_total <= 0:
                raise SystemExit("A positive authorized token extension is required")
            write_json(
                extension_path,
                {
                    "schema_version": "1.0",
                    "extension_id": f"{run_id}-api-token-extension-001",
                    "status": "operator_authorized_scoped_extension",
                    "authorized_at_utc": datetime.now(timezone.utc).isoformat(),
                    "authorization_record": (
                        "User explicitly approved raising total API tokens from "
                        f"{original_total} to {requested_total} in the active Codex task."
                    ),
                    "scope": "complete subject-beta and aggregate the existing demo only",
                    "original_autonomy_envelope": {
                        "path": str(envelope_path),
                        "sha256": sha256_path(envelope_path),
                        "envelope_id": envelope["envelope_id"],
                    },
                    "base_budget_ledger": {
                        "path": str(base_ledger.path),
                        "sha256_at_extension": sha256_path(base_ledger.path),
                        "snapshot": base_snapshot,
                    },
                    "changes": {
                        "max_api_tokens": {
                            "previous_total": original_total,
                            "authorized_total": int(requested_total),
                            "extension": extension_total,
                        },
                        "max_paid_cost": {
                            "previous_total": envelope["resource_budget"]["max_paid_cost"],
                            "authorized_total": envelope["resource_budget"]["max_paid_cost"],
                            "extension": 0.0,
                        },
                    },
                    "unchanged": [
                        "research objective",
                        "dataset and subject scope",
                        "frozen protocol and thresholds",
                        "confirmation access policy",
                        "forbidden actions",
                    ],
                    "confirmation_reopen_authorized": False,
                    "completed_subject_rerun_authorized": False,
                },
                refuse_overwrite=True,
            )
        extension = load_json_object(extension_path)
        authorized_total = int(extension["changes"]["max_api_tokens"]["authorized_total"])
        if requested_total is not None and requested_total != authorized_total:
            raise SystemExit("Requested token limit differs from the recorded budget extension")
        base_snapshot = extension["base_budget_ledger"]["snapshot"]
        remaining = base_snapshot["remaining"]
        extension_limits = {
            "research_cycles": float(remaining["research_cycles"]),
            "candidate_executions": float(remaining["candidate_executions"]),
            "compute_seconds": float(remaining["compute_seconds"]),
            "api_total_tokens": float(extension["changes"]["max_api_tokens"]["extension"]),
            "paid_cost": float(remaining["paid_cost"]),
            "provider_retries": float(remaining["provider_retries"]),
            "recovery_attempts": float(remaining["recovery_attempts"]),
            "confirmation_accesses": float(remaining["confirmation_accesses"]),
        }
        ledger = BudgetLedger(
            extension_ledger_path,
            run_id=f"{run_id}:budget-extension-001",
            limits=extension_limits,
            authority_sha256=sha256_path(extension_path),
            create=not extension_ledger_path.exists(),
        )

    design_root = root / "research_design"

    def provider_factory(stage: str, cycle: int) -> OpenAICompatibleProvider:
        del stage, cycle
        return _raw_provider(max_output_tokens=6144)

    if resume:
        ledger.record_recovery(source_run_id=run_id)
        protocol_path = design_root / "frozen_protocol.json"
        if not protocol_path.is_file():
            raise RuntimeError("Demo resume currently requires a completed frozen research design")
        print("=== Resuming after validated frozen Research Design ===", flush=True)
    else:
        print("=== Research Design Agent: Planner -> independent Critic -> freeze ===", flush=True)
        design = ResearchDesignAgent(
            run_id=f"{run_id}:research_design",
            run_dir=design_root,
            dataset_level_contract_path=contract_path,
            autonomy_envelope_path=envelope_path,
            provider_factory=provider_factory,
            budget_ledger=ledger,
            pricing=PRICING,
            audit=audit,
            audit_path=audit.path,
            max_revision_cycles=2,
        ).run()
        if design.status != "completed" or "frozen_protocol" not in design.artifacts:
            raise RuntimeError(f"Research Design failed: {design.error or design.status}")
        protocol_path = Path(design.artifacts["frozen_protocol"]["path"])
    contract = load_json_object(contract_path)
    dataset_profile_path = Path(contract["provenance"]["dataset_profile"]["path"])
    capability_path = Path("configs/executable_pipeline_capabilities.v0.json").resolve()
    rows = []
    for subject_id in SUBJECT_PHENOTYPES:
        subject_root = root / "subjects" / subject_id
        if (subject_root / "final_internal_evidence_report.json").is_file():
            rows.append(_summarize_completed_subject(subject_root, subject_id))
            continue
        if (subject_root / "confirmation_result.json").is_file():
            rows.append(
                _complete_subject_reporting(
                    run_id=run_id,
                    subject_id=subject_id,
                    subject_root=subject_root,
                    protocol_path=protocol_path,
                    envelope_path=envelope_path,
                    audit=audit,
                    ledger=ledger,
                )
            )
            continue
        if subject_root.exists():
            if any(subject_root.iterdir()):
                raise RuntimeError(
                    f"Refusing ambiguous partial subject recovery; inspect {subject_root}"
                )
            subject_root.rmdir()
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
            )
        )
    _write_summary(root, run_id, rows, ledger)
    write_json(
        root / "run_manifest.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_level_contract": {"path": str(contract_path), "sha256": sha256_path(contract_path)},
            "autonomy_envelope": {"path": str(envelope_path), "sha256": sha256_path(envelope_path)},
            "frozen_protocol": {"path": str(protocol_path), "sha256": sha256_path(protocol_path)},
            "audit": {"path": str(audit.path), "sha256": sha256_path(audit.path)},
            "budget_ledger": {"path": str(ledger.path), "sha256": sha256_path(ledger.path)},
            "demo_summary": {"path": str(root / 'demo_summary.json'), "sha256": sha256_path(root / 'demo_summary.json')},
            "external_scientific_claim_authorized": False,
        },
        refuse_overwrite=True,
    )
    presentation_json, presentation_markdown = build_presentation_bundle(root)
    print(f"\nDemo completed: {root}", flush=True)
    print(f"Presentation report: {presentation_markdown}", flush=True)
    print(f"Presentation evidence: {presentation_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
