"""Resume a Dataset-Level Literature revision using only prior search evidence."""

from __future__ import annotations

import argparse
import shutil
import uuid
from pathlib import Path
from typing import Any

from bci_autodiscovery.literature import LiteratureSourceError, LiteratureStore
from bci_autodiscovery.search import build_combined_search_space_review
from bci_autodiscovery.workflow.autonomy import load_json_object, sha256_path
from bci_autodiscovery.workflow.dataset_intelligence import (
    freeze_dataset_level_contract,
)
from bci_autodiscovery.workflow.protocol_artifacts import atomic_json

from .audit import JsonlAuditSink
from .dataset_critic_cli import _provider
from .dataset_intelligence_critic import (
    DatasetIntelligenceCriticAgent,
    create_dataset_intelligence_critic_tools,
)
from .dataset_level_agent import _frontier_semantic_hash
from .literature_scout import LiteratureScoutAgent, create_literature_scout_tools
from .run_recovery import AgentRecoveryError, assert_source_run_recoverable
from .runtime import AgentRuntime, RuntimeLimits


class _ReusedEvidenceOnlySource:
    def __init__(self, name: str) -> None:
        self.name = name

    def search(self, _query: Any) -> Any:
        raise LiteratureSourceError(
            f"Network access is disabled in evidence-only revision: {self.name}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Revise Literature Scout directions from an immutable critique while "
            "reusing the prior search ledger and forbidding new network searches."
        )
    )
    parser.add_argument("--source-draft", type=Path, required=True)
    parser.add_argument("--revision-critique", type=Path, required=True)
    parser.add_argument("--completed-revision-evidence", type=Path)
    parser.add_argument("--completed-literature-run-id")
    parser.add_argument("--provider", choices=["deepseek", "kimi"], required=True)
    parser.add_argument("--model")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--provider-timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--reasoning-effort", choices=["low", "high", "max"], default="low"
    )
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_output_tokens is not None and not 1 <= args.max_output_tokens <= 32768:
        raise SystemExit("--max-output-tokens must be between 1 and 32768")
    if not 30 <= args.provider_timeout_seconds <= 600:
        raise SystemExit("--provider-timeout-seconds must be between 30 and 600")
    if bool(args.completed_revision_evidence) != bool(
        args.completed_literature_run_id
    ):
        raise SystemExit(
            "--completed-revision-evidence and --completed-literature-run-id "
            "must be provided together"
        )

    source_draft_path = args.source_draft.expanduser().resolve()
    critique_path = args.revision_critique.expanduser().resolve()
    try:
        assert_source_run_recoverable(source_draft_path)
    except AgentRecoveryError as exc:
        raise SystemExit(str(exc)) from exc
    source_draft = load_json_object(source_draft_path)
    critique = load_json_object(critique_path)
    source_ref = critique.get("source_draft") or {}
    if (
        critique.get("verdict") != "revise"
        or Path(str(source_ref.get("path") or "")).expanduser().resolve()
        != source_draft_path
        or str(source_ref.get("sha256") or "") != sha256_path(source_draft_path)
    ):
        raise SystemExit("Revision Critique is not bound to the exact source draft")

    provenance = source_draft.get("provenance") or {}
    canonical_ref = provenance.get("canonical_search_space") or {}
    evidence_ref = provenance.get("evidence_db") or {}
    canonical_path = Path(str(canonical_ref.get("path") or "")).expanduser().resolve()
    source_evidence_path = Path(
        str(evidence_ref.get("path") or "")
    ).expanduser().resolve()
    if (
        not canonical_path.is_file()
        or sha256_path(canonical_path) != canonical_ref.get("sha256")
        or not source_evidence_path.is_file()
        or sha256_path(source_evidence_path) != evidence_ref.get("sha256")
    ):
        raise SystemExit("Source draft provenance failed integrity verification")
    source_literature_run_id = str(
        (source_draft.get("frontier_space") or {}).get("literature_run_id") or ""
    )
    if not source_literature_run_id:
        raise SystemExit("Source draft lacks literature_run_id")

    run_id = args.run_id or f"dataset-literature-revision-{uuid.uuid4().hex[:12]}"
    run_dir = (args.run_dir or Path("artifacts") / "runs" / run_id).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Refusing to append to existing immutable run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_path = run_dir / "audit.jsonl"
    audit = JsonlAuditSink(audit_path, run_id=run_id)
    evidence_path = run_dir / "evidence.sqlite"
    completed_revision_evidence = args.completed_revision_evidence
    if completed_revision_evidence is not None:
        completed_revision_evidence = completed_revision_evidence.expanduser().resolve()
        if not completed_revision_evidence.is_file():
            raise SystemExit(
                f"Completed revision evidence does not exist: {completed_revision_evidence}"
            )
        shutil.copy2(completed_revision_evidence, evidence_path)
        target_literature_run_id = str(args.completed_literature_run_id)
        audit.record(
            "completed_literature_revision_recovered",
            {
                "source_revision_evidence_path": str(completed_revision_evidence),
                "source_revision_evidence_sha256": sha256_path(
                    completed_revision_evidence
                ),
                "target_evidence_path": str(evidence_path),
                "literature_run_id": target_literature_run_id,
                "network_access_allowed": False,
            },
        )
    else:
        shutil.copy2(source_evidence_path, evidence_path)
        target_literature_run_id = f"{run_id}:literature-revision"
        clone = LiteratureStore(evidence_path).clone_search_evidence(
            source_search_run_id=source_literature_run_id,
            target_search_run_id=target_literature_run_id,
        )
        audit.record(
            "literature_evidence_reused",
            {
                "source_draft_path": str(source_draft_path),
                "source_evidence_path": str(source_evidence_path),
                "source_literature_run_id": source_literature_run_id,
                "target_evidence_path": str(evidence_path),
                "target_literature_run_id": target_literature_run_id,
                "network_access_allowed": False,
                **clone,
            },
        )

    literature_tools, literature_context = create_literature_scout_tools(
        search_space_path=canonical_path,
        evidence_db_path=evidence_path,
        search_run_id=target_literature_run_id,
        sources=(
            _ReusedEvidenceOnlySource("crossref"),
            _ReusedEvidenceOnlySource("openalex"),
        ),
        revision_critique_path=critique_path,
    )
    literature_result = None
    if completed_revision_evidence is None:
        literature_result = LiteratureScoutAgent(
            runtime=AgentRuntime(
                provider=_provider(args),
                tools=literature_tools,
                audit=audit,
                limits=RuntimeLimits(max_iterations=12, max_tool_calls=12),
                run_id=f"{run_id}:literature-scout",
                audit_path=str(audit_path),
            ),
            context=literature_context,
        ).run()
        atomic_json(
            run_dir / "literature_agent_result.json", literature_result.to_dict()
        )
    frontier_status = literature_tools.execute(
        "inspect_frontier_discovery_status", {}
    )
    atomic_json(run_dir / "frontier_discovery.json", frontier_status)
    if (
        literature_result is not None
        and literature_result.status != "completed"
    ) or not frontier_status["complete"]:
        print(f"run_id: {run_id}")
        print(
            "status: "
            + (literature_result.status if literature_result is not None else "failed")
        )
        print(f"run_dir: {run_dir}")
        print(
            "error: "
            + (
                literature_result.error
                if literature_result is not None and literature_result.error
                else "Literature revision incomplete"
            )
        )
        return 1

    revised_draft = build_combined_search_space_review(
        canonical_search_space_path=canonical_path,
        evidence_db_path=evidence_path,
        literature_run_id=target_literature_run_id,
    )
    if _frontier_semantic_hash(revised_draft) == _frontier_semantic_hash(source_draft):
        raise SystemExit("Literature revision did not change criticized frontier semantics")
    revised_draft_path = run_dir / "dataset_level_draft.json"
    atomic_json(revised_draft_path, revised_draft)

    critic_tools, critic_context = create_dataset_intelligence_critic_tools(
        dataset_level_draft_path=revised_draft_path
    )
    critic_result = DatasetIntelligenceCriticAgent(
        runtime=AgentRuntime(
            provider=_provider(args),
            tools=critic_tools,
            audit=audit,
            limits=RuntimeLimits(max_iterations=6, max_tool_calls=4),
            run_id=f"{run_id}:dataset-critic",
            audit_path=str(audit_path),
        ),
        context=critic_context,
    ).run()
    atomic_json(run_dir / "critic_agent_result.json", critic_result.to_dict())
    revised_critique = critic_result.latest_tool_result("record_dataset_critique")
    if critic_result.status != "completed" or revised_critique is None:
        print(f"run_id: {run_id}")
        print(f"status: {critic_result.status}")
        print(f"run_dir: {run_dir}")
        print(f"error: {critic_result.error or 'Dataset Critic incomplete'}")
        return 1
    revised_critique_path = run_dir / "dataset_critique.json"
    atomic_json(revised_critique_path, revised_critique)
    verdict = str(revised_critique["verdict"])
    contract_path: Path | None = None
    if verdict == "pass":
        contract_path = run_dir / "dataset_level_contract.json"
        freeze_dataset_level_contract(
            dataset_level_draft_path=revised_draft_path,
            dataset_critique_path=revised_critique_path,
            output_path=contract_path,
        )

    print(f"run_id: {run_id}")
    print("status: completed")
    print(f"verdict: {verdict}")
    print(f"run_dir: {run_dir}")
    print("network_searches_repeated: 0")
    if contract_path is not None:
        print(f"dataset_level_contract: {contract_path}")
    return 0 if verdict == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
