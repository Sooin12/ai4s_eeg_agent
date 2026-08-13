"""Resume an interrupted Dataset-Level run at the independent Critic gate."""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime
from pathlib import Path

from bci_autodiscovery.workflow.dataset_intelligence import (
    freeze_dataset_level_contract,
)
from bci_autodiscovery.workflow.protocol_artifacts import atomic_json

from .audit import JsonlAuditSink
from .dataset_intelligence_critic import (
    DatasetIntelligenceCriticAgent,
    create_dataset_intelligence_critic_tools,
)
from .providers import OpenAICompatibleProvider
from .run_recovery import AgentRecoveryError, assert_source_run_recoverable
from .runtime import AgentRuntime, RuntimeLimits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resume an immutable Dataset-Level draft at the independent Critic gate "
            "without repeating profiling or literature searches."
        )
    )
    parser.add_argument("--dataset-level-draft", type=Path, required=True)
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


def _provider(args: argparse.Namespace) -> OpenAICompatibleProvider:
    if args.provider == "kimi":
        model = args.model or "kimi-k2.7-code"
        provider = OpenAICompatibleProvider.kimi(
            model=model,
            max_output_tokens=args.max_output_tokens
            or (16384 if model.startswith("kimi-k2.7") else 4096),
            reasoning_effort=args.reasoning_effort,
        )
    else:
        provider = OpenAICompatibleProvider.deepseek(
            model=args.model or "deepseek-v4-flash",
            max_output_tokens=args.max_output_tokens or 4096,
        )
    provider.timeout_seconds = args.provider_timeout_seconds

    def progress(event: str, payload: dict) -> None:
        if event not in {
            "model_request",
            "tool_requested",
            "tool_completed",
            "tool_failed",
            "run_completed",
            "run_failed",
        }:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        detail = payload.get("tool_name") or payload.get("iteration") or payload.get(
            "error"
        )
        print(f"[{stamp}] {event}: {detail or ''}", flush=True)

    provider.progress_callback = progress
    return provider


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_output_tokens is not None and not 1 <= args.max_output_tokens <= 32768:
        raise SystemExit("--max-output-tokens must be between 1 and 32768")
    if not 30 <= args.provider_timeout_seconds <= 600:
        raise SystemExit("--provider-timeout-seconds must be between 30 and 600")

    draft_path = args.dataset_level_draft.expanduser().resolve()
    if not draft_path.is_file():
        raise SystemExit(f"Dataset-Level draft does not exist: {draft_path}")
    try:
        assert_source_run_recoverable(draft_path)
    except AgentRecoveryError as exc:
        raise SystemExit(str(exc)) from exc
    run_id = args.run_id or f"dataset-critic-resume-{uuid.uuid4().hex[:12]}"
    run_dir = (args.run_dir or Path("artifacts") / "runs" / run_id).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Refusing to append to existing immutable run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_path = run_dir / "audit.jsonl"
    audit = JsonlAuditSink(audit_path, run_id=run_id)
    tools, context = create_dataset_intelligence_critic_tools(
        dataset_level_draft_path=draft_path
    )
    result = DatasetIntelligenceCriticAgent(
        runtime=AgentRuntime(
            provider=_provider(args),
            tools=tools,
            audit=audit,
            limits=RuntimeLimits(max_iterations=6, max_tool_calls=4),
            run_id=run_id,
            audit_path=str(audit_path),
        ),
        context=context,
    ).run()
    atomic_json(run_dir / "critic_agent_result.json", result.to_dict())
    critique = result.latest_tool_result("record_dataset_critique")
    if result.status != "completed" or critique is None:
        print(f"run_id: {run_id}")
        print(f"status: {result.status}")
        print(f"run_dir: {run_dir}")
        if result.error:
            print(f"error: {result.error}")
        return 1

    critique_path = run_dir / "dataset_critique.json"
    atomic_json(critique_path, critique)
    verdict = str(critique["verdict"])
    contract_path: Path | None = None
    if verdict == "pass":
        contract_path = run_dir / "dataset_level_contract.json"
        audit.record(
            "deterministic_stage_started",
            {
                "stage": "dataset_level_freeze_after_critic_resume",
                "dataset_level_draft_path": str(draft_path),
                "dataset_critique_path": str(critique_path),
            },
        )
        freeze_dataset_level_contract(
            dataset_level_draft_path=draft_path,
            dataset_critique_path=critique_path,
            output_path=contract_path,
        )
        audit.record(
            "deterministic_stage_completed",
            {
                "stage": "dataset_level_freeze_after_critic_resume",
                "dataset_level_contract_path": str(contract_path),
            },
        )

    print(f"run_id: {run_id}")
    print("status: completed")
    print(f"verdict: {verdict}")
    print(f"run_dir: {run_dir}")
    print(f"source_draft: {draft_path}")
    if contract_path is not None:
        print(f"dataset_level_contract: {contract_path}")
    return 0 if verdict == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
