"""Unified CLI for the complete Dataset-Level Agent."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path

from bci_autodiscovery.reporting import AgentOutputPublisher

from .audit import JsonlAuditSink
from .dataset_level_agent import DatasetLevelAgent
from .providers import OpenAICompatibleProvider
from .run_recovery import write_process_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run DatasetProfile -> coarse space -> network discovery -> independent "
            "Dataset Critic -> frozen DatasetLevelContract."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-profile", type=Path)
    source.add_argument("--dataset-root", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--dataset-id", default="auto")
    parser.add_argument(
        "--component-registry",
        type=Path,
        default=Path("configs/component_registry.v0.json"),
    )
    parser.add_argument(
        "--provider",
        choices=["deepseek", "kimi"],
        required=True,
        help="Explicit selection is required because literature synthesis and critique are paid calls.",
    )
    parser.add_argument("--model")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--provider-timeout-seconds", type=float, default=240.0)
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "high", "max"],
        default="low",
    )
    parser.add_argument("--max-revision-cycles", type=int, default=1)
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--agent-output-root", type=Path, default=Path("agent_outputs")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset_root is not None and args.validation is None:
        raise SystemExit("--validation is required with --dataset-root")
    if args.dataset_profile is not None and args.validation is not None:
        raise SystemExit("--validation is only valid with --dataset-root")
    if args.max_output_tokens is not None and not 1 <= args.max_output_tokens <= 32768:
        raise SystemExit("--max-output-tokens must be between 1 and 32768")
    if not 30 <= args.provider_timeout_seconds <= 600:
        raise SystemExit("--provider-timeout-seconds must be between 30 and 600")
    if not 0 <= args.max_revision_cycles <= 4:
        raise SystemExit("--max-revision-cycles must be between 0 and 4")

    run_id = args.run_id or f"dataset-level-{uuid.uuid4().hex[:12]}"
    run_dir = (args.run_dir or Path("artifacts") / "runs" / run_id).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Refusing to append to existing immutable run: {run_dir}")
    audit_path = run_dir / "audit.jsonl"
    audit = JsonlAuditSink(audit_path, run_id=run_id)

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
        detail = payload.get("tool_name") or payload.get("iteration") or payload.get("error")
        print(f"[{stamp}] {event}: {detail or ''}", flush=True)

    def provider_factory(_cycle: int):
        if args.provider == "kimi":
            model = args.model or "kimi-k2.7-code"
            provider = OpenAICompatibleProvider.kimi(
                model=model,
                max_output_tokens=args.max_output_tokens or (
                    16384 if model.startswith("kimi-k2.7") else 4096
                ),
                reasoning_effort=args.reasoning_effort,
            )
        else:
            provider = OpenAICompatibleProvider.deepseek(
                model=args.model or "deepseek-v4-flash",
                max_output_tokens=args.max_output_tokens or 4096,
            )
        provider.timeout_seconds = args.provider_timeout_seconds
        provider.progress_callback = progress
        return provider

    agent = DatasetLevelAgent(
        run_id=run_id,
        run_dir=run_dir,
        component_registry_path=args.component_registry,
        profiler_provider=(
            provider_factory(0) if args.dataset_root is not None else None
        ),
        literature_provider_factory=provider_factory,
        critic_provider_factory=provider_factory,
        audit=audit,
        max_revision_cycles=args.max_revision_cycles,
    )
    write_process_state(run_dir, run_id=run_id, status="running")
    print(f"run_id: {run_id}", flush=True)
    print("Starting complete Dataset-Level Agent...", flush=True)
    try:
        if args.dataset_profile is not None:
            result = agent.run_from_profile(dataset_profile_path=args.dataset_profile)
        else:
            result = agent.run(
                dataset_id=args.dataset_id,
                dataset_root=args.dataset_root,
                validation_path=args.validation,
            )
    except Exception as exc:
        write_process_state(
            run_dir,
            run_id=run_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    write_process_state(
        run_dir, run_id=run_id, status=result.status, error=result.error
    )
    print(f"status: {result.status}")
    print(f"cycles: {result.cycles}")
    print(f"run_dir: {run_dir}")
    print(f"audit: {audit_path}")
    contract = result.artifacts.get("dataset_level_contract")
    if contract:
        print(f"dataset_level_contract: {contract['path']}")
    if result.status == "completed" and contract:
        contract_payload = json.loads(
            Path(contract["path"]).read_text(encoding="utf-8")
        )
        dataset_id = str(contract_payload["dataset_id"])
        publish_specs = []
        for name in (
            "dataset_inspection",
            "dataset_profile",
            "canonical_search_space",
            "dataset_level_contract",
        ):
            artifact = result.artifacts.get(name)
            if artifact:
                publish_specs.append(
                    ("dataset_understanding", name, Path(artifact["path"]))
                )
        records = AgentOutputPublisher(args.agent_output_root).publish_many(
            dataset_id=dataset_id,
            run_id=run_id,
            artifacts=publish_specs,
        )
        if records:
            print(f"agent_output: {Path(records[0]['published_path']).parents[1]}")
    if result.error:
        print(f"error: {result.error}")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
