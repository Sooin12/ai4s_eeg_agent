"""CLI entry point for the first auditable Dataset Profiler Agent."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from .audit import JsonlAuditSink
from .dataset_profiler import DatasetProfilerAgent, create_dataset_profiler_tools
from .providers import DatasetProfilerMockProvider, OpenAICompatibleProvider
from .runtime import AgentRuntime, RuntimeLimits


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _provider(
    name: str,
    model: str | None,
    *,
    max_output_tokens: int,
    reasoning_effort: str,
):
    if name == "mock":
        return DatasetProfilerMockProvider()
    if name == "deepseek":
        return OpenAICompatibleProvider.deepseek(
            model=model or "deepseek-v4-flash",
            max_output_tokens=max_output_tokens,
        )
    if name == "kimi":
        return OpenAICompatibleProvider.kimi(
            model=model or "kimi-k3",
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
    raise ValueError(f"Unknown provider: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the auditable dataset-level understanding agent."
    )
    parser.add_argument(
        "--dataset-id",
        default="auto",
        help="Optional stable ID hint; auto lets the selected adapter derive it.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "deepseek", "kimi"],
        help="mock is offline and is the safe default; paid providers require explicit selection.",
    )
    parser.add_argument("--model", help="Optional explicit provider model ID.")
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help=(
            "Hard cap per paid response. Defaults to 16384 for always-thinking "
            "Kimi K2.7 Code and 1024 for Kimi K3/other smoke tests."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        default="low",
        choices=["low", "high", "max"],
        help="Kimi reasoning effort; low is the guarded smoke-test default.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Output directory; defaults to artifacts/runs/<run-id>.",
    )
    parser.add_argument("--run-id", help="Stable run identifier for controlled replays.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_output_tokens is not None and (
        args.max_output_tokens < 1 or args.max_output_tokens > 32768
    ):
        raise SystemExit("--max-output-tokens must be between 1 and 32768")
    run_id = args.run_id or f"dataset-profile-{uuid.uuid4().hex[:12]}"
    run_dir = (args.run_dir or Path("artifacts") / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_path = run_dir / "audit.jsonl"
    if audit_path.exists():
        raise SystemExit(
            f"Refusing to append to an existing immutable run: {run_dir}. "
            "Choose a new --run-id or --run-dir."
        )
    provider = _provider(
        args.provider,
        args.model,
        max_output_tokens=(
            args.max_output_tokens
            if args.max_output_tokens is not None
            else (16384 if (args.model or "").startswith("kimi-k2.7") else 1024)
        ),
        reasoning_effort=args.reasoning_effort,
    )
    runtime = AgentRuntime(
        provider=provider,
        tools=create_dataset_profiler_tools(),
        audit=JsonlAuditSink(audit_path, run_id=run_id),
        limits=RuntimeLimits(max_iterations=8, max_tool_calls=8),
        run_id=run_id,
        audit_path=str(audit_path),
    )
    result = DatasetProfilerAgent(runtime).run(
        dataset_id=args.dataset_id,
        dataset_root=args.dataset_root,
        validation_path=args.validation,
    )
    _atomic_json(run_dir / "agent_result.json", result.to_dict())
    profile = result.latest_tool_result("profile_dataset")
    inspection = result.latest_tool_result("inspect_dataset")
    if inspection is not None:
        _atomic_json(run_dir / "dataset_inspection.json", inspection)
    if profile is not None:
        _atomic_json(run_dir / "dataset_profile.json", profile)

    print(f"run_id: {run_id}")
    print(f"status: {result.status}")
    print(f"run_dir: {run_dir}")
    print(f"audit: {audit_path}")
    if result.final_text:
        print(result.final_text)
    if result.approval_request:
        print(
            "approval_required: "
            + json.dumps(result.approval_request.to_dict(), ensure_ascii=False)
        )
    if result.error:
        print(f"error: {result.error}")
    return 0 if result.status == "completed" and profile is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
