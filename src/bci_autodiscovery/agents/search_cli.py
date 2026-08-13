"""CLI for generating a dataset-level search-space draft."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from .audit import JsonlAuditSink
from .providers import OpenAICompatibleProvider, SearchSpaceBuilderMockProvider
from .runtime import AgentRuntime, RuntimeLimits
from .search_space_builder import SearchSpaceBuilderAgent, create_search_space_builder_tools


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an auditable search-space draft.")
    parser.add_argument("--dataset-profile", type=Path, required=True)
    parser.add_argument("--component-registry", type=Path, required=True)
    parser.add_argument("--provider", choices=["mock", "kimi"], default="mock")
    parser.add_argument("--model", default="kimi-k2.7-code")
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.max_output_tokens <= 32768:
        raise SystemExit("--max-output-tokens must be between 1 and 32768")
    run_id = args.run_id or f"search-space-{uuid.uuid4().hex[:12]}"
    run_dir = (args.run_dir or Path("artifacts") / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_path = run_dir / "audit.jsonl"
    if audit_path.exists():
        raise SystemExit(f"Refusing to append to an existing immutable run: {run_dir}")
    provider = (
        SearchSpaceBuilderMockProvider()
        if args.provider == "mock"
        else OpenAICompatibleProvider.kimi(
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )
    )
    runtime = AgentRuntime(
        provider=provider,
        tools=create_search_space_builder_tools(),
        audit=JsonlAuditSink(audit_path, run_id=run_id),
        limits=RuntimeLimits(max_iterations=4, max_tool_calls=4),
        run_id=run_id,
        audit_path=str(audit_path),
    )
    result = SearchSpaceBuilderAgent(runtime).run(
        dataset_profile_path=args.dataset_profile,
        component_registry_path=args.component_registry,
    )
    _atomic_json(run_dir / "agent_result.json", result.to_dict())
    draft = result.latest_tool_result("build_search_space_draft")
    if draft is not None:
        _atomic_json(run_dir / "search_space_draft.json", draft)
    print(f"run_id: {run_id}")
    print(f"status: {result.status}")
    print(f"run_dir: {run_dir}")
    print(f"audit: {audit_path}")
    if result.final_text:
        print(result.final_text)
    if result.error:
        print(f"error: {result.error}")
    return 0 if result.status == "completed" and draft is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
