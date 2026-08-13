"""CLI for dataset-specific scholarly discovery with Kimi."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path

from .audit import JsonlAuditSink
from .literature_scout import LiteratureScoutAgent, create_literature_scout_tools
from .providers import OpenAICompatibleProvider
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run required dataset-specific scholarly frontier discovery."
    )
    parser.add_argument("--search-space", type=Path, required=True)
    parser.add_argument("--provider", choices=["kimi"], default="kimi")
    parser.add_argument("--model", default="kimi-k2.7-code")
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--provider-timeout-seconds", type=float, default=240.0)
    parser.add_argument(
        "--evidence-db",
        type=Path,
        help="Optional existing evidence SQLite database for a resumable synthesis run.",
    )
    parser.add_argument(
        "--evidence-run-id",
        help="Search run ID stored in --evidence-db; required when resuming.",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.max_output_tokens <= 32768:
        raise SystemExit("--max-output-tokens must be between 1 and 32768")
    if (args.evidence_db is None) != (args.evidence_run_id is None):
        raise SystemExit("--evidence-db and --evidence-run-id must be supplied together")
    if args.provider_timeout_seconds < 30 or args.provider_timeout_seconds > 600:
        raise SystemExit("--provider-timeout-seconds must be between 30 and 600")
    run_id = args.run_id or f"literature-scout-{uuid.uuid4().hex[:12]}"
    run_dir = (args.run_dir or Path("artifacts") / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_path = run_dir / "audit.jsonl"
    if audit_path.exists():
        raise SystemExit(f"Refusing to append to an existing immutable run: {run_dir}")
    evidence_db = (
        args.evidence_db.expanduser().resolve()
        if args.evidence_db is not None
        else run_dir / "evidence.sqlite"
    )
    evidence_run_id = args.evidence_run_id or run_id
    tools, context = create_literature_scout_tools(
        search_space_path=args.search_space,
        evidence_db_path=evidence_db,
        search_run_id=evidence_run_id,
    )
    def progress(event: str, payload: dict) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        if event == "stream_opened":
            message = f"Model streaming connection established ({payload.get('model')})"
        elif event == "stream_activity":
            message = (
                "Model is analyzing literature evidence "
                f"(reasoning chars={payload.get('reasoning_chars', 0)}, "
                f"response chars={payload.get('content_chars', 0)})"
            )
        elif event == "stream_completed":
            message = f"Model iteration completed (tool calls={payload.get('tool_calls', 0)})"
        elif event == "model_request":
            message = f"Requesting model iteration {payload.get('iteration')}"
        elif event == "tool_requested":
            message = f"Executing tool: {payload.get('tool_name')}"
        elif event == "tool_completed":
            message = f"Tool completed: {payload.get('tool_name')}"
        elif event == "run_failed":
            message = f"Run failed: {payload.get('error')}"
        elif event == "run_completed":
            message = "Agent run completed"
        else:
            return
        print(f"[{stamp}] {message}", flush=True)

    provider = OpenAICompatibleProvider.kimi(
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )
    provider.timeout_seconds = args.provider_timeout_seconds
    provider.progress_callback = progress
    print(f"run_id: {run_id}", flush=True)
    print(f"evidence_run_id: {evidence_run_id}", flush=True)
    print("Starting Literature Scout Agent...", flush=True)
    runtime = AgentRuntime(
        provider=provider,
        tools=tools,
        audit=JsonlAuditSink(audit_path, run_id=run_id),
        limits=RuntimeLimits(max_iterations=32, max_tool_calls=32),
        run_id=run_id,
        audit_path=str(audit_path),
        progress_callback=progress,
    )
    result = LiteratureScoutAgent(runtime=runtime, context=context).run()
    status = tools.execute("inspect_frontier_discovery_status", {})
    _atomic_json(run_dir / "agent_result.json", result.to_dict())
    _atomic_json(run_dir / "frontier_discovery.json", status)
    _atomic_json(
        run_dir / "evidence_reference.json",
        {"path": str(evidence_db), "search_run_id": evidence_run_id},
    )
    print(f"run_id: {run_id}")
    print(f"status: {result.status}")
    print(f"discovery_complete: {status['complete']}")
    print(
        f"network_searches: {status['attempted_search_count']}/"
        f"{status['planned_search_count']}"
    )
    print(f"directions: {status['direction_count']}")
    print(f"run_dir: {run_dir}")
    print(f"audit: {audit_path}")
    if result.final_text:
        print(result.final_text)
    if result.error:
        print(f"error: {result.error}")
    return 0 if result.status == "completed" and status["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
