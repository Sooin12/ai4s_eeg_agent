"""CLI for the non-activating Protocol Planner Agent."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path

from .audit import JsonlAuditSink
from .protocol_planner import ProtocolPlannerAgent, create_protocol_planner_tools
from .providers import OpenAICompatibleProvider
from .runtime import AgentRuntime, RuntimeLimits
from bci_autodiscovery.workflow.protocol_artifacts import ProtocolArtifactRegistry


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
        description="Run the dataset-neutral, non-activating Protocol Planner Agent."
    )
    parser.add_argument("--dataset-profile", type=Path, required=True)
    parser.add_argument("--model", default="kimi-k2.7-code")
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--provider-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--registry-root", type=Path, default=Path("artifacts/protocols"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or f"protocol-planner-{uuid.uuid4().hex[:12]}"
    run_dir = (args.run_dir or Path("artifacts") / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_path = run_dir / "audit.jsonl"
    if audit_path.exists():
        raise SystemExit(f"Refusing to append to an existing immutable run: {run_dir}")
    tools, context = create_protocol_planner_tools(
        dataset_profile_path=args.dataset_profile
    )

    def progress(event: str, payload: dict) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        messages = {
            "model_request": f"Requesting model iteration {payload.get('iteration')}",
            "stream_opened": "Model streaming connection established",
            "tool_requested": f"Executing tool: {payload.get('tool_name')}",
            "tool_completed": f"Tool completed: {payload.get('tool_name')}",
            "tool_failed": f"Tool validation failed; error returned to Agent: {payload.get('tool_name')}",
            "run_failed": f"Protocol Planner failed: {payload.get('error')}",
            "run_completed": "Protocol Planner completed",
        }
        if event in messages:
            print(f"[{stamp}] {messages[event]}", flush=True)

    provider = OpenAICompatibleProvider.kimi(
        model=args.model, max_output_tokens=args.max_output_tokens
    )
    provider.timeout_seconds = args.provider_timeout_seconds
    provider.progress_callback = progress
    print(f"run_id: {run_id}", flush=True)
    print("Starting Protocol Planner Agent...", flush=True)
    runtime = AgentRuntime(
        provider=provider,
        tools=tools,
        audit=JsonlAuditSink(audit_path, run_id=run_id),
        limits=RuntimeLimits(max_iterations=8, max_tool_calls=8),
        run_id=run_id,
        audit_path=str(audit_path),
        progress_callback=progress,
    )
    result = ProtocolPlannerAgent(runtime=runtime, context=context).run()
    proposal = result.latest_tool_result("record_protocol_proposal")
    _atomic_json(run_dir / "agent_result.json", result.to_dict())
    if proposal is not None:
        proposal_path = run_dir / "protocol_proposal.json"
        _atomic_json(proposal_path, proposal)
        registry = ProtocolArtifactRegistry(
            root=args.registry_root,
            dataset_id=proposal["dataset_id"],
        )
        registered = registry.register_revision(
            source_path=proposal_path,
            kind="agent_proposal",
        )
        print(f"registered_proposal_path: {registered['path']}")
    print(f"status: {result.status}")
    print(f"proposal_recorded: {proposal is not None}")
    print(f"run_dir: {run_dir}")
    if result.final_text:
        print(result.final_text)
    if result.error:
        print(f"error: {result.error}")
    return 0 if result.status == "completed" and proposal is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
