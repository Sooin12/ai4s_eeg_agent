"""CLI for one auditable Protocol Reviewer conversation turn."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path

from .audit import JsonlAuditSink
from .protocol_reviewer import ProtocolReviewerAgent, create_protocol_reviewer_tools
from .providers import OpenAICompatibleProvider
from .runtime import AgentRuntime, RuntimeLimits
from bci_autodiscovery.workflow.protocol_artifacts import (
    ProtocolArtifactRegistry,
    atomic_json,
    load_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one persistent, auditable Protocol Reviewer conversation turn."
    )
    parser.add_argument("--dataset-profile", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--feedback", required=True)
    parser.add_argument("--model", default="kimi-k2.7-code")
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--provider-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--run-id")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--registry-root", type=Path, default=Path("artifacts/protocols"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = args.run_id or f"protocol-review-{uuid.uuid4().hex[:12]}"
    run_dir = (args.run_dir or Path("artifacts") / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    audit_path = run_dir / "audit.jsonl"
    if audit_path.exists():
        raise SystemExit(f"Refusing to append to an existing immutable run: {run_dir}")
    tools, context = create_protocol_reviewer_tools(
        dataset_profile_path=args.dataset_profile,
        proposal_path=args.proposal,
        user_feedback=args.feedback,
    )
    source_proposal = load_json(args.proposal.resolve())
    registry = ProtocolArtifactRegistry(
        root=args.registry_root,
        dataset_id=source_proposal["dataset_id"],
    )
    registry.register_revision(
        source_path=args.proposal,
        kind="review_source",
        parent_sha256=(source_proposal.get("revision") or {}).get("parent_sha256"),
    )

    def progress(event: str, payload: dict) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        messages = {
            "model_request": f"Requesting model iteration {payload.get('iteration')}",
            "stream_opened": "Model streaming connection established",
            "tool_requested": f"Executing tool: {payload.get('tool_name')}",
            "tool_completed": f"Tool completed: {payload.get('tool_name')}",
            "tool_failed": f"Tool validation failed; error returned to Agent: {payload.get('tool_name')}",
            "run_failed": f"Protocol Reviewer failed: {payload.get('error')}",
            "run_completed": "Protocol Reviewer turn completed",
        }
        if event in messages:
            print(f"[{stamp}] {messages[event]}", flush=True)

    provider = OpenAICompatibleProvider.kimi(
        model=args.model, max_output_tokens=args.max_output_tokens
    )
    provider.timeout_seconds = args.provider_timeout_seconds
    provider.progress_callback = progress
    print(f"run_id: {run_id}", flush=True)
    print("Starting Protocol Reviewer Agent...", flush=True)
    runtime = AgentRuntime(
        provider=provider,
        tools=tools,
        audit=JsonlAuditSink(audit_path, run_id=run_id),
        limits=RuntimeLimits(max_iterations=8, max_tool_calls=8),
        run_id=run_id,
        audit_path=str(audit_path),
        progress_callback=progress,
    )
    result = ProtocolReviewerAgent(runtime=runtime, context=context).run()
    revision = result.latest_tool_result("record_protocol_revision")
    explanation = result.latest_tool_result("record_protocol_explanation")
    recorded = revision or explanation
    atomic_json(run_dir / "agent_result.json", result.to_dict())
    registry_record = None
    if revision is not None:
        revision_path = run_dir / "protocol_revision.json"
        atomic_json(revision_path, revision, refuse_overwrite=True)
        registry_record = registry.register_revision(
            source_path=revision_path,
            kind="agent_revision",
            parent_sha256=revision["revision"]["parent_sha256"],
        )
    if recorded is not None:
        review_record = {
            "schema_version": "1.0",
            "dataset_id": recorded["dataset_id"],
            "run_id": run_id,
            "user_feedback_verbatim": args.feedback,
            "result_kind": "revision" if revision is not None else "explanation",
            "agent_record": recorded,
            "agent_final_text": result.final_text,
            "audit_path": str(audit_path),
            "registry_revision": registry_record,
        }
        review_path = run_dir / "review_turn.json"
        atomic_json(review_path, review_record, refuse_overwrite=True)
        registry.register_review_turn(review_path=review_path)
    print(f"status: {result.status}")
    print(f"result_recorded: {recorded is not None}")
    print(f"result_kind: {'revision' if revision is not None else 'explanation' if explanation else 'none'}")
    print(f"run_dir: {run_dir}")
    if registry_record:
        print(f"revision_path: {registry_record['path']}")
    if result.final_text:
        print(result.final_text)
    if result.error:
        print(f"error: {result.error}")
    return 0 if result.status == "completed" and recorded is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
