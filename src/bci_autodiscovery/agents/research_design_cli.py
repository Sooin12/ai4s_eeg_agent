"""CLI for the complete budgeted and recoverable Research Design Agent."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path

from bci_autodiscovery.workflow.autonomy import load_autonomy_envelope, sha256_path
from bci_autodiscovery.workflow.budget import BudgetLedger, limits_from_envelope
from bci_autodiscovery.workflow.dataset_contract import load_dataset_level_contract
from bci_autodiscovery.reporting import AgentOutputPublisher

from .audit import JsonlAuditSink
from .budgeted_provider import TokenPricing
from .providers import OpenAICompatibleProvider, enforce_minimum_request_interval
from .research_design_agent import ResearchDesignAgent
from .run_recovery import (
    assert_research_design_run_recoverable,
    write_process_state,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run outcome-blind Research Protocol Planner -> independent Critic -> "
            "deterministic frozen ResearchProtocol with append-only budget accounting."
        )
    )
    parser.add_argument("--dataset-level-contract", type=Path, required=True)
    parser.add_argument("--autonomy-envelope", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--run-dir", type=Path)
    destination.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--provider", choices=["deepseek", "kimi"], required=True)
    parser.add_argument("--model")
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--provider-timeout-seconds", type=float, default=240.0)
    parser.add_argument(
        "--reasoning-effort", choices=["low", "high", "max"], default="low"
    )
    parser.add_argument("--max-revision-cycles", type=int, default=2)
    parser.add_argument("--pricing-currency", default="USD")
    parser.add_argument("--prompt-cost-per-million", type=float, required=True)
    parser.add_argument("--completion-cost-per-million", type=float, required=True)
    parser.add_argument("--cached-prompt-cost-per-million", type=float, default=0.0)
    parser.add_argument("--pricing-source", required=True)
    parser.add_argument(
        "--agent-output-root", type=Path, default=Path("agent_outputs")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 256 <= args.max_output_tokens <= 32768:
        raise SystemExit("--max-output-tokens must be between 256 and 32768")
    if not 30 <= args.provider_timeout_seconds <= 600:
        raise SystemExit("--provider-timeout-seconds must be between 30 and 600")
    if not 0 <= args.max_revision_cycles <= 8:
        raise SystemExit("--max-revision-cycles must be between 0 and 8")

    contract_path = args.dataset_level_contract.expanduser().resolve()
    envelope_path = args.autonomy_envelope.expanduser().resolve()
    contract = load_dataset_level_contract(contract_path)
    envelope = load_autonomy_envelope(
        envelope_path,
        expected_dataset_id=str(contract["dataset_id"]),
        expected_dataset_contract_path=contract_path,
    )
    resume = args.resume_run_dir is not None
    if resume:
        run_dir = args.resume_run_dir.expanduser().resolve()
        existing_state = assert_research_design_run_recoverable(run_dir)
        run_id = str(existing_state["run_id"])
        if args.run_id and args.run_id != run_id:
            raise SystemExit("--run-id does not match the recoverable run")
    else:
        run_id = args.run_id or f"research-design-{uuid.uuid4().hex[:12]}"
        run_dir = (
            args.run_dir or Path("artifacts") / "runs" / run_id
        ).expanduser().resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(f"Refusing to append to existing immutable run: {run_dir}")

    pricing = TokenPricing(
        currency=args.pricing_currency,
        prompt_per_million=args.prompt_cost_per_million,
        completion_per_million=args.completion_cost_per_million,
        cached_prompt_per_million=args.cached_prompt_cost_per_million,
        source=args.pricing_source,
    )
    if pricing.currency != envelope["resource_budget"]["paid_cost_currency"]:
        raise SystemExit(
            "--pricing-currency must match AutonomyEnvelope.resource_budget.paid_cost_currency"
        )
    audit_path = run_dir / "audit.jsonl"
    audit = JsonlAuditSink(audit_path, run_id=run_id, resume=resume)
    ledger_path = run_dir / "budget_ledger.jsonl"
    ledger = BudgetLedger(
        ledger_path,
        run_id=run_id,
        limits=limits_from_envelope(envelope),
        authority_sha256=sha256_path(envelope_path),
        create=not resume,
    )

    def progress(event: str, payload: dict) -> None:
        if event not in {
            "model_request",
            "provider_attempt_started",
            "provider_retry_scheduled",
            "tool_requested",
            "tool_completed",
            "tool_failed",
            "run_completed",
            "run_failed",
        }:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        detail = (
            payload.get("tool_name")
            or payload.get("iteration")
            or payload.get("attempt")
            or payload.get("error")
            or ""
        )
        print(f"[{stamp}] {event}: {detail}", flush=True)

    def provider_factory(_stage: str, _cycle: int):
        if args.provider == "kimi":
            model = args.model or "kimi-k2.7-code"
            provider = OpenAICompatibleProvider.kimi(
                model=model,
                max_output_tokens=args.max_output_tokens,
                reasoning_effort=args.reasoning_effort,
            )
            enforce_minimum_request_interval(
                provider,
                seconds=21.0,
                scope="kimi-organization-rpm",
            )
        else:
            provider = OpenAICompatibleProvider.deepseek(
                model=args.model or "deepseek-v4-flash",
                max_output_tokens=args.max_output_tokens,
            )
        provider.timeout_seconds = args.provider_timeout_seconds
        provider.progress_callback = progress
        return provider

    agent = ResearchDesignAgent(
        run_id=run_id,
        run_dir=run_dir,
        dataset_level_contract_path=contract_path,
        autonomy_envelope_path=envelope_path,
        provider_factory=provider_factory,
        budget_ledger=ledger,
        pricing=pricing,
        audit=audit,
        audit_path=audit_path,
        max_revision_cycles=args.max_revision_cycles,
    )
    write_process_state(run_dir, run_id=run_id, status="running")
    print(f"run_id: {run_id}", flush=True)
    print(f"mode: {'resume' if resume else 'new'}", flush=True)
    print("Starting complete Research Design Agent...", flush=True)
    try:
        result = agent.run(resume=resume)
    except Exception as exc:
        write_process_state(
            run_dir,
            run_id=run_id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        print(f"status: failed\nerror: {type(exc).__name__}: {exc}", flush=True)
        return 1
    write_process_state(run_dir, run_id=run_id, status=result.status, error=result.error)
    print(f"status: {result.status}")
    print(f"cycles: {result.cycles}")
    print(f"run_dir: {run_dir}")
    print(f"audit: {audit_path}")
    print(f"budget_ledger: {ledger_path}")
    if result.artifacts.get("frozen_protocol"):
        print(f"frozen_protocol: {result.artifacts['frozen_protocol']['path']}")
        if result.status == "completed":
            record = AgentOutputPublisher(args.agent_output_root).publish(
                dataset_id=str(contract["dataset_id"]),
                stage="research_design",
                artifact_name="frozen_protocol",
                source_path=Path(result.artifacts["frozen_protocol"]["path"]),
                run_id=run_id,
            )
            print(f"agent_output: {Path(record['published_path']).parents[1]}")
    if result.error:
        print(f"error: {result.error}")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
