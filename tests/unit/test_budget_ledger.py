from __future__ import annotations

import json

import pytest

from bci_autodiscovery.agents.budgeted_provider import BudgetedProvider, TokenPricing
from bci_autodiscovery.agents.contracts import ModelResponse, ModelUsage
from bci_autodiscovery.agents.providers import ProviderError, ScriptedProvider
from bci_autodiscovery.workflow.budget import (
    BudgetExceededError,
    BudgetLedger,
    BudgetLedgerError,
)


def _limits() -> dict[str, float]:
    return {
        "research_cycles": 3,
        "candidate_executions": 4,
        "compute_seconds": 60,
        "api_total_tokens": 1000,
        "paid_cost": 1.0,
        "provider_retries": 2,
        "recovery_attempts": 1,
        "confirmation_accesses": 1,
    }


def test_budget_ledger_replays_hash_chain_and_rejects_overrun(tmp_path) -> None:
    path = tmp_path / "budget.jsonl"
    ledger = BudgetLedger(
        path,
        run_id="budget-fixture",
        limits=_limits(),
        authority_sha256="a" * 64,
        create=True,
    )
    ledger.precheck("planner", {"api_total_tokens": 100})
    ledger.account(
        "planner",
        {
            "api_prompt_tokens": 60,
            "api_completion_tokens": 20,
            "api_total_tokens": 80,
            "paid_cost": 0.01,
        },
    )
    replayed = BudgetLedger(
        path,
        run_id="budget-fixture",
        limits=_limits(),
        authority_sha256="a" * 64,
        create=False,
    )
    assert replayed.totals["api_total_tokens"] == 80
    with pytest.raises(BudgetExceededError, match="precheck rejected"):
        replayed.precheck("too-large", {"api_total_tokens": 921})


def test_budget_ledger_detects_tampering(tmp_path) -> None:
    path = tmp_path / "budget.jsonl"
    ledger = BudgetLedger(
        path,
        run_id="tamper-fixture",
        limits=_limits(),
        authority_sha256="b" * 64,
        create=True,
    )
    ledger.account("cycle", {"research_cycles": 1})
    lines = path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[-1])
    event["delta"]["research_cycles"] = 2
    lines[-1] = json.dumps(event, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(BudgetLedgerError, match="integrity"):
        BudgetLedger(
            path,
            run_id="tamper-fixture",
            limits=_limits(),
            authority_sha256="b" * 64,
            create=False,
        )


def test_budgeted_provider_accounts_exact_usage(tmp_path) -> None:
    ledger = BudgetLedger(
        tmp_path / "budget.jsonl",
        run_id="provider-fixture",
        limits=_limits(),
        authority_sha256="c" * 64,
        create=True,
    )
    provider = ScriptedProvider(
        [
            ModelResponse(
                content="done",
                usage=ModelUsage(
                    prompt_tokens=40,
                    completion_tokens=10,
                    total_tokens=50,
                    cached_tokens=5,
                ),
            )
        ]
    )
    wrapped = BudgetedProvider(
        provider=provider,
        ledger=ledger,
        pricing=None,
        stage="planner",
    )
    wrapped.complete(messages=[{"role": "user", "content": "go"}], tools=[])
    assert ledger.totals["api_prompt_tokens"] == 40
    assert ledger.totals["api_completion_tokens"] == 10
    assert ledger.totals["api_cached_tokens"] == 5
    assert ledger.totals["api_total_tokens"] == 50


class _PaidMissingUsageProvider(ScriptedProvider):
    def audit_config(self):
        return {"provider": self.name, "model": self.model, "paid": True}


def test_paid_provider_without_usage_fails_closed(tmp_path) -> None:
    ledger = BudgetLedger(
        tmp_path / "budget.jsonl",
        run_id="missing-usage",
        limits=_limits(),
        authority_sha256="d" * 64,
        create=True,
    )
    wrapped = BudgetedProvider(
        provider=_PaidMissingUsageProvider([ModelResponse(content="unmetered")]),
        ledger=ledger,
        pricing=TokenPricing(
            currency="USD",
            prompt_per_million=1.0,
            completion_per_million=2.0,
        ),
        stage="critic",
    )
    with pytest.raises(ProviderError, match="omitted token usage"):
        wrapped.complete(messages=[{"role": "user", "content": "go"}], tools=[])
