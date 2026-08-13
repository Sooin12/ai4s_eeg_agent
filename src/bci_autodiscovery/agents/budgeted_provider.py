"""Provider decorator enforcing the shared append-only budget ledger."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from bci_autodiscovery.workflow.budget import BudgetLedger

from .contracts import ModelResponse
from .providers import ModelProvider, ProviderError


@dataclass(frozen=True)
class TokenPricing:
    currency: str
    prompt_per_million: float
    completion_per_million: float
    cached_prompt_per_million: float = 0.0
    source: str = "operator_supplied_versioned_pricing"

    def __post_init__(self) -> None:
        if not self.currency.strip():
            raise ValueError("Pricing currency must be non-empty")
        if self.prompt_per_million < 0 or self.completion_per_million < 0:
            raise ValueError("Token prices cannot be negative")
        if self.cached_prompt_per_million < 0:
            raise ValueError("Cached token price cannot be negative")

    def cost(
        self, *, prompt_tokens: int, completion_tokens: int, cached_tokens: int
    ) -> float:
        cached = min(max(cached_tokens, 0), max(prompt_tokens, 0))
        uncached = max(prompt_tokens - cached, 0)
        return (
            uncached * self.prompt_per_million
            + cached * self.cached_prompt_per_million
            + max(completion_tokens, 0) * self.completion_per_million
        ) / 1_000_000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "prompt_per_million": self.prompt_per_million,
            "completion_per_million": self.completion_per_million,
            "cached_prompt_per_million": self.cached_prompt_per_million,
            "source": self.source,
        }


class BudgetedProvider:
    """Run preflight reservation checks and account exact provider-reported usage."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        ledger: BudgetLedger,
        pricing: TokenPricing | None,
        stage: str,
    ) -> None:
        self.provider = provider
        self.ledger = ledger
        self.pricing = pricing
        self.stage = stage
        self.name = provider.name
        self.model = provider.model
        paid = bool(provider.audit_config().get("paid"))
        if paid and pricing is None:
            raise ValueError("A paid provider requires an explicit versioned pricing record")
        self._install_retry_accounting()

    def audit_config(self) -> dict[str, Any]:
        config = dict(self.provider.audit_config())
        config["budget_enforced"] = True
        config["budget_stage"] = self.stage
        config["pricing"] = self.pricing.to_dict() if self.pricing else None
        return config

    def complete(
        self,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> ModelResponse:
        estimated_prompt = self._estimate_prompt_tokens(messages, tools)
        maximum_completion = self._maximum_completion_tokens()
        estimated_cost = (
            self.pricing.cost(
                prompt_tokens=estimated_prompt,
                completion_tokens=maximum_completion,
                cached_tokens=0,
            )
            if self.pricing
            else 0.0
        )
        self.ledger.precheck(
            f"{self.stage}:model_call",
            {
                "api_total_tokens": estimated_prompt + maximum_completion,
                "paid_cost": estimated_cost,
            },
        )
        try:
            response = self.provider.complete(messages=messages, tools=tools)
        except Exception as exc:
            self.ledger.account(
                f"{self.stage}:provider_failure",
                {"provider_failures": 1},
                metadata={
                    "provider": self.name,
                    "model": self.model,
                    "error_type": type(exc).__name__,
                    "usage_accounting": "unavailable_before_provider_response",
                },
            )
            raise
        usage = response.usage
        total_tokens = max(
            int(usage.total_tokens),
            int(usage.prompt_tokens) + int(usage.completion_tokens),
        )
        if self.provider.audit_config().get("paid") and total_tokens <= 0:
            raise ProviderError(
                "Paid provider omitted token usage; budget accounting fails closed"
            )
        cost = (
            self.pricing.cost(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cached_tokens=usage.cached_tokens,
            )
            if self.pricing
            else 0.0
        )
        self.ledger.account(
            f"{self.stage}:model_call",
            {
                "api_prompt_tokens": usage.prompt_tokens,
                "api_completion_tokens": usage.completion_tokens,
                "api_cached_tokens": usage.cached_tokens,
                "api_total_tokens": total_tokens,
                "paid_cost": cost,
            },
            metadata={
                "provider": self.name,
                "model": response.model or self.model,
                "provider_response_id": response.provider_response_id,
                "pricing": self.pricing.to_dict() if self.pricing else None,
            },
        )
        return response

    def _maximum_completion_tokens(self) -> int:
        config = self.provider.audit_config()
        parameters = config.get("request_parameters") or {}
        value = parameters.get("max_completion_tokens", parameters.get("max_tokens", 0))
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return 0
        return value

    @staticmethod
    def _estimate_prompt_tokens(
        messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]
    ) -> int:
        serialized = json.dumps(
            {"messages": list(messages), "tools": list(tools)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        # Conservative enough for mixed Chinese/JSON prompts while remaining provider-neutral.
        return max(1, (len(serialized.encode("utf-8")) + 2) // 3)

    def _install_retry_accounting(self) -> None:
        if not hasattr(self.provider, "progress_callback"):
            return
        previous = getattr(self.provider, "progress_callback")

        def progress(event: str, payload: dict[str, Any]) -> None:
            if event == "provider_retry_scheduled":
                self.ledger.precheck(
                    f"{self.stage}:provider_retry", {"provider_retries": 1}
                )
                self.ledger.account(
                    f"{self.stage}:provider_retry",
                    {"provider_retries": 1},
                    metadata={"provider": self.name, **payload},
                )
            if previous is not None:
                previous(event, payload)

        setattr(self.provider, "progress_callback", progress)
