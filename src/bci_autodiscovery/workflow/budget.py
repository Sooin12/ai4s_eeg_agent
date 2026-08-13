"""Append-only, hash-chained budget accounting for autonomous research runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BudgetLedgerError(RuntimeError):
    pass


class BudgetExceededError(BudgetLedgerError):
    pass


COUNTER_FIELDS = (
    "research_cycles",
    "candidate_executions",
    "compute_seconds",
    "api_prompt_tokens",
    "api_completion_tokens",
    "api_cached_tokens",
    "api_total_tokens",
    "paid_cost",
    "provider_retries",
    "provider_failures",
    "recovery_attempts",
    "confirmation_accesses",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def limits_from_envelope(envelope: dict[str, Any]) -> dict[str, float]:
    budget = envelope["resource_budget"]
    confirmation = envelope["confirmation_policy"]
    return {
        "research_cycles": float(budget["max_research_cycles"]),
        "candidate_executions": float(budget["max_candidate_executions"]),
        "compute_seconds": float(budget["max_compute_seconds"]),
        "api_total_tokens": float(budget["max_api_tokens"]),
        "paid_cost": float(budget["max_paid_cost"]),
        "provider_retries": float(
            budget.get("max_api_retries", budget["max_research_cycles"])
        ),
        "recovery_attempts": float(
            budget.get("max_recovery_attempts", budget["max_research_cycles"])
        ),
        "confirmation_accesses": float(confirmation["max_access_count"]),
    }


class BudgetLedger:
    """Durable counter ledger; replay verifies every previous hash and total."""

    def __init__(
        self,
        path: Path,
        *,
        run_id: str,
        limits: dict[str, float],
        authority_sha256: str,
        create: bool,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.run_id = run_id
        self.limits = {key: float(value) for key, value in limits.items()}
        self.authority_sha256 = authority_sha256
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[dict[str, Any]] = []
        self._totals = {field: 0.0 for field in COUNTER_FIELDS}
        if create:
            if self.path.exists():
                raise BudgetLedgerError(f"Refusing to overwrite budget ledger: {self.path}")
            self._append(
                "ledger_opened",
                {},
                metadata={
                    "limits": self.limits,
                    "authority_sha256": authority_sha256,
                },
            )
        else:
            self._replay()

    @property
    def totals(self) -> dict[str, float]:
        return dict(self._totals)

    @property
    def event_count(self) -> int:
        return len(self._events)

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "authority_sha256": self.authority_sha256,
            "limits": dict(self.limits),
            "totals": self.totals,
            "remaining": {
                field: limit - self._totals.get(field, 0.0)
                for field, limit in self.limits.items()
            },
            "event_count": self.event_count,
            "last_event_sha256": (
                self._events[-1]["event_sha256"] if self._events else None
            ),
        }

    def precheck(self, operation: str, requested: dict[str, float]) -> dict[str, Any]:
        delta = self._normalize_delta(requested)
        exceeded = self._exceeded(delta)
        if exceeded:
            self._append(
                "precheck_rejected",
                {},
                metadata={
                    "operation": operation,
                    "requested": delta,
                    "exceeded": exceeded,
                },
            )
            raise BudgetExceededError(
                f"Budget precheck rejected {operation}: {exceeded}"
            )
        self._append(
            "precheck_passed",
            {},
            metadata={"operation": operation, "requested": delta},
        )
        return self.snapshot()

    def account(
        self,
        operation: str,
        delta: dict[str, float],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = self._normalize_delta(delta)
        self._append(
            "usage_recorded",
            normalized,
            metadata={"operation": operation, **(metadata or {})},
        )
        exceeded = self._exceeded({})
        if exceeded:
            self._append(
                "post_account_exceeded",
                {},
                metadata={"operation": operation, "exceeded": exceeded},
            )
            raise BudgetExceededError(
                f"Budget exceeded after accounting {operation}: {exceeded}"
            )
        return self.snapshot()

    def record_recovery(self, *, source_run_id: str) -> dict[str, Any]:
        self.precheck("recovery", {"recovery_attempts": 1})
        return self.account(
            "recovery",
            {"recovery_attempts": 1},
            metadata={"source_run_id": source_run_id},
        )

    def close(self, status: str) -> dict[str, Any]:
        self._append("ledger_closed", {}, metadata={"status": status})
        return self.snapshot()

    def _normalize_delta(self, delta: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(delta).difference(COUNTER_FIELDS))
        if unknown:
            raise BudgetLedgerError(f"Unknown budget counters: {unknown}")
        result: dict[str, float] = {}
        for field, raw in delta.items():
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise BudgetLedgerError(f"Budget counter {field} must be numeric")
            value = float(raw)
            if value < 0:
                raise BudgetLedgerError(f"Budget counter {field} cannot decrease")
            if value:
                result[field] = value
        return result

    def _exceeded(self, pending: dict[str, float]) -> dict[str, dict[str, float]]:
        exceeded: dict[str, dict[str, float]] = {}
        for field, limit in self.limits.items():
            projected = self._totals.get(field, 0.0) + pending.get(field, 0.0)
            if projected > limit + 1e-12:
                exceeded[field] = {"projected": projected, "limit": limit}
        return exceeded

    def _append(
        self,
        event_type: str,
        delta: dict[str, float],
        *,
        metadata: dict[str, Any],
    ) -> None:
        for field, value in delta.items():
            self._totals[field] += float(value)
        unsigned = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "sequence": len(self._events),
            "timestamp_utc": _utc_now(),
            "event_type": event_type,
            "previous_event_sha256": (
                self._events[-1]["event_sha256"] if self._events else None
            ),
            "delta": delta,
            "totals": self.totals,
            "metadata": metadata,
        }
        event = {**unsigned, "event_sha256": _hash(unsigned)}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        self._events.append(event)

    def _replay(self) -> None:
        if not self.path.is_file():
            raise BudgetLedgerError(f"Budget ledger does not exist: {self.path}")
        previous: str | None = None
        totals = {field: 0.0 for field in COUNTER_FIELDS}
        events: list[dict[str, Any]] = []
        for index, line in enumerate(self.path.read_text(encoding="utf-8").splitlines()):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BudgetLedgerError(f"Invalid budget ledger line {index + 1}: {exc}") from exc
            if not isinstance(event, dict):
                raise BudgetLedgerError(f"Budget ledger line {index + 1} is not an object")
            if event.get("run_id") != self.run_id or event.get("sequence") != index:
                raise BudgetLedgerError("Budget ledger run or sequence mismatch")
            if event.get("previous_event_sha256") != previous:
                raise BudgetLedgerError("Budget ledger hash chain is broken")
            unsigned = {key: value for key, value in event.items() if key != "event_sha256"}
            if event.get("event_sha256") != _hash(unsigned):
                raise BudgetLedgerError("Budget ledger event integrity check failed")
            delta = self._normalize_delta(event.get("delta") or {})
            for field, value in delta.items():
                totals[field] += value
            recorded_totals = event.get("totals") or {}
            if any(
                abs(float(recorded_totals.get(field, 0.0)) - totals[field]) > 1e-12
                for field in COUNTER_FIELDS
            ):
                raise BudgetLedgerError("Budget ledger cumulative totals are inconsistent")
            if index == 0:
                metadata = event.get("metadata") or {}
                if metadata.get("authority_sha256") != self.authority_sha256:
                    raise BudgetLedgerError("Budget ledger authority binding changed")
                recorded_limits = {
                    key: float(value)
                    for key, value in (metadata.get("limits") or {}).items()
                }
                if recorded_limits != self.limits:
                    raise BudgetLedgerError("Budget ledger limits changed")
            previous = str(event["event_sha256"])
            events.append(event)
        if not events or events[0].get("event_type") != "ledger_opened":
            raise BudgetLedgerError("Budget ledger lacks its opening authority event")
        self._events = events
        self._totals = totals
