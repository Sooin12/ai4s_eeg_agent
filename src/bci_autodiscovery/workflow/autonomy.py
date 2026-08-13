"""One-time authorization contract for autonomous research runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .dataset_contract import DatasetLevelContractError, load_dataset_level_contract


class AutonomyEnvelopeError(ValueError):
    pass


def sha256_path(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AutonomyEnvelopeError(f"Cannot load JSON contract {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise AutonomyEnvelopeError(f"Contract must be a JSON object: {resolved}")
    return value


def _nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AutonomyEnvelopeError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_integer(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AutonomyEnvelopeError(f"{field} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise AutonomyEnvelopeError(f"{field} must be >= {minimum}")
    return value


def validate_autonomy_envelope(
    envelope: dict[str, Any],
    *,
    expected_dataset_id: str | None = None,
    expected_dataset_contract_path: Path | None = None,
) -> None:
    """Fail closed when an autonomous run exceeds or lacks its one-time authority."""

    if envelope.get("schema_version") != "2.0":
        raise AutonomyEnvelopeError("Unsupported AutonomyEnvelope schema_version")
    _nonempty_text(envelope.get("envelope_id"), "envelope_id")
    if envelope.get("status") != "authorized":
        raise AutonomyEnvelopeError("AutonomyEnvelope status must be authorized")

    objective = envelope.get("objective")
    if not isinstance(objective, dict):
        raise AutonomyEnvelopeError("objective must be an object")
    _nonempty_text(objective.get("research_question"), "objective.research_question")
    _nonempty_text(objective.get("paradigm"), "objective.paradigm")
    if objective.get("conclusion_scope") != "internal_evidence_report":
        raise AutonomyEnvelopeError(
            "Autonomous runs may only produce internal_evidence_report conclusions"
        )

    dataset = envelope.get("dataset")
    if not isinstance(dataset, dict):
        raise AutonomyEnvelopeError("dataset must be an object")
    dataset_id = _nonempty_text(dataset.get("dataset_id"), "dataset.dataset_id")
    if expected_dataset_id is not None and dataset_id != expected_dataset_id:
        raise AutonomyEnvelopeError("AutonomyEnvelope belongs to another dataset")
    contract_path_text = _nonempty_text(
        dataset.get("dataset_level_contract_path"),
        "dataset.dataset_level_contract_path",
    )
    contract_hash = _nonempty_text(
        dataset.get("dataset_level_contract_sha256"),
        "dataset.dataset_level_contract_sha256",
    )
    contract_path = Path(contract_path_text).expanduser().resolve()
    if expected_dataset_contract_path is not None:
        expected_contract = Path(expected_dataset_contract_path).expanduser().resolve()
        if contract_path != expected_contract:
            raise AutonomyEnvelopeError(
                "AutonomyEnvelope authorizes a different DatasetLevelContract path"
            )
    if not contract_path.is_file() or sha256_path(contract_path) != contract_hash:
        raise AutonomyEnvelopeError(
            "Authorized DatasetLevelContract failed integrity check"
        )
    try:
        contract = load_dataset_level_contract(contract_path)
    except DatasetLevelContractError as exc:
        raise AutonomyEnvelopeError(
            f"Authorized DatasetLevelContract is invalid: {exc}"
        ) from exc
    if contract.get("dataset_id") != dataset_id:
        raise AutonomyEnvelopeError(
            "AutonomyEnvelope DatasetLevelContract belongs to another dataset"
        )

    budget = envelope.get("resource_budget")
    if not isinstance(budget, dict):
        raise AutonomyEnvelopeError("resource_budget must be an object")
    for field in (
        "max_research_cycles",
        "max_candidate_executions",
        "max_compute_seconds",
        "max_api_tokens",
    ):
        _positive_integer(
            budget.get(field),
            f"resource_budget.{field}",
            allow_zero=field == "max_api_tokens",
        )
    max_paid_cost = budget.get("max_paid_cost")
    if isinstance(max_paid_cost, bool) or not isinstance(max_paid_cost, (int, float)):
        raise AutonomyEnvelopeError("resource_budget.max_paid_cost must be a number")
    if float(max_paid_cost) < 0:
        raise AutonomyEnvelopeError("resource_budget.max_paid_cost must be >= 0")
    _nonempty_text(
        budget.get("paid_cost_currency"),
        "resource_budget.paid_cost_currency",
    )

    permissions = envelope.get("permissions")
    if not isinstance(permissions, dict):
        raise AutonomyEnvelopeError("permissions must be an object")
    for field in (
        "allow_network_literature",
        "allow_logical_exclusions",
        "allow_first_confirmation_access",
    ):
        if not isinstance(permissions.get(field), bool):
            raise AutonomyEnvelopeError(f"permissions.{field} must be boolean")

    forbidden = envelope.get("forbidden_actions")
    if not isinstance(forbidden, list) or not forbidden:
        raise AutonomyEnvelopeError("forbidden_actions must be a non-empty array")
    if any(not isinstance(item, str) or not item.strip() for item in forbidden):
        raise AutonomyEnvelopeError("forbidden_actions must contain non-empty strings")
    normalized_forbidden = {item.strip().lower() for item in forbidden}
    required_forbidden = {
        "modify_or_delete_raw_data",
        "use_confirmation_for_search",
        "publish_external_claims",
    }
    missing = sorted(required_forbidden.difference(normalized_forbidden))
    if missing:
        raise AutonomyEnvelopeError(
            f"AutonomyEnvelope lacks mandatory forbidden actions: {missing}"
        )

    confirmation = envelope.get("confirmation_policy")
    if not isinstance(confirmation, dict):
        raise AutonomyEnvelopeError("confirmation_policy must be an object")
    if confirmation.get("max_access_count") != 1:
        raise AutonomyEnvelopeError("Frozen confirmation access count must equal one")
    if confirmation.get("requires_pipeline_lock") is not True:
        raise AutonomyEnvelopeError("Frozen confirmation must require a pipeline lock")
    if confirmation.get("reopen_search_after_confirmation") is not False:
        raise AutonomyEnvelopeError("Search cannot reopen after frozen confirmation")


def load_autonomy_envelope(
    path: Path,
    *,
    expected_dataset_id: str | None = None,
    expected_dataset_contract_path: Path | None = None,
) -> dict[str, Any]:
    envelope = load_json_object(path)
    validate_autonomy_envelope(
        envelope,
        expected_dataset_id=expected_dataset_id,
        expected_dataset_contract_path=expected_dataset_contract_path,
    )
    return envelope


def budget_subset(
    requested: dict[str, Any], authorized: dict[str, Any]
) -> None:
    """Verify a child protocol budget stays inside its authorized envelope."""

    for field in (
        "max_research_cycles",
        "max_candidate_executions",
        "max_compute_seconds",
        "max_api_tokens",
        "max_paid_cost",
    ):
        value = requested.get(field)
        limit = authorized.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AutonomyEnvelopeError(f"Protocol budget {field} must be numeric")
        if value < 0 or value > limit:
            raise AutonomyEnvelopeError(
                f"Protocol budget {field}={value} exceeds authorized limit {limit}"
            )
    requested_currency = _nonempty_text(
        requested.get("paid_cost_currency"),
        "Protocol budget paid_cost_currency",
    )
    authorized_currency = _nonempty_text(
        authorized.get("paid_cost_currency"),
        "Authorized budget paid_cost_currency",
    )
    if requested_currency != authorized_currency:
        raise AutonomyEnvelopeError(
            "Protocol paid-cost currency differs from the AutonomyEnvelope"
        )
