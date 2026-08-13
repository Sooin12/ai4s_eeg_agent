"""Consumer-side validation for frozen DatasetLevelContract artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bci_autodiscovery.agents.dataset_intelligence_critic import (
    DatasetIntelligenceCriticError,
    validate_dataset_critique,
)
from bci_autodiscovery.profiling import (
    DatasetProfileError,
    validate_dataset_profile_provenance,
)


class DatasetLevelContractError(ValueError):
    """A frozen dataset-level artifact cannot be trusted by a downstream stage."""


_PROVENANCE_FIELDS = {
    "dataset_profile",
    "canonical_search_space",
    "component_registry",
    "evidence_db",
}

_STAGE_BOUNDARY_FIELDS = {
    "session_roles_assigned",
    "evaluation_metrics_selected",
    "experiment_budget_allocated",
    "subject_data_accessed",
    "confirmation_data_accessed",
    "execution_activation_performed",
}

_DRAFT_BOUND_FIELDS = {
    "dataset_id",
    "canonical_space",
    "compatibility_rules",
    "dataset_hard_constraints",
    "deferred_to_downstream_agents",
    "excluded_components",
    "frontier_space",
    "provenance",
    "stage_boundary",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetLevelContractError(
            f"Cannot load {field} JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise DatasetLevelContractError(f"{field} must be a JSON object: {path}")
    return value


def _nonempty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetLevelContractError(f"{field} must be a non-empty string")
    return value.strip()


def _resolve_hashed_reference(reference: Any, *, field: str) -> Path:
    if not isinstance(reference, dict):
        raise DatasetLevelContractError(f"{field} must be a path/SHA object")
    path_text = _nonempty_text(reference.get("path"), field=f"{field}.path")
    expected_hash = _nonempty_text(reference.get("sha256"), field=f"{field}.sha256")
    if len(expected_hash) != 64:
        raise DatasetLevelContractError(f"{field}.sha256 must be a SHA-256 digest")
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        raise DatasetLevelContractError(f"{field}.path must be absolute")
    path = path.resolve()
    if not path.is_file() or _sha256(path) != expected_hash:
        raise DatasetLevelContractError(f"{field} provenance integrity check failed")
    return path


def validate_dataset_level_contract(
    contract: dict[str, Any],
    *,
    contract_path: Path,
) -> None:
    """Fail closed unless a contract is an intact Dataset-Critic-approved freeze."""

    resolved_contract = Path(contract_path).expanduser().resolve()
    if not resolved_contract.is_file():
        raise DatasetLevelContractError(
            f"DatasetLevelContract file does not exist: {resolved_contract}"
        )
    if _load_json(resolved_contract, field="DatasetLevelContract") != contract:
        raise DatasetLevelContractError(
            "DatasetLevelContract object differs from the exact file being validated"
        )

    if contract.get("schema_version") != "2.0":
        raise DatasetLevelContractError("Unsupported DatasetLevelContract schema_version")
    if contract.get("status") != "frozen_dataset_level_contract":
        raise DatasetLevelContractError(
            "DatasetLevelContract status must be frozen_dataset_level_contract"
        )
    dataset_id = _nonempty_text(contract.get("dataset_id"), field="dataset_id")
    _nonempty_text(contract.get("contract_id"), field="contract_id")
    _nonempty_text(contract.get("frozen_at_utc"), field="frozen_at_utc")

    stage_boundary = contract.get("stage_boundary")
    if (
        not isinstance(stage_boundary, dict)
        or set(stage_boundary) != _STAGE_BOUNDARY_FIELDS
    ):
        raise DatasetLevelContractError(
            "DatasetLevelContract stage_boundary has missing or unknown fields"
        )
    violated = sorted(
        field for field in _STAGE_BOUNDARY_FIELDS if stage_boundary.get(field) is not False
    )
    if violated:
        raise DatasetLevelContractError(
            f"DatasetLevelContract stage_boundary was exceeded: {violated}"
        )

    freeze_record = contract.get("freeze_record")
    if not isinstance(freeze_record, dict):
        raise DatasetLevelContractError(
            "DatasetLevelContract freeze_record must be an object"
        )
    required_freeze_values = {
        "human_itemized_approval_used": False,
        "network_discovery_completed": True,
        "executable_activation_performed": False,
        "session_protocol_created": False,
        "subject_or_confirmation_data_accessed": False,
    }
    invalid_freeze = sorted(
        field
        for field, expected in required_freeze_values.items()
        if freeze_record.get(field) is not expected
    )
    if invalid_freeze:
        raise DatasetLevelContractError(
            f"DatasetLevelContract freeze_record violates stage isolation: {invalid_freeze}"
        )
    draft_path = _resolve_hashed_reference(
        freeze_record.get("source_draft"),
        field="freeze_record.source_draft",
    )
    draft = _load_json(draft_path, field="DatasetLevelContract source draft")
    if draft.get("dataset_id") != dataset_id:
        raise DatasetLevelContractError(
            "DatasetLevelContract source draft belongs to another dataset"
        )
    changed_fields = sorted(
        field
        for field in _DRAFT_BOUND_FIELDS
        if contract.get(field) != draft.get(field)
    )
    if changed_fields:
        raise DatasetLevelContractError(
            "Frozen DatasetLevelContract differs from its Critic-reviewed draft: "
            f"{changed_fields}"
        )

    provenance = contract.get("provenance")
    if not isinstance(provenance, dict):
        raise DatasetLevelContractError("DatasetLevelContract provenance must be an object")
    missing_provenance = sorted(_PROVENANCE_FIELDS.difference(provenance))
    if missing_provenance:
        raise DatasetLevelContractError(
            f"DatasetLevelContract lacks provenance: {missing_provenance}"
        )
    provenance_paths = {
        field: _resolve_hashed_reference(
            provenance[field],
            field=f"provenance.{field}",
        )
        for field in sorted(_PROVENANCE_FIELDS)
    }
    profile = _load_json(
        provenance_paths["dataset_profile"],
        field="DatasetProfile",
    )
    try:
        validate_dataset_profile_provenance(
            profile,
            require_hashed_evidence=True,
            require_current_constraints=True,
        )
    except (DatasetProfileError, KeyError, TypeError, ValueError) as exc:
        raise DatasetLevelContractError(
            f"DatasetLevelContract DatasetProfile provenance is invalid: {exc}"
        ) from exc
    if profile["dataset"]["id"] != dataset_id:
        raise DatasetLevelContractError(
            "DatasetLevelContract and DatasetProfile belong to different datasets"
        )

    critic_reference = contract.get("dataset_critic")
    if (
        not isinstance(critic_reference, dict)
        or critic_reference.get("verdict") != "pass"
    ):
        raise DatasetLevelContractError("Dataset Critic verdict must be pass")
    critique_path = _resolve_hashed_reference(
        critic_reference,
        field="dataset_critic",
    )
    critique = _load_json(critique_path, field="Dataset Critique")
    if critique.get("verdict") != "pass":
        raise DatasetLevelContractError("Dataset Critic artifact verdict must be pass")
    source_draft = critique.get("source_draft")
    if not isinstance(source_draft, dict):
        raise DatasetLevelContractError("Dataset Critic lacks source_draft binding")
    if (
        Path(str(source_draft.get("path") or "")).expanduser().resolve() != draft_path
        or source_draft.get("sha256") != _sha256(draft_path)
    ):
        raise DatasetLevelContractError(
            "Dataset Critic is not bound to the exact frozen source draft"
        )
    try:
        validate_dataset_critique(
            critique,
            draft=draft,
            draft_sha256=_sha256(draft_path),
            deterministic_validation_passed=True,
        )
    except (DatasetIntelligenceCriticError, KeyError, TypeError, ValueError) as exc:
        raise DatasetLevelContractError(
            f"Dataset Critic artifact is invalid: {exc}"
        ) from exc


def load_dataset_level_contract(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    contract = _load_json(resolved, field="DatasetLevelContract")
    validate_dataset_level_contract(contract, contract_path=resolved)
    return contract


def dataset_profile_path_from_contract(contract: dict[str, Any]) -> Path:
    provenance = contract.get("provenance") or {}
    reference = provenance.get("dataset_profile")
    return _resolve_hashed_reference(reference, field="provenance.dataset_profile")
