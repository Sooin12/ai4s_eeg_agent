"""Dataset-neutral protocol planner; proposals never self-approve."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci_autodiscovery.profiling import validate_dataset_profile

from .contracts import AgentRunResult
from .language import DEFAULT_LANGUAGE_INSTRUCTION
from .runtime import AgentRuntime
from .tools import ToolDefinition, ToolRegistry


PROTOCOL_PLANNER_SYSTEM_PROMPT = f"""{DEFAULT_LANGUAGE_INSTRUCTION}
You are the Protocol Planner Agent in an auditable research system. First call
read_dataset_profile_contract to read the normalized dataset profile. From its sessions,
runs, events, volume, quality constraints, and requires_human_decision items, independently
propose the profiling/calibration, pipeline-search-and-lock, and frozen-confirmation data
roles. Do not use a developer-supplied concrete split and do not access raw signals.

The protocol must prevent leakage: the three role sets must be disjoint and cover the
selected split unit. Frozen-confirmation data cannot be used for thresholds, normalization,
feature fitting, model selection, early stopping, or candidate ranking. Explain the
rationale, alternatives, risks, and quality-anomaly policy. Call record_protocol_proposal
to record the draft, then stop. You have no protocol-approval tool. State clearly that the
draft awaits user review and has not become active."""


class ProtocolProposalError(ValueError):
    pass


def _load_profile(path: Path) -> dict[str, Any]:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolProposalError(f"Cannot load dataset profile: {exc}") from exc
    validate_dataset_profile(profile)
    return profile


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_proposal_schema() -> dict[str, Any]:
    """Return the shared proposal schema used by planner and reviewer agents."""
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string"},
            "protocol_id": {"type": "string"},
            "dataset_id": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["proposed_requires_human_approval"],
            },
            "split_unit": {
                "type": "string",
                "enum": ["session", "run", "subject", "trial_group"],
            },
            "data_roles": {
                "type": "object",
                "properties": {
                    "profiling_and_calibration": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "pipeline_search_and_lock": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "frozen_confirmation": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "profiling_and_calibration",
                    "pipeline_search_and_lock",
                    "frozen_confirmation",
                ],
                "additionalProperties": False,
            },
            "leakage_rules": {
                "type": "object",
                "properties": {
                    "confirmation_inaccessible_before_lock": {"type": "boolean"},
                    "confirmation_cannot_select_pipeline": {"type": "boolean"},
                    "confirmation_cannot_set_thresholds": {"type": "boolean"},
                    "all_fitting_training_partition_only": {"type": "boolean"},
                    "repeat_confirmation_access_requires_approval": {"type": "boolean"},
                },
                "required": [
                    "confirmation_inaccessible_before_lock",
                    "confirmation_cannot_select_pipeline",
                    "confirmation_cannot_set_thresholds",
                    "all_fitting_training_partition_only",
                    "repeat_confirmation_access_requires_approval",
                ],
                "additionalProperties": False,
            },
            "rationale": {"type": "array", "items": {"type": "string"}},
            "quality_anomaly_policy": {"type": "array", "items": {"type": "string"}},
            "alternatives_considered": {"type": "array", "items": {"type": "string"}},
            "risks_and_open_decisions": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "schema_version",
            "protocol_id",
            "dataset_id",
            "status",
            "split_unit",
            "data_roles",
            "leakage_rules",
            "rationale",
            "quality_anomaly_policy",
            "alternatives_considered",
            "risks_and_open_decisions",
        ],
        "additionalProperties": False,
    }


def validate_protocol_proposal(
    proposal: dict[str, Any], *, profile: dict[str, Any], dataset_id: str
) -> None:
    """Apply deterministic safety invariants to an agent-authored proposal."""
    if proposal["dataset_id"] != dataset_id:
        raise ProtocolProposalError("Protocol proposal belongs to another dataset")
    if proposal["status"] != "proposed_requires_human_approval":
        raise ProtocolProposalError("Protocol Agent cannot approve its own proposal")
    roles = proposal["data_roles"]
    role_names = {
        "profiling_and_calibration",
        "pipeline_search_and_lock",
        "frozen_confirmation",
    }
    if set(roles) != role_names:
        raise ProtocolProposalError(
            f"data_roles must contain exactly {sorted(role_names)}"
        )
    normalized = {name: set(str(value) for value in roles[name]) for name in role_names}
    if any(not values for values in normalized.values()):
        raise ProtocolProposalError("Every data role must contain at least one unit")
    if any(
        normalized[left].intersection(normalized[right])
        for left in role_names
        for right in role_names
        if left < right
    ):
        raise ProtocolProposalError("Protocol data roles must be disjoint")
    if proposal["split_unit"] == "session":
        observed = {
            str(value) for value in profile["sessions"].get("session_indices") or []
        }
        covered = set().union(*normalized.values())
        if covered != observed:
            raise ProtocolProposalError(
                f"Session protocol must cover observed sessions exactly: {sorted(observed)}"
            )
    rules = proposal["leakage_rules"]
    required_true = {
        "confirmation_inaccessible_before_lock",
        "confirmation_cannot_select_pipeline",
        "confirmation_cannot_set_thresholds",
        "all_fitting_training_partition_only",
    }
    if any(rules.get(name) is not True for name in required_true):
        raise ProtocolProposalError(
            f"Leakage rules must explicitly set true: {sorted(required_true)}"
        )


def create_protocol_planner_tools(
    *, dataset_profile_path: Path
) -> tuple[ToolRegistry, dict[str, Any]]:
    profile_path = Path(dataset_profile_path).expanduser().resolve()
    profile = _load_profile(profile_path)
    dataset_id = str(profile["dataset"]["id"])
    registry = ToolRegistry()
    profile_was_read = False
    recorded = False

    def read_profile() -> dict[str, Any]:
        nonlocal profile_was_read
        profile_was_read = True
        return profile

    def record_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
        nonlocal recorded
        if not profile_was_read:
            raise ProtocolProposalError(
                "read_dataset_profile_contract must be called before recording a proposal"
            )
        if recorded:
            raise ProtocolProposalError("Only one protocol proposal may be recorded per run")
        validate_protocol_proposal(proposal, profile=profile, dataset_id=dataset_id)
        recorded = True
        result = json.loads(json.dumps(proposal))
        result["activation_performed"] = False
        result["source_profile"] = {
            "path": str(profile_path),
            "sha256": _hash(profile_path),
        }
        return result

    registry.register(
        ToolDefinition(
            name="read_dataset_profile_contract",
            description=(
                "Read the authoritative normalized dataset profile used to propose a "
                "leakage-safe experimental protocol."
            ),
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="read_only_contract",
            tags=("read-only", "dataset-profile", "authoritative"),
        ),
        read_profile,
    )
    proposal_schema = protocol_proposal_schema()
    registry.register(
        ToolDefinition(
            name="record_protocol_proposal",
            description=(
                "Record one non-activated protocol proposal for explicit human review."
            ),
            input_schema={
                "type": "object",
                "properties": {"proposal": proposal_schema},
                "required": ["proposal"],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="scientific_proposal_only",
            tags=("local-write", "proposal", "human-approval-required"),
        ),
        record_proposal,
    )
    return registry, {
        "dataset_id": dataset_id,
        "dataset_profile_path": str(profile_path),
        "task": "propose_a_leakage_safe_protocol_without_activating_it",
    }


@dataclass
class ProtocolPlannerAgent:
    runtime: AgentRuntime
    context: dict[str, Any]

    def run(self) -> AgentRunResult:
        return self.runtime.run(
            system_prompt=PROTOCOL_PLANNER_SYSTEM_PROMPT,
            user_prompt=json.dumps(self.context, ensure_ascii=False, sort_keys=True),
        )
