"""Autonomous, dataset-neutral research-protocol planning agent."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci_autodiscovery.profiling import validate_dataset_profile
from bci_autodiscovery.workflow.autonomy import (
    AutonomyEnvelopeError,
    budget_subset,
    load_autonomy_envelope,
    load_json_object,
    sha256_path,
)
from bci_autodiscovery.workflow.dataset_contract import (
    DatasetLevelContractError,
    dataset_profile_path_from_contract,
    load_dataset_level_contract,
)
from bci_autodiscovery.workflow.research_contracts import (
    METRIC_REGISTRY,
    RESEARCH_PROTOCOL_SCHEMA_VERSION,
    ResearchContractError,
    build_authoritative_unit_catalog,
    build_candidate_universe_contract,
    external_authority_requirements,
    validate_candidate_universe_binding,
    validate_evaluation_contract,
    validate_quality_policy,
    validate_stopping_policy,
    validate_unit_partition,
)

from .contracts import AgentRunResult
from .language import DEFAULT_LANGUAGE_INSTRUCTION
from .runtime import AgentRuntime
from .tools import ToolArgumentError, ToolDefinition, ToolRegistry, validate_json_value


RESEARCH_PROTOCOL_PLANNER_SYSTEM_PROMPT = f"""{DEFAULT_LANGUAGE_INSTRUCTION}
You are the autonomous Research Protocol Planner in an auditable scientific system.
First call read_autonomous_research_context. Work only from the authorized frozen
DatasetLevelContract, its normalized DatasetProfile, the AutonomyEnvelope, and generated
decision requirements. Do not access raw signals or frozen-confirmation results.

Independently define the data roles, evaluation metrics, statistical tests, success and
refusal criteria, a machine-executable decision policy, finite individual oracle, resource
allocation, stopping conditions, and quality-anomaly policy. Resolve every decision requirement yourself with a stable decision
ID, evidence references, rationale, and calibrated confidence. Your choices must fit inside
the authorized budget and prevent all confirmation leakage.

Call record_research_protocol_proposal once. The proposal is not sent to a human for
itemized approval; an independent Critic Agent and deterministic validator will review it.
If evidence is insufficient, encode a conservative refusal criterion or blocker rather than
asking the user to make the scientific decision."""


RESEARCH_PROTOCOL_REVISION_SYSTEM_PROMPT = f"""{DEFAULT_LANGUAGE_INSTRUCTION}
You are the autonomous Research Protocol Planner revising an outcome-blind proposal after
an independent Critic review. First call read_protocol_revision_context. Apply every
required revision and emit a complete replacement protocol, not a patch. Preserve stable
decision IDs and the authorized DatasetLevelContract and AutonomyEnvelope bindings. Do not
ask a human to choose the scientific design and do not access experiment or confirmation results.
Call record_revised_research_protocol once; the result will be reviewed by a fresh Critic
turn and can freeze only after a pass verdict."""


class ResearchProtocolError(ValueError):
    pass


def _load_profile(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        profile = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchProtocolError(f"Cannot load DatasetProfile: {exc}") from exc
    validate_dataset_profile(profile)
    return profile


def _load_authorized_research_context(
    *,
    dataset_level_contract_path: Path,
    autonomy_envelope_path: Path,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    contract_path = Path(dataset_level_contract_path).expanduser().resolve()
    envelope_path = Path(autonomy_envelope_path).expanduser().resolve()
    try:
        contract = load_dataset_level_contract(contract_path)
        profile_path = dataset_profile_path_from_contract(contract)
        profile = _load_profile(profile_path)
        dataset_id = str(contract["dataset_id"])
        envelope = load_autonomy_envelope(
            envelope_path,
            expected_dataset_id=dataset_id,
            expected_dataset_contract_path=contract_path,
        )
    except (DatasetLevelContractError, AutonomyEnvelopeError) as exc:
        raise ResearchProtocolError(str(exc)) from exc
    if str(profile["dataset"]["id"]) != dataset_id:
        raise ResearchProtocolError(
            "DatasetLevelContract and DatasetProfile belong to different datasets"
        )
    return contract_path, contract, profile_path, profile, envelope_path, envelope


def _stable_decision_id(question: str) -> str:
    normalized = " ".join(question.strip().lower().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"profile-decision-{digest}"


def decision_requirements_from_profile(profile: dict[str, Any]) -> list[dict[str, str]]:
    constraints = profile.get("constraints", {})
    questions = constraints.get("requires_research_design_decision") or []
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in questions:
        question = str(raw).strip()
        if not question:
            continue
        decision_id = _stable_decision_id(question)
        if decision_id in seen:
            continue
        seen.add(decision_id)
        result.append(
            {
                "decision_id": decision_id,
                "question": question,
                "source": "DatasetProfile.constraints.requires_research_design_decision",
            }
        )
    return result


def research_protocol_schema() -> dict[str, Any]:
    role_array = {"type": "array", "items": {"type": "string"}}
    metric_ids = sorted(METRIC_REGISTRY)
    statistical_analysis = {
        "type": "object",
        "properties": {
            "analysis_id": {"type": "string"},
            "test_id": {
                "type": "string",
                "enum": ["paired_permutation", "wilcoxon_signed_rank"],
            },
            "comparison": {"type": "string"},
            "alternative": {
                "type": "string",
                "enum": ["two_sided", "greater", "less"],
            },
            "alpha": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
            "permutations": {"type": "integer", "minimum": 999},
            "random_seed": {"type": "integer"},
            "multiple_comparison": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "enum": ["none", "holm"]},
                    "family_id": {"type": "string"},
                },
                "required": ["method", "family_id"],
                "additionalProperties": False,
            },
        },
        "required": [
            "analysis_id",
            "test_id",
            "comparison",
            "alternative",
            "alpha",
            "permutations",
            "random_seed",
            "multiple_comparison",
        ],
        "additionalProperties": False,
    }
    quality_rule = {
        "type": "object",
        "properties": {
            "rule_id": {"type": "string"},
            "applies_to": {
                "type": "string",
                "enum": ["subject", "session", "run", "trial", "channel", "operation"],
            },
            "predicate": {
                "type": "object",
                "properties": {
                    "field": {"type": "string"},
                    "operator": {
                        "type": "string",
                        "enum": ["eq", "gt", "gte", "lt", "lte", "in", "is_unknown"],
                    },
                    "value": {},
                },
                "required": ["field", "operator", "value"],
                "additionalProperties": False,
            },
            "action": {
                "type": "string",
                "enum": ["retain_and_flag", "logical_exclude", "fail_stage"],
            },
            "reason_code": {"type": "string"},
            "evidence_refs": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "rule_id",
            "applies_to",
            "predicate",
            "action",
            "reason_code",
            "evidence_refs",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "schema_version": {
                "type": "string",
                "enum": [RESEARCH_PROTOCOL_SCHEMA_VERSION],
            },
            "protocol_id": {"type": "string"},
            "dataset_id": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["proposed_for_autonomous_review"],
            },
            "split_unit": {
                "type": "string",
                "enum": ["session", "run", "subject", "trial_group"],
            },
            "unit_catalog_sha256": {"type": "string"},
            "data_roles": {
                "type": "object",
                "properties": {
                    "profiling_and_calibration": role_array,
                    "pipeline_search_and_lock": role_array,
                    "frozen_confirmation": role_array,
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
                    "confirmation_access_once": {"type": "boolean"},
                    "confirmation_cannot_reopen_search": {"type": "boolean"},
                },
                "required": [
                    "confirmation_inaccessible_before_lock",
                    "confirmation_cannot_select_pipeline",
                    "confirmation_cannot_set_thresholds",
                    "all_fitting_training_partition_only",
                    "confirmation_access_once",
                    "confirmation_cannot_reopen_search",
                ],
                "additionalProperties": False,
            },
            "evaluation": {
                "type": "object",
                "properties": {
                    "primary_metric": {"type": "string", "enum": metric_ids},
                    "secondary_metrics": {
                        "type": "array",
                        "items": {"type": "string", "enum": metric_ids},
                        "uniqueItems": True,
                    },
                    "aggregation": {
                        "type": "object",
                        "properties": {
                            "unit": {
                                "type": "string",
                                "enum": ["subject", "subject_session"],
                            },
                            "reducer": {
                                "type": "string",
                                "enum": ["macro_mean", "median"],
                            },
                            "minimum_evaluable_units": {
                                "type": "integer",
                                "minimum": 2,
                            },
                            "missing_value_policy": {
                                "type": "string",
                                "enum": ["fail_closed", "complete_case_with_audit"],
                            },
                        },
                        "required": [
                            "unit",
                            "reducer",
                            "minimum_evaluable_units",
                            "missing_value_policy",
                        ],
                        "additionalProperties": False,
                    },
                    "statistical_analysis": {
                        "type": "array",
                        "items": statistical_analysis,
                    },
                    "confidence_interval": {
                        "type": "object",
                        "properties": {
                            "method": {
                                "type": "string",
                                "enum": ["bootstrap_percentile", "bootstrap_bca"],
                            },
                            "level": {
                                "type": "number",
                                "exclusiveMinimum": 0,
                                "exclusiveMaximum": 1,
                            },
                            "resamples": {"type": "integer", "minimum": 999},
                            "random_seed": {"type": "integer"},
                        },
                        "required": ["method", "level", "resamples", "random_seed"],
                        "additionalProperties": False,
                    },
                    "decision_policy": {
                        "type": "object",
                        "properties": {
                            "policy_version": {"type": "string", "enum": ["2.0"]},
                            "chance_level": {"type": "number"},
                            "minimum_confirmation_score": {"type": "number"},
                            "maximum_search_to_confirmation_drop": {"type": "number"},
                            "minimum_distinct_search_candidates": {"type": "integer"},
                            "minimum_evaluable_units": {"type": "integer"},
                            "success_requires_all_thresholds": {"type": "boolean"},
                            "below_chance_outcome": {
                                "type": "string",
                                "enum": ["refuse"],
                            },
                            "otherwise_outcome": {
                                "type": "string",
                                "enum": ["inconclusive"],
                            },
                            "confirmation_failure_outcome": {
                                "type": "string",
                                "enum": ["refuse"],
                            },
                        },
                        "required": [
                            "policy_version",
                            "chance_level",
                            "minimum_confirmation_score",
                            "maximum_search_to_confirmation_drop",
                            "minimum_distinct_search_candidates",
                            "minimum_evaluable_units",
                            "success_requires_all_thresholds",
                            "below_chance_outcome",
                            "otherwise_outcome",
                            "confirmation_failure_outcome",
                        ],
                        "additionalProperties": False,
                    },
                },
                "required": [
                    "primary_metric",
                    "secondary_metrics",
                    "aggregation",
                    "statistical_analysis",
                    "confidence_interval",
                    "decision_policy",
                ],
                "additionalProperties": False,
            },
            "individual_oracle": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["finite_individual_oracle"],
                    },
                    "candidate_universe": {
                        "type": "object",
                        "properties": {
                            "schema_version": {"type": "string", "enum": ["1.0"]},
                            "source": {
                                "type": "string",
                                "enum": ["DatasetLevelContract.canonical_space"],
                            },
                            "source_contract_sha256": {"type": "string"},
                            "selector_rule_id": {
                                "type": "string",
                                "enum": ["all_frozen_canonical_components_v1"],
                            },
                            "candidate_universe_sha256": {"type": "string"},
                            "frontier_semantics": {
                                "type": "string",
                                "enum": ["hypothesis_only_not_effective_method"],
                            },
                            "materialization_gate": {
                                "type": "string",
                                "enum": [
                                    "method_engineering_must_emit_finite_executable_pipeline_ids_before_search"
                                ],
                            },
                        },
                        "required": [
                            "schema_version",
                            "source",
                            "source_contract_sha256",
                            "selector_rule_id",
                            "candidate_universe_sha256",
                            "frontier_semantics",
                            "materialization_gate",
                        ],
                        "additionalProperties": False,
                    },
                    "selection_data_role": {
                        "type": "string",
                        "enum": ["pipeline_search_and_lock"],
                    },
                    "confirmation_use_forbidden": {"type": "boolean"},
                },
                "required": [
                    "kind",
                    "candidate_universe",
                    "selection_data_role",
                    "confirmation_use_forbidden",
                ],
                "additionalProperties": False,
            },
            "resource_budget": {
                "type": "object",
                "properties": {
                    "max_research_cycles": {"type": "integer"},
                    "max_candidate_executions": {"type": "integer"},
                    "max_compute_seconds": {"type": "integer"},
                    "max_api_tokens": {"type": "integer"},
                    "max_paid_cost": {"type": "number"},
                    "paid_cost_currency": {"type": "string"},
                },
                "required": [
                    "max_research_cycles",
                    "max_candidate_executions",
                    "max_compute_seconds",
                    "max_api_tokens",
                    "max_paid_cost",
                    "paid_cost_currency",
                ],
                "additionalProperties": False,
            },
            "stopping_policy": {
                "type": "object",
                "properties": {
                    "policy_version": {"type": "string", "enum": ["1.0"]},
                    "stop_on_budget_exhaustion": {"type": "boolean"},
                    "stop_on_candidate_universe_exhaustion": {"type": "boolean"},
                    "plateau": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean"},
                            "minimum_candidates": {"type": "integer", "minimum": 1},
                            "patience": {"type": "integer", "minimum": 1},
                            "minimum_improvement": {"type": "number", "minimum": 0},
                        },
                        "required": [
                            "enabled",
                            "minimum_candidates",
                            "patience",
                            "minimum_improvement",
                        ],
                        "additionalProperties": False,
                    },
                },
                "required": [
                    "policy_version",
                    "stop_on_budget_exhaustion",
                    "stop_on_candidate_universe_exhaustion",
                    "plateau",
                ],
                "additionalProperties": False,
            },
            "autonomous_decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "decision_id": {"type": "string"},
                        "question": {"type": "string"},
                        "decision": {"type": "string"},
                        "rationale": {"type": "string"},
                        "evidence_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "decision_id",
                        "question",
                        "decision",
                        "rationale",
                        "evidence_refs",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            },
            "quality_anomaly_policy": {
                "type": "object",
                "properties": {
                    "policy_version": {"type": "string", "enum": ["1.0"]},
                    "default_action": {
                        "type": "string",
                        "enum": ["retain_and_flag"],
                    },
                    "unknown_metadata_action": {
                        "type": "string",
                        "enum": ["block_dependent_operation"],
                    },
                    "exclusions_require_audit_record": {"type": "boolean"},
                    "rules": {"type": "array", "items": quality_rule},
                },
                "required": [
                    "policy_version",
                    "default_action",
                    "unknown_metadata_action",
                    "exclusions_require_audit_record",
                    "rules",
                ],
                "additionalProperties": False,
            },
            "execution_preconditions": {
                "type": "object",
                "properties": {
                    "external_authority_blockers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "blocker_id": {"type": "string"},
                                "description": {"type": "string"},
                                "owner": {
                                    "type": "string",
                                    "enum": ["external_authority"],
                                },
                                "status": {"type": "string", "enum": ["unresolved"]},
                                "blocks": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": ["pipeline_execution", "confirmation_access"],
                                    },
                                },
                            },
                            "required": [
                                "blocker_id",
                                "description",
                                "owner",
                                "status",
                                "blocks",
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["external_authority_blockers"],
                "additionalProperties": False,
            },
            "rationale": {"type": "array", "items": {"type": "string"}},
            "alternatives_considered": {
                "type": "array",
                "items": {"type": "string"},
            },
            "risks": {"type": "array", "items": {"type": "string"}},
            "unresolved_blockers": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "schema_version",
            "protocol_id",
            "dataset_id",
            "status",
            "split_unit",
            "unit_catalog_sha256",
            "data_roles",
            "leakage_rules",
            "evaluation",
            "individual_oracle",
            "resource_budget",
            "stopping_policy",
            "autonomous_decisions",
            "quality_anomaly_policy",
            "execution_preconditions",
            "rationale",
            "alternatives_considered",
            "risks",
            "unresolved_blockers",
        ],
        "additionalProperties": False,
    }


def _require_nonempty_strings(values: Any, field: str) -> None:
    if not isinstance(values, list) or not values:
        raise ResearchProtocolError(f"{field} must be a non-empty array")
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise ResearchProtocolError(f"{field} must contain non-empty strings")


def validate_research_protocol_proposal(
    proposal: dict[str, Any],
    *,
    dataset_contract: dict[str, Any],
    profile: dict[str, Any],
    envelope: dict[str, Any],
) -> None:
    """Apply outcome-blind deterministic invariants before critic review."""

    schema = research_protocol_schema()
    schema_fields = set(schema["properties"])
    allowed_metadata = {
        "activation_state",
        "dataset_level_contract",
        "autonomy_envelope",
        "autonomous_revision",
    }
    unknown = sorted(set(proposal).difference(schema_fields | allowed_metadata))
    if unknown:
        raise ResearchProtocolError(f"Research protocol has unknown fields: {unknown}")
    core = {field: proposal[field] for field in schema_fields if field in proposal}
    try:
        validate_json_value(core, schema, location="research_protocol")
    except ToolArgumentError as exc:
        raise ResearchProtocolError(str(exc)) from exc

    dataset_id = str(profile["dataset"]["id"])
    if dataset_contract.get("status") != "frozen_dataset_level_contract":
        raise ResearchProtocolError("Research Design requires a frozen DatasetLevelContract")
    if dataset_contract.get("dataset_id") != dataset_id:
        raise ResearchProtocolError("DatasetLevelContract belongs to another dataset")
    if proposal.get("schema_version") != RESEARCH_PROTOCOL_SCHEMA_VERSION:
        raise ResearchProtocolError("Unsupported research protocol schema_version")
    if proposal.get("dataset_id") != dataset_id:
        raise ResearchProtocolError("Research protocol belongs to another dataset")
    if envelope["dataset"]["dataset_id"] != dataset_id:
        raise ResearchProtocolError("AutonomyEnvelope belongs to another dataset")
    if proposal.get("status") != "proposed_for_autonomous_review":
        raise ResearchProtocolError("Planner output must await autonomous critic review")
    if not isinstance(proposal.get("protocol_id"), str) or not proposal["protocol_id"].strip():
        raise ResearchProtocolError("protocol_id must be non-empty")

    unit_catalog = build_authoritative_unit_catalog(profile)
    try:
        validate_unit_partition(proposal=proposal, unit_catalog=unit_catalog)
    except ResearchContractError as exc:
        raise ResearchProtocolError(str(exc)) from exc

    rules = proposal.get("leakage_rules") or {}
    required_true = {
        "confirmation_inaccessible_before_lock",
        "confirmation_cannot_select_pipeline",
        "confirmation_cannot_set_thresholds",
        "all_fitting_training_partition_only",
        "confirmation_access_once",
        "confirmation_cannot_reopen_search",
    }
    if any(rules.get(name) is not True for name in required_true):
        raise ResearchProtocolError(
            f"Leakage rules must explicitly set true: {sorted(required_true)}"
        )
    if not envelope["permissions"].get("allow_first_confirmation_access"):
        raise ResearchProtocolError("AutonomyEnvelope does not authorize confirmation access")

    evaluation = proposal.get("evaluation") or {}
    try:
        validate_evaluation_contract(evaluation)
    except ResearchContractError as exc:
        raise ResearchProtocolError(str(exc)) from exc
    decision_policy = evaluation.get("decision_policy")
    if not isinstance(decision_policy, dict):
        raise ResearchProtocolError("evaluation.decision_policy must be an object")
    if decision_policy.get("policy_version") != "2.0":
        raise ResearchProtocolError("evaluation.decision_policy must use version 2.0")
    numeric_thresholds: dict[str, float] = {}
    for field in (
        "chance_level",
        "minimum_confirmation_score",
        "maximum_search_to_confirmation_drop",
    ):
        raw = decision_policy.get(field)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ResearchProtocolError(f"evaluation.decision_policy.{field} must be numeric")
        numeric_thresholds[field] = float(raw)
        if not 0 <= numeric_thresholds[field] <= 1:
            raise ResearchProtocolError(
                f"evaluation.decision_policy.{field} must be in [0, 1]"
            )
    if numeric_thresholds["minimum_confirmation_score"] <= numeric_thresholds["chance_level"]:
        raise ResearchProtocolError(
            "minimum_confirmation_score must be greater than chance_level"
        )
    minimum_candidates = decision_policy.get("minimum_distinct_search_candidates")
    if (
        isinstance(minimum_candidates, bool)
        or not isinstance(minimum_candidates, int)
        or minimum_candidates < 2
    ):
        raise ResearchProtocolError(
            "minimum_distinct_search_candidates must be an integer >= 2"
        )
    minimum_units = decision_policy.get("minimum_evaluable_units")
    if (
        isinstance(minimum_units, bool)
        or not isinstance(minimum_units, int)
        or minimum_units < 2
        or minimum_units
        != int((evaluation.get("aggregation") or {}).get("minimum_evaluable_units") or 0)
    ):
        raise ResearchProtocolError(
            "decision policy and aggregation must share minimum_evaluable_units >= 2"
        )
    if decision_policy.get("success_requires_all_thresholds") is not True:
        raise ResearchProtocolError("Success must require all frozen numeric thresholds")
    if decision_policy.get("below_chance_outcome") != "refuse":
        raise ResearchProtocolError("Below-chance confirmation must produce refusal")
    if decision_policy.get("otherwise_outcome") != "inconclusive":
        raise ResearchProtocolError("Non-success, non-refusal outcome must be inconclusive")
    if decision_policy.get("confirmation_failure_outcome") != "refuse":
        raise ResearchProtocolError("Confirmation failure must produce refusal")
    try:
        validate_stopping_policy(proposal.get("stopping_policy"))
        validate_quality_policy(
            proposal.get("quality_anomaly_policy"),
            logical_exclusions_allowed=bool(
                envelope["permissions"].get("allow_logical_exclusions")
            ),
        )
    except ResearchContractError as exc:
        raise ResearchProtocolError(str(exc)) from exc

    oracle = proposal.get("individual_oracle") or {}
    if oracle.get("kind") != "finite_individual_oracle":
        raise ResearchProtocolError("Only a finite individual oracle is permitted")
    if oracle.get("selection_data_role") != "pipeline_search_and_lock":
        raise ResearchProtocolError("Individual oracle must use search-and-lock data only")
    if oracle.get("confirmation_use_forbidden") is not True:
        raise ResearchProtocolError("Individual oracle cannot read confirmation data")
    expected_universe = build_candidate_universe_contract(
        dataset_contract,
        dataset_contract_sha256=str(
            envelope["dataset"]["dataset_level_contract_sha256"]
        ),
    )
    try:
        validate_candidate_universe_binding(
            proposal=proposal,
            expected=expected_universe,
        )
    except ResearchContractError as exc:
        raise ResearchProtocolError(str(exc)) from exc

    try:
        budget_subset(proposal.get("resource_budget") or {}, envelope["resource_budget"])
    except AutonomyEnvelopeError as exc:
        raise ResearchProtocolError(str(exc)) from exc
    if minimum_candidates > int(proposal["resource_budget"]["max_candidate_executions"]):
        raise ResearchProtocolError(
            "Decision policy requires more candidates than the protocol budget permits"
        )
    stopping = proposal.get("stopping_policy") or {}
    plateau = stopping.get("plateau") or {}
    if int(plateau.get("minimum_candidates") or 0) > int(
        proposal["resource_budget"]["max_candidate_executions"]
    ):
        raise ResearchProtocolError(
            "Stopping plateau requires more candidates than the protocol budget permits"
        )

    expected_blockers = external_authority_requirements(dataset_contract)
    supplied_blockers = (
        (proposal.get("execution_preconditions") or {}).get(
            "external_authority_blockers"
        )
        or []
    )
    if supplied_blockers != expected_blockers:
        raise ResearchProtocolError(
            "execution_preconditions must preserve exact external-authority blockers"
        )

    requirements = {
        item["decision_id"]: item for item in decision_requirements_from_profile(profile)
    }
    decisions = proposal.get("autonomous_decisions")
    if not isinstance(decisions, list):
        raise ResearchProtocolError("autonomous_decisions must be an array")
    observed: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ResearchProtocolError("Every autonomous decision must be an object")
        decision_id = str(decision.get("decision_id") or "").strip()
        if not decision_id or decision_id in observed:
            raise ResearchProtocolError("Autonomous decision IDs must be non-empty and unique")
        observed[decision_id] = decision
        for field in ("question", "decision", "rationale"):
            if not isinstance(decision.get(field), str) or not decision[field].strip():
                raise ResearchProtocolError(
                    f"Autonomous decision {decision_id} has empty {field}"
                )
        _require_nonempty_strings(
            decision.get("evidence_refs"),
            f"autonomous_decisions[{decision_id}].evidence_refs",
        )
        confidence = decision.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ResearchProtocolError(f"Decision {decision_id} confidence must be numeric")
        if not 0 <= float(confidence) <= 1:
            raise ResearchProtocolError(f"Decision {decision_id} confidence must be in [0, 1]")
    if set(observed) != set(requirements):
        missing = sorted(set(requirements).difference(observed))
        extra = sorted(set(observed).difference(requirements))
        raise ResearchProtocolError(
            f"Autonomous decisions must resolve profile requirements exactly; missing={missing}, extra={extra}"
        )
    for decision_id, requirement in requirements.items():
        if observed[decision_id]["question"] != requirement["question"]:
            raise ResearchProtocolError(f"Decision {decision_id} changed its source question")
    if proposal.get("unresolved_blockers"):
        raise ResearchProtocolError("Protocol cannot advance while unresolved_blockers remain")


def validate_research_protocol_authority_bindings(
    proposal: dict[str, Any],
    *,
    dataset_contract_path: Path,
    autonomy_envelope_path: Path,
) -> None:
    expected = (
        (
            "dataset_level_contract",
            Path(dataset_contract_path).expanduser().resolve(),
        ),
        (
            "autonomy_envelope",
            Path(autonomy_envelope_path).expanduser().resolve(),
        ),
    )
    for field, expected_path in expected:
        binding = proposal.get(field)
        if not isinstance(binding, dict):
            raise ResearchProtocolError(f"Research protocol lacks {field} binding")
        bound_path = Path(str(binding.get("path") or "")).expanduser().resolve()
        if (
            bound_path != expected_path
            or binding.get("sha256") != sha256_path(expected_path)
        ):
            raise ResearchProtocolError(
                f"Research protocol {field} binding does not match exact authority"
            )


def create_research_protocol_planner_tools(
    *,
    dataset_level_contract_path: Path,
    autonomy_envelope_path: Path,
) -> tuple[ToolRegistry, dict[str, Any]]:
    (
        contract_path,
        contract,
        profile_path,
        profile,
        envelope_path,
        envelope,
    ) = _load_authorized_research_context(
        dataset_level_contract_path=dataset_level_contract_path,
        autonomy_envelope_path=autonomy_envelope_path,
    )
    dataset_id = str(contract["dataset_id"])
    unit_catalog = build_authoritative_unit_catalog(profile)
    candidate_universe = build_candidate_universe_contract(
        contract,
        dataset_contract_sha256=sha256_path(contract_path),
    )
    authority_blockers = external_authority_requirements(contract)

    registry = ToolRegistry()
    context_read = False
    recorded = False

    def read_context() -> dict[str, Any]:
        nonlocal context_read
        context_read = True
        return {
            "autonomy_envelope": envelope,
            "dataset_level_contract": contract,
            "dataset_profile": profile,
            "decision_requirements": decision_requirements_from_profile(profile),
            "authoritative_unit_catalog": unit_catalog,
            "candidate_universe_contract": candidate_universe,
            "registered_metric_ids": sorted(METRIC_REGISTRY),
            "external_authority_requirements": authority_blockers,
            "provenance": {
                "autonomy_envelope": {
                    "path": str(envelope_path),
                    "sha256": sha256_path(envelope_path),
                },
                "dataset_level_contract": {
                    "path": str(contract_path),
                    "sha256": sha256_path(contract_path),
                },
                "dataset_profile": {
                    "path": str(profile_path),
                    "sha256": sha256_path(profile_path),
                },
            },
        }

    def record_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
        nonlocal recorded
        if not context_read:
            raise ResearchProtocolError("read_autonomous_research_context must be called first")
        if recorded:
            raise ResearchProtocolError("Only one research protocol may be recorded per run")
        validate_research_protocol_proposal(
            proposal,
            dataset_contract=contract,
            profile=profile,
            envelope=envelope,
        )
        recorded = True
        result = json.loads(json.dumps(proposal))
        result["activation_state"] = {
            "protocol_frozen": False,
            "session_role_contract_activated": False,
            "raw_data_accessed": False,
            "confirmation_accessed": False,
            "pipeline_execution_started": False,
        }
        result["dataset_level_contract"] = {
            "path": str(contract_path),
            "sha256": sha256_path(contract_path),
            "contract_id": contract["contract_id"],
        }
        result["autonomy_envelope"] = {
            "path": str(envelope_path),
            "sha256": sha256_path(envelope_path),
            "envelope_id": envelope["envelope_id"],
        }
        return result

    registry.register(
        ToolDefinition(
            name="read_autonomous_research_context",
            description=(
                "Read the authoritative AutonomyEnvelope, DatasetProfile, and all stable "
                "decision requirements through the frozen DatasetLevelContract without "
                "accessing raw or confirmation data."
            ),
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            approval="never",
            decision_kind="read_only_research_contract",
            tags=("read-only", "autonomy", "protocol"),
        ),
        read_context,
    )
    registry.register(
        ToolDefinition(
            name="record_research_protocol_proposal",
            description=(
                "Record a complete outcome-blind protocol for independent autonomous review."
            ),
            input_schema={
                "type": "object",
                "properties": {"proposal": research_protocol_schema()},
                "required": ["proposal"],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="autonomous_scientific_protocol",
            tags=("local-write", "autonomous-decision", "critic-required"),
        ),
        record_proposal,
    )
    return registry, {
        "dataset_id": dataset_id,
        "envelope_id": envelope["envelope_id"],
        "dataset_level_contract_sha256": sha256_path(contract_path),
        "task": "design_complete_outcome_blind_research_protocol",
        "required_decision_count": len(decision_requirements_from_profile(profile)),
    }


def create_research_protocol_revision_tools(
    *,
    dataset_level_contract_path: Path,
    autonomy_envelope_path: Path,
    proposal_path: Path,
    critique_path: Path,
) -> tuple[ToolRegistry, dict[str, Any]]:
    (
        contract_path,
        contract,
        profile_path,
        profile,
        envelope_path,
        envelope,
    ) = _load_authorized_research_context(
        dataset_level_contract_path=dataset_level_contract_path,
        autonomy_envelope_path=autonomy_envelope_path,
    )
    current_path = Path(proposal_path).expanduser().resolve()
    critique_path = Path(critique_path).expanduser().resolve()
    dataset_id = str(contract["dataset_id"])
    unit_catalog = build_authoritative_unit_catalog(profile)
    candidate_universe = build_candidate_universe_contract(
        contract,
        dataset_contract_sha256=sha256_path(contract_path),
    )
    authority_blockers = external_authority_requirements(contract)
    current = load_json_object(current_path)
    critique = load_json_object(critique_path)
    current_hash = sha256_path(current_path)
    critique_hash = sha256_path(critique_path)
    if current.get("dataset_id") != dataset_id:
        raise ResearchProtocolError("Current proposal belongs to another dataset")
    if critique.get("dataset_id") != dataset_id:
        raise ResearchProtocolError("Critique belongs to another dataset")
    if critique.get("protocol_id") != current.get("protocol_id"):
        raise ResearchProtocolError("Critique belongs to another protocol")
    if critique.get("reviewed_protocol_sha256") != current_hash:
        raise ResearchProtocolError("Critique is not bound to the current proposal SHA")
    if critique.get("verdict") != "revise":
        raise ResearchProtocolError("Revision requires a critic verdict of revise")
    if not critique.get("required_revisions"):
        raise ResearchProtocolError("Critic did not provide concrete revision requirements")
    validate_research_protocol_authority_bindings(
        current,
        dataset_contract_path=contract_path,
        autonomy_envelope_path=envelope_path,
    )

    registry = ToolRegistry()
    context_read = False
    recorded = False

    def read_context() -> dict[str, Any]:
        nonlocal context_read
        context_read = True
        return {
            "autonomy_envelope": envelope,
            "dataset_level_contract": contract,
            "dataset_profile": profile,
            "decision_requirements": decision_requirements_from_profile(profile),
            "authoritative_unit_catalog": unit_catalog,
            "candidate_universe_contract": candidate_universe,
            "registered_metric_ids": sorted(METRIC_REGISTRY),
            "external_authority_requirements": authority_blockers,
            "current_proposal": current,
            "critic_review": critique,
            "provenance": {
                "current_proposal": {
                    "path": str(current_path),
                    "sha256": current_hash,
                },
                "critic_review": {
                    "path": str(critique_path),
                    "sha256": critique_hash,
                },
                "dataset_level_contract": {
                    "path": str(contract_path),
                    "sha256": sha256_path(contract_path),
                },
            },
        }

    def record_revision(
        revised_proposal: dict[str, Any],
        revision_summary: list[str],
        critic_resolution: list[str],
    ) -> dict[str, Any]:
        nonlocal recorded
        if not context_read:
            raise ResearchProtocolError("read_protocol_revision_context must be called first")
        if recorded:
            raise ResearchProtocolError("Only one revised protocol may be recorded per run")
        if not revision_summary or not critic_resolution:
            raise ResearchProtocolError(
                "Revision summary and itemized critic resolution must be non-empty"
            )
        validate_research_protocol_proposal(
            revised_proposal,
            dataset_contract=contract,
            profile=profile,
            envelope=envelope,
        )
        recorded = True
        result = json.loads(json.dumps(revised_proposal))
        result["activation_state"] = {
            "protocol_frozen": False,
            "session_role_contract_activated": False,
            "raw_data_accessed": False,
            "confirmation_accessed": False,
            "pipeline_execution_started": False,
        }
        result["dataset_level_contract"] = {
            "path": str(contract_path),
            "sha256": sha256_path(contract_path),
            "contract_id": contract["contract_id"],
        }
        result["autonomy_envelope"] = {
            "path": str(envelope_path),
            "sha256": sha256_path(envelope_path),
            "envelope_id": envelope["envelope_id"],
        }
        result["autonomous_revision"] = {
            "parent_path": str(current_path),
            "parent_sha256": current_hash,
            "critique_path": str(critique_path),
            "critique_sha256": critique_hash,
            "revision_summary": revision_summary,
            "critic_resolution": critic_resolution,
        }
        return result

    registry.register(
        ToolDefinition(
            name="read_protocol_revision_context",
            description=(
                "Read the current proposal, outcome-blind Critic findings, and authoritative "
                "contracts for an autonomous revision."
            ),
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            approval="never",
            decision_kind="read_only_protocol_revision",
            tags=("read-only", "critic-feedback", "outcome-blind"),
        ),
        read_context,
    )
    registry.register(
        ToolDefinition(
            name="record_revised_research_protocol",
            description="Record a complete revised protocol for another independent Critic turn.",
            input_schema={
                "type": "object",
                "properties": {
                    "revised_proposal": research_protocol_schema(),
                    "revision_summary": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "critic_resolution": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "revised_proposal",
                    "revision_summary",
                    "critic_resolution",
                ],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="autonomous_scientific_protocol_revision",
            tags=("local-write", "autonomous-decision", "critic-required"),
        ),
        record_revision,
    )
    return registry, {
        "dataset_id": dataset_id,
        "protocol_id": current.get("protocol_id"),
        "task": "resolve_critic_findings_without_outcome_access",
        "required_revision_count": len(critique["required_revisions"]),
    }


@dataclass
class ResearchProtocolPlannerAgent:
    runtime: AgentRuntime
    context: dict[str, Any]

    def run(self) -> AgentRunResult:
        return self.runtime.run(
            system_prompt=RESEARCH_PROTOCOL_PLANNER_SYSTEM_PROMPT,
            user_prompt=json.dumps(self.context, ensure_ascii=False, sort_keys=True),
        )


@dataclass
class ResearchProtocolRevisionAgent:
    runtime: AgentRuntime
    context: dict[str, Any]

    def run(self) -> AgentRunResult:
        return self.runtime.run(
            system_prompt=RESEARCH_PROTOCOL_REVISION_SYSTEM_PROMPT,
            user_prompt=json.dumps(self.context, ensure_ascii=False, sort_keys=True),
        )
