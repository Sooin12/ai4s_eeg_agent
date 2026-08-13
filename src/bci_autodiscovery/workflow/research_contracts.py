"""Deterministic contracts used by the outcome-blind Research Design stage."""

from __future__ import annotations

import hashlib
import json
from typing import Any


RESEARCH_PROTOCOL_SCHEMA_VERSION = "3.0"
UNIT_CATALOG_SCHEMA_VERSION = "1.0"
CANDIDATE_UNIVERSE_SCHEMA_VERSION = "1.0"

METRIC_REGISTRY = {
    "accuracy": {"direction": "maximize", "value_range": [0.0, 1.0]},
    "balanced_accuracy": {"direction": "maximize", "value_range": [0.0, 1.0]},
    "cohen_kappa": {"direction": "maximize", "value_range": [-1.0, 1.0]},
    "macro_f1": {"direction": "maximize", "value_range": [0.0, 1.0]},
    "roc_auc": {"direction": "maximize", "value_range": [0.0, 1.0]},
}

STATISTICAL_TEST_REGISTRY = {
    "paired_permutation",
    "wilcoxon_signed_rank",
}


class ResearchContractError(ValueError):
    pass


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_ids(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result = sorted({str(value).strip() for value in values if str(value).strip()})
    return result


def build_authoritative_unit_catalog(profile: dict[str, Any]) -> dict[str, Any]:
    """Expose only unit identifiers explicitly present in the normalized profile.

    Counts are deliberately insufficient: a planner may not invent subject/run identifiers
    from a count. Session indices are authoritative because DatasetProfile validates the
    complete observed index set.
    """

    dataset_id = str((profile.get("dataset") or {}).get("id") or "")
    sessions = profile.get("sessions") or {}
    catalogs: dict[str, dict[str, Any]] = {}
    session_ids = _normalized_ids(sessions.get("session_indices"))
    if session_ids:
        catalogs["session"] = {
            "unit_ids": session_ids,
            "source_field": "DatasetProfile.sessions.session_indices",
            "coverage": "complete_observed_set",
        }
    explicit_sources = {
        "run": (
            ("sessions", "run_ids"),
            ("volume", "run_ids"),
        ),
        "subject": (
            ("sessions", "subject_ids"),
            ("volume", "subject_ids"),
        ),
        "trial_group": (
            ("events", "trial_group_ids"),
            ("volume", "trial_group_ids"),
        ),
    }
    for unit_kind, candidates in explicit_sources.items():
        for section, field in candidates:
            values = _normalized_ids((profile.get(section) or {}).get(field))
            if values:
                catalogs[unit_kind] = {
                    "unit_ids": values,
                    "source_field": f"DatasetProfile.{section}.{field}",
                    "coverage": "explicit_profile_set",
                }
                break
    unsigned = {
        "schema_version": UNIT_CATALOG_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "catalogs": catalogs,
    }
    return {**unsigned, "catalog_sha256": canonical_json_sha256(unsigned)}


def build_candidate_universe_contract(
    dataset_contract: dict[str, Any], *, dataset_contract_sha256: str
) -> dict[str, Any]:
    """Freeze the finite pre-activation universe definition without claiming executability."""

    canonical_space = dataset_contract.get("canonical_space") or {}
    dimensions = canonical_space.get("dimensions") or {}
    components: list[dict[str, str]] = []
    if isinstance(dimensions, dict):
        for dimension, entries in sorted(dimensions.items()):
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                component_id = str(entry.get("component_id") or "").strip()
                if component_id:
                    components.append(
                        {"dimension": str(dimension), "component_id": component_id}
                    )
    components.sort(key=lambda item: (item["dimension"], item["component_id"]))
    if not components:
        raise ResearchContractError(
            "DatasetLevelContract contains no canonical candidate components"
        )
    definition = {
        "schema_version": CANDIDATE_UNIVERSE_SCHEMA_VERSION,
        "source": "DatasetLevelContract.canonical_space",
        "source_contract_sha256": dataset_contract_sha256,
        "selector_rule_id": "all_frozen_canonical_components_v1",
        "components": components,
        "frontier_semantics": "hypothesis_only_not_effective_method",
        "materialization_gate": (
            "method_engineering_must_emit_finite_executable_pipeline_ids_before_search"
        ),
    }
    return {
        **definition,
        "candidate_universe_sha256": canonical_json_sha256(definition),
    }


def external_authority_requirements(
    dataset_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    deferred = dataset_contract.get("deferred_to_downstream_agents") or {}
    if not isinstance(deferred, dict):
        deferred = {}
    blockers = deferred.get("external_authority_blockers") or []
    result: list[dict[str, Any]] = []
    for raw in blockers:
        description = str(raw).strip()
        if not description:
            continue
        result.append(
            {
                "blocker_id": "external-" + canonical_json_sha256(description)[:12],
                "description": description,
                "owner": "external_authority",
                "status": "unresolved",
                "blocks": ["pipeline_execution", "confirmation_access"],
            }
        )
    return result


def validate_unit_partition(
    *, proposal: dict[str, Any], unit_catalog: dict[str, Any]
) -> None:
    split_unit = proposal.get("split_unit")
    catalog = (unit_catalog.get("catalogs") or {}).get(split_unit)
    if not isinstance(catalog, dict):
        raise ResearchContractError(
            f"split_unit {split_unit!r} has no authoritative identifier catalog"
        )
    if proposal.get("unit_catalog_sha256") != unit_catalog.get("catalog_sha256"):
        raise ResearchContractError("Protocol is not bound to the authoritative unit catalog")
    roles = proposal.get("data_roles")
    expected_names = {
        "profiling_and_calibration",
        "pipeline_search_and_lock",
        "frozen_confirmation",
    }
    if not isinstance(roles, dict) or set(roles) != expected_names:
        raise ResearchContractError(
            f"data_roles must contain exactly {sorted(expected_names)}"
        )
    normalized: dict[str, set[str]] = {}
    for name in expected_names:
        values = roles.get(name)
        if not isinstance(values, list) or not values:
            raise ResearchContractError(f"data_roles.{name} must be non-empty")
        normalized[name] = {str(value) for value in values}
        if len(normalized[name]) != len(values):
            raise ResearchContractError(f"data_roles.{name} contains duplicate units")
    if any(
        normalized[left].intersection(normalized[right])
        for left in expected_names
        for right in expected_names
        if left < right
    ):
        raise ResearchContractError("Research protocol data roles must be disjoint")
    observed = set(catalog.get("unit_ids") or [])
    assigned = set().union(*normalized.values())
    if assigned != observed:
        raise ResearchContractError(
            "Data roles must cover the authoritative unit catalog exactly; "
            f"missing={sorted(observed - assigned)}, extra={sorted(assigned - observed)}"
        )


def validate_candidate_universe_binding(
    *, proposal: dict[str, Any], expected: dict[str, Any]
) -> None:
    oracle = proposal.get("individual_oracle") or {}
    universe = oracle.get("candidate_universe")
    if not isinstance(universe, dict):
        raise ResearchContractError("individual_oracle.candidate_universe must be an object")
    required = {
        "schema_version": expected["schema_version"],
        "source": expected["source"],
        "source_contract_sha256": expected["source_contract_sha256"],
        "selector_rule_id": expected["selector_rule_id"],
        "candidate_universe_sha256": expected["candidate_universe_sha256"],
        "frontier_semantics": expected["frontier_semantics"],
        "materialization_gate": expected["materialization_gate"],
    }
    for field, value in required.items():
        if universe.get(field) != value:
            raise ResearchContractError(
                f"individual_oracle.candidate_universe.{field} changed authority"
            )


def validate_evaluation_contract(evaluation: Any) -> None:
    if not isinstance(evaluation, dict):
        raise ResearchContractError("evaluation must be an object")
    primary = evaluation.get("primary_metric")
    if primary not in METRIC_REGISTRY:
        raise ResearchContractError("evaluation.primary_metric is not a registered metric ID")
    secondary = evaluation.get("secondary_metrics")
    if not isinstance(secondary, list) or any(
        item not in METRIC_REGISTRY or item == primary for item in secondary
    ):
        raise ResearchContractError("evaluation.secondary_metrics contains invalid metric IDs")
    if len(set(secondary)) != len(secondary):
        raise ResearchContractError("evaluation.secondary_metrics contains duplicates")

    aggregation = evaluation.get("aggregation")
    if not isinstance(aggregation, dict):
        raise ResearchContractError("evaluation.aggregation must be an object")
    if aggregation.get("unit") not in {"subject", "subject_session"}:
        raise ResearchContractError("evaluation.aggregation.unit is unsupported")
    if aggregation.get("reducer") not in {"macro_mean", "median"}:
        raise ResearchContractError("evaluation.aggregation.reducer is unsupported")
    if aggregation.get("missing_value_policy") not in {
        "fail_closed",
        "complete_case_with_audit",
    }:
        raise ResearchContractError("evaluation aggregation missing-value policy is unsafe")
    minimum_units = aggregation.get("minimum_evaluable_units")
    if isinstance(minimum_units, bool) or not isinstance(minimum_units, int) or minimum_units < 2:
        raise ResearchContractError("minimum_evaluable_units must be an integer >= 2")

    tests = evaluation.get("statistical_analysis")
    if not isinstance(tests, list) or not tests:
        raise ResearchContractError("evaluation.statistical_analysis must be non-empty")
    test_ids: set[str] = set()
    for item in tests:
        if not isinstance(item, dict):
            raise ResearchContractError("Every statistical analysis must be an object")
        analysis_id = str(item.get("analysis_id") or "").strip()
        if not analysis_id or analysis_id in test_ids:
            raise ResearchContractError("Statistical analysis IDs must be unique")
        test_ids.add(analysis_id)
        if item.get("test_id") not in STATISTICAL_TEST_REGISTRY:
            raise ResearchContractError(f"Statistical analysis {analysis_id} uses unknown test")
        if item.get("alternative") not in {"two_sided", "greater", "less"}:
            raise ResearchContractError(f"Statistical analysis {analysis_id} has bad alternative")
        alpha = item.get("alpha")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
            raise ResearchContractError(f"Statistical analysis {analysis_id} has bad alpha")
        permutations = item.get("permutations")
        if isinstance(permutations, bool) or not isinstance(permutations, int) or permutations < 999:
            raise ResearchContractError(
                f"Statistical analysis {analysis_id} needs at least 999 permutations"
            )
        if not isinstance(item.get("random_seed"), int):
            raise ResearchContractError(f"Statistical analysis {analysis_id} lacks integer seed")
        correction = item.get("multiple_comparison") or {}
        if correction.get("method") not in {"none", "holm"}:
            raise ResearchContractError(
                f"Statistical analysis {analysis_id} has unsupported correction"
            )
        if not str(correction.get("family_id") or "").strip():
            raise ResearchContractError(
                f"Statistical analysis {analysis_id} lacks comparison family"
            )

    interval = evaluation.get("confidence_interval") or {}
    if interval.get("method") not in {"bootstrap_percentile", "bootstrap_bca"}:
        raise ResearchContractError("Unsupported confidence interval method")
    level = interval.get("level")
    if isinstance(level, bool) or not isinstance(level, (int, float)) or not 0 < level < 1:
        raise ResearchContractError("confidence_interval.level must be in (0, 1)")
    resamples = interval.get("resamples")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 999:
        raise ResearchContractError("confidence_interval.resamples must be >= 999")
    if not isinstance(interval.get("random_seed"), int):
        raise ResearchContractError("confidence_interval.random_seed must be an integer")


def validate_stopping_policy(policy: Any) -> None:
    if not isinstance(policy, dict) or policy.get("policy_version") != "1.0":
        raise ResearchContractError("stopping_policy must use policy_version 1.0")
    for field in (
        "stop_on_budget_exhaustion",
        "stop_on_candidate_universe_exhaustion",
    ):
        if policy.get(field) is not True:
            raise ResearchContractError(f"stopping_policy.{field} must be true")
    plateau = policy.get("plateau")
    if not isinstance(plateau, dict) or not isinstance(plateau.get("enabled"), bool):
        raise ResearchContractError("stopping_policy.plateau is invalid")
    for field in ("minimum_candidates", "patience"):
        value = plateau.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ResearchContractError(f"stopping_policy.plateau.{field} must be positive")
    improvement = plateau.get("minimum_improvement")
    if isinstance(improvement, bool) or not isinstance(improvement, (int, float)) or improvement < 0:
        raise ResearchContractError(
            "stopping_policy.plateau.minimum_improvement must be non-negative"
        )


def validate_quality_policy(policy: Any, *, logical_exclusions_allowed: bool) -> None:
    if not isinstance(policy, dict) or policy.get("policy_version") != "1.0":
        raise ResearchContractError("quality_anomaly_policy must use policy_version 1.0")
    if policy.get("unknown_metadata_action") != "block_dependent_operation":
        raise ResearchContractError("Unknown metadata must block dependent operations")
    if policy.get("default_action") != "retain_and_flag":
        raise ResearchContractError("Quality policy default action must retain and flag")
    if policy.get("exclusions_require_audit_record") is not True:
        raise ResearchContractError("Every logical exclusion must require an audit record")
    rules = policy.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ResearchContractError("quality_anomaly_policy.rules must be non-empty")
    rule_ids: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict):
            raise ResearchContractError("Every quality rule must be an object")
        rule_id = str(rule.get("rule_id") or "").strip()
        if not rule_id or rule_id in rule_ids:
            raise ResearchContractError("Quality rule IDs must be unique")
        rule_ids.add(rule_id)
        predicate = rule.get("predicate") or {}
        if predicate.get("operator") not in {
            "eq",
            "gt",
            "gte",
            "lt",
            "lte",
            "in",
            "is_unknown",
        }:
            raise ResearchContractError(f"Quality rule {rule_id} has invalid operator")
        if not str(predicate.get("field") or "").strip():
            raise ResearchContractError(f"Quality rule {rule_id} lacks predicate field")
        action = rule.get("action")
        if action not in {"retain_and_flag", "logical_exclude", "fail_stage"}:
            raise ResearchContractError(f"Quality rule {rule_id} has invalid action")
        if action == "logical_exclude" and not logical_exclusions_allowed:
            raise ResearchContractError("Protocol requests unauthorized logical exclusions")
        if not str(rule.get("reason_code") or "").strip():
            raise ResearchContractError(f"Quality rule {rule_id} lacks reason_code")
        refs = rule.get("evidence_refs")
        if not isinstance(refs, list) or not refs or any(
            not isinstance(item, str) or not item.strip() for item in refs
        ):
            raise ResearchContractError(f"Quality rule {rule_id} lacks evidence references")
