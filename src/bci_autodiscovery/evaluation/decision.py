"""Machine-executable conclusion policy frozen before confirmation access."""

from __future__ import annotations

import hashlib
import json
from typing import Any


class FrozenDecisionError(ValueError):
    pass


def _hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_frozen_decision(
    *,
    protocol: dict[str, Any],
    pipeline_lock: dict[str, Any],
    confirmation_result: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a conclusion without delegating thresholds or outcome choice to an LLM."""

    if protocol.get("status") != "frozen_autonomous":
        raise FrozenDecisionError("Decision requires a frozen autonomous protocol")
    if pipeline_lock.get("status") != "locked_awaiting_confirmation":
        raise FrozenDecisionError("Decision requires a pre-confirmation pipeline lock")
    for field in ("dataset_id", "subject_id", "protocol_id", "lock_id"):
        expected = pipeline_lock.get(field)
        if field == "protocol_id":
            expected = protocol.get("protocol_id")
        if field == "dataset_id":
            expected = protocol.get("dataset_id")
        if confirmation_result.get(field) != expected:
            raise FrozenDecisionError(f"Confirmation result has mismatched {field}")

    policy = (protocol.get("evaluation") or {}).get("decision_policy")
    if not isinstance(policy, dict):
        raise FrozenDecisionError("Frozen protocol lacks a machine-executable decision policy")
    required = {
        "chance_level",
        "minimum_confirmation_score",
        "maximum_search_to_confirmation_drop",
        "minimum_distinct_search_candidates",
        "success_requires_all_thresholds",
        "below_chance_outcome",
        "otherwise_outcome",
        "confirmation_failure_outcome",
    }
    if set(policy) != required:
        raise FrozenDecisionError("Frozen decision policy has incomplete or unknown fields")
    if (
        policy.get("success_requires_all_thresholds") is not True
        or policy.get("below_chance_outcome") != "refuse"
        or policy.get("otherwise_outcome") != "inconclusive"
        or policy.get("confirmation_failure_outcome") != "refuse"
    ):
        raise FrozenDecisionError("Frozen decision policy has unsafe outcome semantics")

    trace = pipeline_lock.get("search_trace")
    if not isinstance(trace, list):
        raise FrozenDecisionError("Pipeline lock lacks search trace")
    distinct_configurations = {
        str(item.get("configuration_sha256") or item.get("pipeline_sha256"))
        for item in trace
        if isinstance(item, dict)
    }
    required_candidates = int(policy["minimum_distinct_search_candidates"])
    candidate_criterion = len(distinct_configurations) >= required_candidates

    if confirmation_result.get("status") != "completed_one_shot":
        return {
            "schema_version": "1.0",
            "outcome": "refuse",
            "reason_code": "confirmation_failed_or_incomplete",
            "policy_sha256": _hash(policy),
            "criteria": {
                "confirmation_completed_one_shot": False,
                "minimum_distinct_search_candidates": candidate_criterion,
            },
            "observed": {
                "distinct_search_candidates": len(distinct_configurations),
                "confirmation_status": confirmation_result.get("status"),
            },
            "thresholds": policy,
        }

    primary_metric = str((protocol.get("evaluation") or {}).get("primary_metric") or "")
    if confirmation_result.get("primary_metric") != primary_metric:
        raise FrozenDecisionError("Confirmation result uses another primary metric")
    if confirmation_result.get("selection_or_refit_after_confirmation") is not False:
        raise FrozenDecisionError("Confirmation result indicates post-confirmation selection/refit")
    if confirmation_result.get("search_reopened") is not False:
        raise FrozenDecisionError("Confirmation result indicates search reopening")

    confirmation_score = float(confirmation_result["confirmation_score"])
    search_score = float(confirmation_result["search_score"])
    observed_drop = search_score - confirmation_score
    score_criterion = confirmation_score >= float(policy["minimum_confirmation_score"])
    drop_criterion = observed_drop <= float(policy["maximum_search_to_confirmation_drop"])
    criteria = {
        "confirmation_completed_one_shot": True,
        "minimum_distinct_search_candidates": candidate_criterion,
        "minimum_confirmation_score": score_criterion,
        "maximum_search_to_confirmation_drop": drop_criterion,
    }
    if all(criteria.values()):
        outcome = "success"
        reason = "all_frozen_success_thresholds_met"
    elif confirmation_score < float(policy["chance_level"]):
        outcome = "refuse"
        reason = "confirmation_below_frozen_chance_level"
    else:
        outcome = "inconclusive"
        reason = "not_all_success_thresholds_met"
    return {
        "schema_version": "1.0",
        "outcome": outcome,
        "reason_code": reason,
        "policy_sha256": _hash(policy),
        "criteria": criteria,
        "observed": {
            "primary_metric": primary_metric,
            "search_score": search_score,
            "confirmation_score": confirmation_score,
            "search_to_confirmation_drop": observed_drop,
            "distinct_search_candidates": len(distinct_configurations),
        },
        "thresholds": policy,
    }
