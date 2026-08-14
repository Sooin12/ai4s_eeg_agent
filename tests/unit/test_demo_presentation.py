from __future__ import annotations

from bci_autodiscovery.demo.presentation import _aggregate_decision


def _protocol() -> dict:
    return {
        "evaluation": {
            "aggregation": {
                "unit": "subject",
                "reducer": "macro_mean",
                "minimum_evaluable_units": 3,
                "missing_value_policy": "fail_closed",
            },
            "decision_policy": {
                "policy_version": "2.0",
                "chance_level": 0.5,
                "minimum_confirmation_score": 0.6,
                "maximum_search_to_confirmation_drop": 0.1,
                "minimum_distinct_search_candidates": 2,
                "minimum_evaluable_units": 3,
                "success_requires_all_thresholds": True,
                "below_chance_outcome": "refuse",
                "otherwise_outcome": "inconclusive",
                "confirmation_failure_outcome": "refuse",
            },
        }
    }


def _row(index: int) -> dict:
    return {
        "confirmation_access_count": 1,
        "search_candidates": 2,
        "confirmation_score": 0.8,
        "search_score": 0.82,
        "lock_critic": "pass",
        "scientific_critic": "pass",
        "pipeline_sha256": f"pipeline-{index}",
        "family": "csp_lda" if index < 2 else "bandpower_lda",
    }


def test_demo_aggregate_waits_for_all_subjects_and_independent_gates() -> None:
    complete = _aggregate_decision(_protocol(), [_row(0), _row(1), _row(2)])
    assert complete["outcome"] == "success"
    assert complete["observed"]["distinct_locked_pipeline_hashes"] == 3
    assert complete["criteria"]["minimum_evaluable_units"] is True

    incomplete = _aggregate_decision(_protocol(), [_row(0), _row(1)])
    assert incomplete["outcome"] == "inconclusive"
    assert incomplete["criteria"]["minimum_evaluable_units"] is False

    failed_review_rows = [_row(0), _row(1), _row(2)]
    failed_review_rows[2]["scientific_critic"] = "reject"
    failed_review = _aggregate_decision(_protocol(), failed_review_rows)
    assert failed_review["outcome"] == "inconclusive"
    assert failed_review["criteria"]["all_scientific_critics_passed"] is False
