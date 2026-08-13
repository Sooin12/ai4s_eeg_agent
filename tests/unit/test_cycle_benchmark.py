from __future__ import annotations

from bci_autodiscovery.evaluation import run_synthetic_cycle_benchmark


def test_cycle_benchmark_is_auditable_and_uses_no_confirmation_refit() -> None:
    result = run_synthetic_cycle_benchmark(
        {
            "schema_version": "1.0",
            "benchmark_id": "small-cycle-fixture",
            "random_seed": 20260806,
            "subjects_per_group": 1,
            "trials_per_class": 9,
            "near_oracle_tolerance": 0.02,
            "guided_minimum_cycles": 2,
            "guided_maximum_cycles": 4,
            "guided_stop_score": 0.75,
            "random_search_repetitions": 5,
        }
    )
    assert result["status"] == "completed_engineering_benchmark"
    assert result["scope"] == "synthetic_fixture_not_scientific_eeg_claim"
    assert result["summary"]["subject_count"] == 4
    assert result["summary"]["finite_oracle_candidate_count"] == 8
    assert all(
        subject["guided"]["confirmation_refit"] is False
        and subject["guided"]["locked_cycle"] <= 4
        for subject in result["subjects"]
    )
