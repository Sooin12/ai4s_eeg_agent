from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _runner_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_standard_epoch_multisubject_demo.py"
    spec = importlib.util.spec_from_file_location("multisubject_demo_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _envelope() -> dict:
    return {
        "envelope_id": "test-envelope",
        "resource_budget": {
            "max_research_cycles": 40,
            "max_candidate_executions": 100,
            "max_compute_seconds": 7200,
            "max_api_tokens": 2_000_000,
            "max_paid_cost": 6.0,
            "max_api_retries": 24,
        },
    }


@pytest.mark.parametrize(
    ("subjects", "expected_subject_candidates", "expected_retry_total"),
    [
        (["03", "04", "05"], 8, 22),
        (["02", "03", "04", "05"], 6, 24),
    ],
)
def test_budget_allocation_respects_global_caps(
    subjects: list[str], expected_subject_candidates: int, expected_retry_total: int
) -> None:
    runner = _runner_module()

    allocation = runner._allocate_budget(
        envelope=_envelope(), subjects=subjects, candidate_count=16
    )

    subject_limits = [allocation["per_subject"][item] for item in subjects]
    assert {item["candidate_executions"] for item in subject_limits} == {
        expected_subject_candidates
    }
    assert allocation["dataset_incumbent"]["research_cycles"] + sum(
        item["research_cycles"] for item in subject_limits
    ) <= 40
    assert allocation["dataset_incumbent"]["candidate_executions"] + sum(
        item["candidate_executions"] for item in subject_limits
    ) <= 100
    assert allocation["design"]["provider_retries"] + sum(
        item["provider_retries"] for item in subject_limits
    ) == expected_retry_total


def test_budget_allocation_fails_when_subject_search_cannot_be_funded() -> None:
    runner = _runner_module()
    envelope = _envelope()
    envelope["resource_budget"]["max_research_cycles"] = 20

    with pytest.raises(ValueError, match="at least two individualized candidates"):
        runner._allocate_budget(
            envelope=envelope, subjects=["03", "04", "05"], candidate_count=16
        )
