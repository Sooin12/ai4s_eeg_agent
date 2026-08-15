"""Deterministic, subject-balanced selection of a dataset-wide pipeline incumbent."""

from __future__ import annotations

import hashlib
import json
from math import isclose
from statistics import fmean, pstdev
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from bci_autodiscovery.pipelines import (
    DeterministicPipelineExecutor,
    PipelineSpec,
    pipeline_configuration_hash,
)

if TYPE_CHECKING:
    from bci_autodiscovery.workflow.budget import BudgetLedger


class DatasetIncumbentError(ValueError):
    """Raised when a dataset-wide incumbent artifact is unsafe or inconsistent."""


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _score_summary(
    subject_scores: Mapping[str, float], *, stability_penalty: float
) -> dict[str, float]:
    scores = [float(value) for value in subject_scores.values()]
    if not scores or any(not 0.0 <= value <= 1.0 for value in scores):
        raise DatasetIncumbentError("Subject scores must be non-empty and within [0, 1]")
    macro_mean = fmean(scores)
    subject_std = pstdev(scores)
    return {
        "macro_mean": macro_mean,
        "subject_std": subject_std,
        "worst_subject_score": min(scores),
        "robust_score": macro_mean - stability_penalty * subject_std,
    }


def build_dataset_incumbent(
    *,
    dataset_id: str,
    executors: Mapping[str, DeterministicPipelineExecutor],
    candidates: Sequence[PipelineSpec | dict[str, Any]],
    primary_metric: str = "balanced_accuracy",
    stability_penalty: float = 0.25,
    minimum_subjects: int = 3,
    minimum_candidates: int = 2,
    minimum_personalization_gain: float = 0.03,
    source_contracts: Mapping[str, Any] | None = None,
    budget_ledger: BudgetLedger | None = None,
) -> dict[str, Any]:
    """Evaluate fixed all-subject candidates and freeze the robust incumbent.

    Every subject is weighted equally, independent of its number of trials. Candidate
    generation remains an Agent responsibility; this function performs only numerical
    execution and the predeclared deterministic ranking rule.
    """

    if not dataset_id.strip():
        raise DatasetIncumbentError("dataset_id must be non-empty")
    if len(executors) < minimum_subjects:
        raise DatasetIncumbentError(
            f"At least {minimum_subjects} subjects are required for an incumbent"
        )
    if len(candidates) < minimum_candidates:
        raise DatasetIncumbentError(
            f"At least {minimum_candidates} candidates are required for an incumbent"
        )
    if stability_penalty < 0:
        raise DatasetIncumbentError("stability_penalty must be non-negative")
    if not 0 <= minimum_personalization_gain <= 1:
        raise DatasetIncumbentError("minimum_personalization_gain must be within [0, 1]")

    subject_ids = sorted(str(item) for item in executors)
    if len(subject_ids) != len(set(subject_ids)):
        raise DatasetIncumbentError("Executor subject identifiers must be unique")
    for subject_id, executor in executors.items():
        if str(subject_id) != str(executor.subject_id):
            raise DatasetIncumbentError("Executor mapping key differs from its subject_id")

    trace: list[dict[str, Any]] = []
    configuration_hashes: set[str] = set()
    for candidate in candidates:
        spec = candidate if isinstance(candidate, PipelineSpec) else PipelineSpec.from_dict(candidate)
        if spec.channel_strategy != "all" or spec.selected_channels:
            raise DatasetIncumbentError(
                "Dataset-wide incumbent candidates must use the common all-channel contract"
            )
        configuration_hash = pipeline_configuration_hash(spec)
        if configuration_hash in configuration_hashes:
            raise DatasetIncumbentError("Equivalent incumbent candidate was evaluated twice")
        configuration_hashes.add(configuration_hash)
        if budget_ledger is not None:
            budget_ledger.precheck(
                "dataset_incumbent_candidate",
                {
                    "research_cycles": 1,
                    "candidate_executions": len(subject_ids),
                },
            )
        results: dict[str, dict[str, Any]] = {}
        scores: dict[str, float] = {}
        for subject_id in subject_ids:
            result = executors[subject_id].evaluate(spec)
            if result.get("confirmation_data_accessed") is not False:
                raise DatasetIncumbentError("Incumbent evaluation accessed confirmation data")
            if primary_metric not in result.get("metrics", {}):
                raise DatasetIncumbentError(
                    f"Executor result lacks primary metric {primary_metric!r}"
                )
            score = float(result["metrics"][primary_metric])
            scores[subject_id] = score
            results[subject_id] = {
                "experiment_id": result["experiment_id"],
                "session_ids": result["session_ids"],
                "trial_count": result["trial_count"],
                "metrics": result["metrics"],
                "elapsed_seconds": result["elapsed_seconds"],
                "confirmation_data_accessed": False,
            }
        summary = _score_summary(scores, stability_penalty=stability_penalty)
        if budget_ledger is not None:
            budget_ledger.account(
                "dataset_incumbent_candidate",
                {
                    "research_cycles": 1,
                    "candidate_executions": len(subject_ids),
                    "compute_seconds": sum(
                        float(item["elapsed_seconds"]) for item in results.values()
                    ),
                },
                metadata={
                    "pipeline_configuration_sha256": configuration_hash,
                    "subject_ids": subject_ids,
                },
            )
        trace.append(
            {
                "aggregate_experiment_id": f"dataset-experiment-{configuration_hash[:16]}",
                "pipeline": spec.to_dict(),
                "pipeline_configuration_sha256": configuration_hash,
                "subject_results": results,
                "subject_scores": scores,
                "score_summary": summary,
                "subject_weighting": "equal_subject_macro",
                "confirmation_data_accessed": False,
            }
        )

    ranked = sorted(
        trace,
        key=lambda item: (
            -float(item["score_summary"]["robust_score"]),
            -float(item["score_summary"]["macro_mean"]),
            -float(item["score_summary"]["worst_subject_score"]),
            str(item["pipeline_configuration_sha256"]),
        ),
    )
    selected = ranked[0]
    policy = {
        "selection_rule": "maximize_macro_mean_minus_stability_penalty_times_subject_std",
        "subject_weighting": "equal_subject_macro",
        "stability_penalty": float(stability_penalty),
        "minimum_subjects": int(minimum_subjects),
        "minimum_candidates": int(minimum_candidates),
        "tie_breakers": ["macro_mean", "worst_subject_score", "configuration_sha256"],
    }
    personalization_policy = {
        "mode": "selective_personalization_with_dataset_fallback",
        "minimum_search_gain_over_incumbent": float(minimum_personalization_gain),
        "fallback_when_gain_is_insufficient": True,
        "confirmation_cannot_change_route": True,
    }
    artifact = {
        "schema_version": "1.0",
        "incumbent_id": f"dataset-incumbent-{selected['pipeline_configuration_sha256'][:16]}",
        "status": "frozen_dataset_pipeline_incumbent",
        "dataset_id": dataset_id,
        "primary_metric": primary_metric,
        "cohort_subject_ids": subject_ids,
        "selection_policy": policy,
        "personalization_policy": personalization_policy,
        "selected_aggregate_experiment_id": selected["aggregate_experiment_id"],
        "selected_pipeline": selected["pipeline"],
        "pipeline_configuration_sha256": selected["pipeline_configuration_sha256"],
        "selected_score_summary": selected["score_summary"],
        "search_trace": trace,
        "resource_usage": {
            "candidate_configurations": len(trace),
            "subject_candidate_executions": len(trace) * len(subject_ids),
            "compute_seconds": sum(
                float(result["elapsed_seconds"])
                for item in trace
                for result in item["subject_results"].values()
            ),
        },
        "source_contracts": dict(source_contracts or {}),
        "confirmation_data_accessed": False,
    }
    artifact["artifact_sha256_without_self"] = _canonical_hash(artifact)
    validate_dataset_incumbent(artifact)
    return artifact


def validate_dataset_incumbent(artifact: dict[str, Any]) -> None:
    if artifact.get("schema_version") != "1.0":
        raise DatasetIncumbentError("Unsupported incumbent schema_version")
    if artifact.get("status") != "frozen_dataset_pipeline_incumbent":
        raise DatasetIncumbentError("Incumbent artifact is not frozen")
    if artifact.get("confirmation_data_accessed") is not False:
        raise DatasetIncumbentError("Incumbent was selected with confirmation data")
    if not isinstance(artifact.get("dataset_id"), str) or not artifact["dataset_id"].strip():
        raise DatasetIncumbentError("Incumbent artifact lacks dataset_id")
    policy = artifact.get("selection_policy") or {}
    personalization = artifact.get("personalization_policy") or {}
    if policy.get("selection_rule") != (
        "maximize_macro_mean_minus_stability_penalty_times_subject_std"
    ):
        raise DatasetIncumbentError("Unknown incumbent selection rule")
    if policy.get("subject_weighting") != "equal_subject_macro":
        raise DatasetIncumbentError("Incumbent must use equal subject weighting")
    penalty = float(policy.get("stability_penalty", -1))
    if penalty < 0:
        raise DatasetIncumbentError("Invalid incumbent stability penalty")
    gain = float(personalization.get("minimum_search_gain_over_incumbent", -1))
    if (
        personalization.get("mode")
        != "selective_personalization_with_dataset_fallback"
        or personalization.get("fallback_when_gain_is_insufficient") is not True
        or personalization.get("confirmation_cannot_change_route") is not True
        or not 0 <= gain <= 1
    ):
        raise DatasetIncumbentError("Invalid selective personalization policy")
    trace = artifact.get("search_trace")
    cohort = [str(item) for item in artifact.get("cohort_subject_ids") or []]
    if not isinstance(trace, list) or len(trace) < int(policy.get("minimum_candidates", 0)):
        raise DatasetIncumbentError("Incumbent search trace is incomplete")
    if len(cohort) < int(policy.get("minimum_subjects", 0)) or len(cohort) != len(set(cohort)):
        raise DatasetIncumbentError("Incumbent cohort is incomplete or duplicated")
    usage = artifact.get("resource_usage") or {}
    if (
        int(usage.get("candidate_configurations", -1)) != len(trace)
        or int(usage.get("subject_candidate_executions", -1))
        != len(trace) * len(cohort)
    ):
        raise DatasetIncumbentError("Incumbent resource counts are inconsistent")
    seen: set[str] = set()
    for item in trace:
        spec = PipelineSpec.from_dict(item.get("pipeline") or {})
        if spec.channel_strategy != "all" or spec.selected_channels:
            raise DatasetIncumbentError("Incumbent trace contains a subject-specific pipeline")
        configuration_hash = pipeline_configuration_hash(spec)
        if configuration_hash != item.get("pipeline_configuration_sha256"):
            raise DatasetIncumbentError("Incumbent pipeline configuration hash changed")
        if configuration_hash in seen:
            raise DatasetIncumbentError("Incumbent trace contains duplicate configurations")
        seen.add(configuration_hash)
        scores = item.get("subject_scores") or {}
        if sorted(str(key) for key in scores) != sorted(cohort):
            raise DatasetIncumbentError("Incumbent candidate has a different subject cohort")
        if any(
            (result or {}).get("confirmation_data_accessed") is not False
            for result in (item.get("subject_results") or {}).values()
        ):
            raise DatasetIncumbentError("Incumbent trace contains confirmation evidence")
        expected = _score_summary(scores, stability_penalty=penalty)
        observed = item.get("score_summary") or {}
        if any(not isclose(expected[key], float(observed.get(key)), abs_tol=1e-12) for key in expected):
            raise DatasetIncumbentError("Incumbent score summary is inconsistent")
    ranked = sorted(
        trace,
        key=lambda item: (
            -float(item["score_summary"]["robust_score"]),
            -float(item["score_summary"]["macro_mean"]),
            -float(item["score_summary"]["worst_subject_score"]),
            str(item["pipeline_configuration_sha256"]),
        ),
    )
    selected = ranked[0]
    if (
        artifact.get("selected_aggregate_experiment_id")
        != selected.get("aggregate_experiment_id")
        or artifact.get("selected_pipeline") != selected.get("pipeline")
        or artifact.get("pipeline_configuration_sha256")
        != selected.get("pipeline_configuration_sha256")
        or artifact.get("selected_score_summary") != selected.get("score_summary")
    ):
        raise DatasetIncumbentError("Frozen incumbent is not the deterministic winner")
    expected_compute = sum(
        float(result["elapsed_seconds"])
        for item in trace
        for result in (item.get("subject_results") or {}).values()
    )
    if not isclose(
        expected_compute,
        float(usage.get("compute_seconds", -1)),
        abs_tol=1e-12,
    ):
        raise DatasetIncumbentError("Incumbent compute accounting is inconsistent")
    unsigned = dict(artifact)
    declared_hash = unsigned.pop("artifact_sha256_without_self", None)
    if declared_hash != _canonical_hash(unsigned):
        raise DatasetIncumbentError("Dataset incumbent artifact hash is inconsistent")


def expected_selective_route(
    *,
    incumbent_configuration_sha256: str,
    minimum_gain: float,
    experiments: Sequence[dict[str, Any]],
    primary_metric: str,
) -> dict[str, Any]:
    """Return the outcome-blind route required by the frozen fallback rule."""

    incumbent_matches = [
        item
        for item in experiments
        if item.get("configuration_sha256") == incumbent_configuration_sha256
    ]
    if len(incumbent_matches) != 1:
        raise DatasetIncumbentError("Exactly one incumbent evaluation is required per subject")
    incumbent = incumbent_matches[0]
    incumbent_score = float(incumbent["metrics"][primary_metric])
    alternatives = [item for item in experiments if item is not incumbent]
    if not alternatives:
        raise DatasetIncumbentError("Selective personalization requires an alternative candidate")
    best = max(
        alternatives,
        key=lambda item: (
            float(item["metrics"][primary_metric]),
            str(item.get("configuration_sha256", "")),
        ),
    )
    best_score = float(best["metrics"][primary_metric])
    improvement = best_score - incumbent_score
    gate_passed = improvement >= minimum_gain
    selected = best if gate_passed else incumbent
    return {
        "mode": "personalized" if gate_passed else "fallback_to_dataset_incumbent",
        "gate_passed": gate_passed,
        "minimum_required_gain": float(minimum_gain),
        "incumbent_experiment_id": incumbent["experiment_id"],
        "incumbent_search_score": incumbent_score,
        "best_personalized_experiment_id": best["experiment_id"],
        "best_personalized_search_score": best_score,
        "observed_personalization_gain": improvement,
        "required_selected_experiment_id": selected["experiment_id"],
        "confirmation_outcomes_observed": False,
    }
