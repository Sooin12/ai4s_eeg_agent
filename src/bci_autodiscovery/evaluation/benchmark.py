"""Synthetic individualized-pipeline benchmark for research-cycle efficiency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bci_autodiscovery.pipelines import DeterministicPipelineExecutor
from bci_autodiscovery.profiling.subject_measurements import (
    EpochSession,
    measure_class_separability,
    measure_spectral_profile,
)


class CycleBenchmarkError(ValueError):
    pass


def _pipeline(family: str, band: tuple[float, float], seed: int) -> dict[str, Any]:
    band_id = f"{int(band[0])}-{int(band[1])}"
    return {
        "pipeline_id": f"{family}-{band_id}",
        "family": family,
        "bandpass_hz": list(band),
        "spatial_filter": "csp" if family == "csp_lda" else "none",
        "csp_components": 4 if family == "csp_lda" else 0,
        "feature": "csp_log_variance" if family == "csp_lda" else "log_bandpower",
        "model": "shrinkage_lda",
        "lda_shrinkage": 0.1,
        "cv_folds": 3,
        "random_seed": seed,
    }


def executable_benchmark_candidates(seed: int) -> list[dict[str, Any]]:
    bands = ((4.0, 40.0), (8.0, 13.0), (8.0, 30.0), (13.0, 30.0))
    return [
        _pipeline(family, band, seed)
        for band in bands
        for family in ("bandpower_lda", "csp_lda")
    ]


def _synthetic_session(
    *,
    subject_id: str,
    session_id: str,
    seed: int,
    frequency_hz: float,
    pattern: str,
    trials_per_class: int,
    amplitude: float,
    confirmation_drift_hz: float = 0.0,
    fixture_generator_version: str = "v1_saturated",
    noise_scale: float = 1.0,
    nuisance_scale: float = 0.55,
) -> EpochSession:
    rng = np.random.default_rng(seed)
    sfreq = 128.0
    samples = 256
    labels = np.repeat(np.asarray([0, 1]), trials_per_class)
    time = np.arange(samples) / sfreq
    data = rng.normal(scale=noise_scale, size=(labels.size, 6, samples))
    frequency = frequency_hz + confirmation_drift_hz
    phase = rng.uniform(-0.2, 0.2, size=labels.size)
    rhythm = np.sin(2 * np.pi * frequency * time[None, :] + phase[:, None])
    if pattern == "spatial":
        data[labels == 0, 1, :] += amplitude * rhythm[labels == 0]
        data[labels == 1, 4, :] += amplitude * rhythm[labels == 1]
        if fixture_generator_version == "v1_saturated":
            nuisance = nuisance_scale * np.sin(2 * np.pi * 26.0 * time)
            data += nuisance[None, None, :]
    elif pattern == "global_power":
        scales = np.where(labels == 0, amplitude, amplitude * 0.32)
        data += scales[:, None, None] * rhythm[:, None, :]
        # Channel-specific noise prevents a spatial filter from exploiting exact symmetry.
        data += rng.normal(scale=0.15, size=(labels.size, 6, 1))
    else:
        raise CycleBenchmarkError(f"Unknown synthetic pattern: {pattern}")
    if fixture_generator_version == "v2_frequency_nuisance":
        nuisance_frequency = 20.0 if frequency_hz <= 13.0 else 10.0
        nuisance_phase = rng.uniform(0.0, 2 * np.pi, size=(labels.size, 6, 1))
        nuisance_amplitude = np.abs(
            rng.normal(scale=nuisance_scale, size=(labels.size, 6, 1))
        )
        nuisance_wave = np.sin(
            2 * np.pi * nuisance_frequency * time[None, None, :] + nuisance_phase
        )
        data += nuisance_amplitude * nuisance_wave
    elif fixture_generator_version != "v1_saturated":
        raise CycleBenchmarkError(
            f"Unknown fixture_generator_version: {fixture_generator_version}"
        )
    return EpochSession(
        subject_id=subject_id,
        session_id=session_id,
        data=data,
        labels=labels,
        sampling_frequency_hz=sfreq,
        channel_names=("F3", "C3", "Cz", "C4", "P4", "Oz"),
        provenance={
            "source": "synthetic_cycle_benchmark",
            "subject_id": subject_id,
            "session_id": session_id,
            "seed": seed,
            "frequency_hz": frequency,
            "pattern": pattern,
        },
    )


def _profile_guided_order(
    profiling: EpochSession,
    candidates: list[dict[str, Any]],
    *,
    guidance_policy_version: str = "v1_mean_spectrum",
) -> tuple[list[str], dict[str, Any]]:
    spectral = measure_spectral_profile(profiling)
    separability = measure_class_separability(profiling)
    sep = separability["payload"]
    spectral_payload = spectral["payload"]
    mu = spectral_payload["mu_peak"]
    beta = spectral_payload["beta_peak"]
    if guidance_policy_version == "v1_mean_spectrum":
        if float(mu["power"] or 0.0) >= float(beta["power"] or 0.0):
            target_band = (8.0, 13.0)
            profile_peak = float(mu["frequency_hz"])
        else:
            target_band = (13.0, 30.0)
            profile_peak = float(beta["frequency_hz"])
        maximum = float(sep["max_absolute_standardized_effect"] or 0.0)
        median = float(sep["median_absolute_standardized_effect"] or 0.0)
        guidance_evidence = "mean_spectral_peak_plus_broadband_class_effect"
    elif guidance_policy_version == "v2_bandwise_class_effect":
        bandwise = sep["bandwise"]
        selected_name = max(
            bandwise,
            key=lambda name: float(
                bandwise[name]["max_absolute_standardized_effect"] or 0.0
            ),
        )
        selected_band = bandwise[selected_name]
        target_band = tuple(float(value) for value in selected_band["band_hz"])
        profile_peak = float(
            mu["frequency_hz"] if target_band == (8.0, 13.0) else beta["frequency_hz"]
        )
        maximum = float(selected_band["max_absolute_standardized_effect"] or 0.0)
        median = float(selected_band["median_absolute_standardized_effect"] or 0.0)
        guidance_evidence = "bandwise_class_effect_then_spatial_distribution"
    else:
        raise CycleBenchmarkError(
            f"Unknown guidance_policy_version: {guidance_policy_version}"
        )
    distribution_ratio = median / maximum if maximum > 0 else 0.0
    preferred_family = "bandpower_lda" if distribution_ratio >= 0.55 else "csp_lda"
    alternative_family = "csp_lda" if preferred_family == "bandpower_lda" else "bandpower_lda"
    priorities = [
        (preferred_family, target_band),
        (alternative_family, target_band),
        (preferred_family, (8.0, 30.0)),
        (alternative_family, (8.0, 30.0)),
        (preferred_family, (4.0, 40.0)),
        (alternative_family, (4.0, 40.0)),
    ]
    other_band = (13.0, 30.0) if target_band == (8.0, 13.0) else (8.0, 13.0)
    priorities.extend(
        [(preferred_family, other_band), (alternative_family, other_band)]
    )
    lookup = {
        (item["family"], tuple(float(value) for value in item["bandpass_hz"])): item[
            "pipeline_id"
        ]
        for item in candidates
    }
    order = [lookup[item] for item in priorities]
    return order, {
        "spectral_measurement_id": spectral["measurement_id"],
        "separability_measurement_id": separability["measurement_id"],
        "profile_peak_hz": profile_peak,
        "target_band_hz": list(target_band),
        "class_effect_distribution_ratio": distribution_ratio,
        "preferred_family": preferred_family,
        "guidance_policy_version": guidance_policy_version,
        "guidance_evidence": guidance_evidence,
    }


def _fixed_order(candidates: list[dict[str, Any]]) -> list[str]:
    priorities = [
        ("bandpower_lda", (8.0, 30.0)),
        ("csp_lda", (8.0, 30.0)),
        ("bandpower_lda", (4.0, 40.0)),
        ("csp_lda", (4.0, 40.0)),
        ("bandpower_lda", (8.0, 13.0)),
        ("csp_lda", (8.0, 13.0)),
        ("bandpower_lda", (13.0, 30.0)),
        ("csp_lda", (13.0, 30.0)),
    ]
    lookup = {
        (item["family"], tuple(float(value) for value in item["bandpass_hz"])): item[
            "pipeline_id"
        ]
        for item in candidates
    }
    return [lookup[item] for item in priorities]


def _strategy_result(
    *,
    order: list[str],
    scores: dict[str, float],
    oracle_score: float,
    tolerance: float,
    minimum_cycles: int,
    maximum_cycles: int,
    stop_score: float,
) -> dict[str, Any]:
    best_id = order[0]
    best_score = scores[best_id]
    cycles_to_near_oracle: int | None = None
    locked_cycle = maximum_cycles
    for index, pipeline_id in enumerate(order, start=1):
        score = scores[pipeline_id]
        if score > best_score:
            best_id, best_score = pipeline_id, score
        if cycles_to_near_oracle is None and oracle_score - best_score <= tolerance:
            cycles_to_near_oracle = index
        if index >= minimum_cycles and (best_score >= stop_score or index >= maximum_cycles):
            locked_cycle = index
            break
    return {
        "locked_pipeline_id": best_id,
        "locked_search_score": best_score,
        "locked_cycle": locked_cycle,
        "oracle_gap_at_lock": oracle_score - best_score,
        "near_oracle_at_lock": oracle_score - best_score <= tolerance,
        "cycles_to_near_oracle": cycles_to_near_oracle,
        "evaluated_order": order[:locked_cycle],
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def run_synthetic_cycle_benchmark(spec: dict[str, Any]) -> dict[str, Any]:
    """Run an outcome-blind profile-guided search comparison on heterogeneous fixtures."""

    seed = int(spec["random_seed"])
    subjects_per_group = int(spec["subjects_per_group"])
    trials_per_class = int(spec["trials_per_class"])
    tolerance = float(spec["near_oracle_tolerance"])
    minimum_cycles = int(spec["guided_minimum_cycles"])
    maximum_cycles = int(spec["guided_maximum_cycles"])
    stop_score = float(spec["guided_stop_score"])
    random_repetitions = int(spec["random_search_repetitions"])
    fixture_version = str(spec.get("fixture_generator_version", "v1_saturated"))
    guidance_policy_version = str(
        spec.get("guidance_policy_version", "v1_mean_spectrum")
    )
    noise_scale = float(spec.get("noise_scale", 1.0))
    nuisance_scale = float(spec.get("nuisance_scale", 0.55))
    base_amplitude = float(spec.get("base_amplitude", 1.05))
    amplitude_jitter = float(spec.get("amplitude_jitter", 0.12))
    confirmation_amplitude_ratio = float(
        spec.get("confirmation_amplitude_ratio", 0.94)
    )
    if min(subjects_per_group, trials_per_class, minimum_cycles, maximum_cycles, random_repetitions) < 1:
        raise CycleBenchmarkError("Benchmark counts must be positive")
    if minimum_cycles > maximum_cycles:
        raise CycleBenchmarkError("guided_minimum_cycles exceeds guided_maximum_cycles")
    candidates = executable_benchmark_candidates(seed)
    candidate_by_id = {item["pipeline_id"]: item for item in candidates}
    groups = [
        ("mu_spatial", 10.0, "spatial"),
        ("beta_spatial", 20.0, "spatial"),
        ("mu_global_power", 10.0, "global_power"),
        ("beta_global_power", 20.0, "global_power"),
    ]
    subjects: list[dict[str, Any]] = []
    all_candidate_scores: dict[str, list[float]] = {
        item["pipeline_id"]: [] for item in candidates
    }
    rng = np.random.default_rng(seed)
    for group_index, (group, frequency, pattern) in enumerate(groups):
        for member in range(subjects_per_group):
            subject_index = group_index * subjects_per_group + member
            subject_id = f"syn-{subject_index + 1:03d}"
            amplitude = base_amplitude + amplitude_jitter * float(rng.random())
            base_seed = seed + subject_index * 20
            profiling = _synthetic_session(
                subject_id=subject_id,
                session_id="profiling",
                seed=base_seed + 1,
                frequency_hz=frequency,
                pattern=pattern,
                trials_per_class=trials_per_class,
                amplitude=amplitude,
                fixture_generator_version=fixture_version,
                noise_scale=noise_scale,
                nuisance_scale=nuisance_scale,
            )
            search = _synthetic_session(
                subject_id=subject_id,
                session_id="search",
                seed=base_seed + 2,
                frequency_hz=frequency,
                pattern=pattern,
                trials_per_class=trials_per_class,
                amplitude=amplitude,
                fixture_generator_version=fixture_version,
                noise_scale=noise_scale,
                nuisance_scale=nuisance_scale,
            )
            confirmation = _synthetic_session(
                subject_id=subject_id,
                session_id="confirmation",
                seed=base_seed + 3,
                frequency_hz=frequency,
                pattern=pattern,
                trials_per_class=trials_per_class,
                amplitude=amplitude * confirmation_amplitude_ratio,
                confirmation_drift_hz=(-0.35 if member % 2 else 0.35),
                fixture_generator_version=fixture_version,
                noise_scale=noise_scale,
                nuisance_scale=nuisance_scale,
            )
            executor = DeterministicPipelineExecutor(sessions=[search])
            experiments = {
                item["pipeline_id"]: executor.evaluate(item) for item in candidates
            }
            scores = {
                pipeline_id: float(value["metrics"]["balanced_accuracy"])
                for pipeline_id, value in experiments.items()
            }
            for pipeline_id, score in scores.items():
                all_candidate_scores[pipeline_id].append(score)
            oracle_id = max(scores, key=scores.get)
            oracle_score = scores[oracle_id]
            guided_order, profile_summary = _profile_guided_order(
                profiling,
                candidates,
                guidance_policy_version=guidance_policy_version,
            )
            guided = _strategy_result(
                order=guided_order,
                scores=scores,
                oracle_score=oracle_score,
                tolerance=tolerance,
                minimum_cycles=minimum_cycles,
                maximum_cycles=maximum_cycles,
                stop_score=stop_score,
            )
            fixed = _strategy_result(
                order=_fixed_order(candidates),
                scores=scores,
                oracle_score=oracle_score,
                tolerance=tolerance,
                minimum_cycles=minimum_cycles,
                maximum_cycles=maximum_cycles,
                stop_score=stop_score,
            )
            random_runs = []
            for repetition in range(random_repetitions):
                order = list(candidate_by_id)
                np.random.default_rng(base_seed + 1000 + repetition).shuffle(order)
                random_runs.append(
                    _strategy_result(
                        order=order,
                        scores=scores,
                        oracle_score=oracle_score,
                        tolerance=tolerance,
                        minimum_cycles=minimum_cycles,
                        maximum_cycles=maximum_cycles,
                        stop_score=stop_score,
                    )
                )
            fitted = executor.fit(candidate_by_id[guided["locked_pipeline_id"]])
            confirmed = executor.evaluate_fitted(
                fitted,
                sessions=[confirmation],
                data_role="synthetic_frozen_confirmation",
            )
            subjects.append(
                {
                    "subject_id": subject_id,
                    "fixture_group": group,
                    "latent_fixture_parameters": {
                        "frequency_hz": frequency,
                        "pattern": pattern,
                        "amplitude": amplitude,
                    },
                    "profile_guidance": profile_summary,
                    "finite_oracle": {
                        "pipeline_id": oracle_id,
                        "search_score": oracle_score,
                        "candidate_count": len(candidates),
                    },
                    "guided": {
                        **guided,
                        "confirmation_score": float(
                            confirmed["metrics"]["balanced_accuracy"]
                        ),
                        "search_to_confirmation_drop": guided["locked_search_score"]
                        - float(confirmed["metrics"]["balanced_accuracy"]),
                        "confirmation_refit": confirmed[
                            "fitting_performed_on_evaluation_data"
                        ],
                    },
                    "fixed_order": fixed,
                    "random_search": {
                        "mean_locked_cycle": _mean(
                            [float(item["locked_cycle"]) for item in random_runs]
                        ),
                        "mean_oracle_gap_at_lock": _mean(
                            [float(item["oracle_gap_at_lock"]) for item in random_runs]
                        ),
                        "near_oracle_lock_rate": _mean(
                            [float(item["near_oracle_at_lock"]) for item in random_runs]
                        ),
                        "mean_cycles_to_near_oracle": _mean(
                            [
                                float(item["cycles_to_near_oracle"] or len(candidates) + 1)
                                for item in random_runs
                            ]
                        ),
                    },
                    "search_scores": scores,
                }
            )

    global_best_id = max(
        all_candidate_scores,
        key=lambda item: _mean(all_candidate_scores[item]),
    )
    guided_cycles = [float(item["guided"]["locked_cycle"]) for item in subjects]
    guided_to_oracle = [
        float(item["guided"]["cycles_to_near_oracle"] or len(candidates) + 1)
        for item in subjects
    ]
    fixed_to_oracle = [
        float(item["fixed_order"]["cycles_to_near_oracle"] or len(candidates) + 1)
        for item in subjects
    ]
    locked_distribution: dict[str, int] = {}
    for item in subjects:
        pipeline_id = item["guided"]["locked_pipeline_id"]
        locked_distribution[pipeline_id] = locked_distribution.get(pipeline_id, 0) + 1
    summary = {
        "subject_count": len(subjects),
        "fixture_group_count": len(groups),
        "finite_oracle_candidate_count": len(candidates),
        "guided": {
            "mean_locked_cycles": _mean(guided_cycles),
            "mean_cycles_to_near_oracle": _mean(guided_to_oracle),
            "near_oracle_lock_rate": _mean(
                [float(item["guided"]["near_oracle_at_lock"]) for item in subjects]
            ),
            "mean_oracle_gap_at_lock": _mean(
                [float(item["guided"]["oracle_gap_at_lock"]) for item in subjects]
            ),
            "mean_confirmation_balanced_accuracy": _mean(
                [float(item["guided"]["confirmation_score"]) for item in subjects]
            ),
            "mean_search_to_confirmation_drop": _mean(
                [float(item["guided"]["search_to_confirmation_drop"]) for item in subjects]
            ),
        },
        "fixed_order": {
            "mean_cycles_to_near_oracle": _mean(fixed_to_oracle),
            "near_oracle_lock_rate": _mean(
                [float(item["fixed_order"]["near_oracle_at_lock"]) for item in subjects]
            ),
            "mean_oracle_gap_at_lock": _mean(
                [float(item["fixed_order"]["oracle_gap_at_lock"]) for item in subjects]
            ),
        },
        "random_search": {
            "mean_cycles_to_near_oracle": _mean(
                [float(item["random_search"]["mean_cycles_to_near_oracle"]) for item in subjects]
            ),
            "near_oracle_lock_rate": _mean(
                [float(item["random_search"]["near_oracle_lock_rate"]) for item in subjects]
            ),
            "mean_oracle_gap_at_lock": _mean(
                [float(item["random_search"]["mean_oracle_gap_at_lock"]) for item in subjects]
            ),
        },
        "global_best": {
            "pipeline_id": global_best_id,
            "mean_search_score": _mean(all_candidate_scores[global_best_id]),
            "mean_individual_oracle_gap": _mean(
                [
                    float(item["finite_oracle"]["search_score"])
                    - float(item["search_scores"][global_best_id])
                    for item in subjects
                ]
            ),
        },
        "individualized_locked_pipeline_distribution": locked_distribution,
        "unique_guided_locked_pipelines": len(locked_distribution),
    }
    return {
        "schema_version": "1.0",
        "benchmark_id": str(spec["benchmark_id"]),
        "status": "completed_engineering_benchmark",
        "scope": "synthetic_fixture_not_scientific_eeg_claim",
        "specification": spec,
        "candidate_universe": candidates,
        "summary": summary,
        "subjects": subjects,
    }


def run_synthetic_cycle_benchmark_file(
    *, specification_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    spec_path = Path(specification_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise CycleBenchmarkError("Benchmark specification must be a JSON object")
    result = run_synthetic_cycle_benchmark(spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
