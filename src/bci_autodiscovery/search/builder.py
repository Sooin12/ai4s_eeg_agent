"""Derive a broad but legal dataset-level pipeline search-space draft."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from bci_autodiscovery.profiling import validate_dataset_profile_provenance

from .components import ComponentRegistry


class SearchSpaceBuildError(RuntimeError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SearchSpaceBuildError(f"Cannot load JSON {path}: {exc}") from exc


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _requirement_failures(component: dict[str, Any], profile: dict[str, Any]) -> list[str]:
    requirements = component.get("requirements") or {}
    failures: list[str] = []
    signal = profile["signal"]
    paradigm = profile["paradigm"]
    volume = profile["volume"]
    sessions = profile["sessions"]
    modalities = set(signal.get("modalities") or [])
    actions = paradigm.get("actions") or []
    if requirements.get("modalities_any") and not modalities.intersection(
        requirements["modalities_any"]
    ):
        failures.append("required signal modality is unavailable")
    if requirements.get("paradigms") and paradigm.get("family") not in requirements["paradigms"]:
        failures.append("component does not support the detected paradigm")
    if int(signal.get("channel_count", 0)) < int(requirements.get("min_channels", 0)):
        failures.append("insufficient channel count")
    if int(volume.get("trials", 0)) < int(requirements.get("min_trials", 0)):
        failures.append("insufficient trial count")
    if requirements.get("binary_classes") and len(actions) != 2:
        failures.append("component requires exactly two action classes")
    if requirements.get("multiple_sessions") and int(sessions.get("sessions_per_subject", 0)) < 2:
        failures.append("component requires multiple sessions")
    montage_verified = signal.get("montage_verified")
    if montage_verified is None:
        montage_verified = bool(signal.get("placement_scheme"))
    if requirements.get("requires_montage") and not montage_verified:
        failures.append("component requires verified electrode geometry/montage")
    if requirements.get("requires_eog") and int(signal.get("eog_channel_count", 0)) <= 0:
        failures.append("component requires EOG channels")
    if requirements.get("requires_resting_state") and not profile["resting_state"].get("present"):
        failures.append("component requires independent resting-state data")
    return failures


def _valid_resample_rates(sfreq: float) -> list[float]:
    candidates = [sfreq, 256.0, 128.0, 100.0]
    return sorted({value for value in candidates if value <= sfreq and value >= 100.0}, reverse=True)


def _valid_windows(bounds: list[float]) -> list[list[float]]:
    start_bound, stop_bound = (float(bounds[0]), float(bounds[1]))
    starts = [start_bound, start_bound + 0.5, start_bound + 1.0]
    lengths = [1.0, 2.0, 3.0, 4.0]
    windows = {
        (round(start, 6), round(start + length, 6))
        for start in starts
        for length in lengths
        if start >= start_bound and start + length <= stop_bound
    }
    return [list(item) for item in sorted(windows)]


def _resolve_parameters(component: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(component.get("parameters") or {}))
    for parameter in result.values():
        if parameter.get("type") != "dataset_bound":
            continue
        binding = parameter.get("binding")
        if binding == "valid_resample_rates":
            parameter["values"] = _valid_resample_rates(
                float(profile["signal"]["sampling_frequency_hz"])
            )
        elif binding == "valid_epoch_windows":
            parameter["values"] = _valid_windows(
                profile["events"]["common_analysis_window_s"]
            )
        else:
            raise SearchSpaceBuildError(f"Unknown dataset parameter binding: {binding}")
    return result


def _literature_query_plan(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Build dataset-grounded discovery queries, not a closed method shortlist."""
    family = profile["paradigm"]["family"].replace("_", " ")
    modalities = " ".join(profile["signal"].get("modalities") or ["biosignal"])
    actions = [str(item.get("label")) for item in profile["paradigm"].get("actions") or []]
    sessions = int(profile["sessions"].get("sessions_per_subject", 0))
    has_eog = int(profile["signal"].get("eog_channel_count", 0)) > 0
    has_rest = bool(profile["resting_state"].get("present"))
    context = {
        "paradigm": profile["paradigm"]["family"],
        "sessions_per_subject": sessions,
        "eog_available": has_eog,
        "independent_rest_available": has_rest,
        "channel_count": int(profile["signal"].get("channel_count", 0)),
        "sampling_frequency_hz": float(profile["signal"]["sampling_frequency_hz"]),
        "actions": actions,
    }
    raw = [
        (
            "paradigm_method_landscape",
            f"{family} {modalities} decoding systematic review analysis pipeline methods",
            "Establish the method landscape for the detected paradigm before specializing the search.",
            ["preprocessing", "features", "models", "validation"],
        ),
        (
            "cross_session_nonstationarity",
            f"{family} {modalities} longitudinal multi-session nonstationarity session drift domain adaptation",
            "The dataset has repeated sessions, so session shift is both a failure mode and a research opportunity.",
            ["features", "models", "session_adaptation", "validation"],
        ),
        (
            "individual_motor_rhythms",
            f"{family} {modalities} subject-specific neural signature frequency spatial channel feature selection",
            "Per-subject temporal and spatial variability can constrain individualized preprocessing and features.",
            ["temporal_filtering", "referencing", "features", "subject_profiling"],
        ),
        (
            "artifact_without_auxiliary_eog",
            f"{family} {modalities} artifact removal {'without EOG' if not has_eog else 'with auxiliary channels'} robust trial quality",
            "Artifact handling must respect the auxiliary modalities that actually exist in the profile.",
            ["artifact_handling", "quality_control"],
        ),
        (
            "adaptive_bci_learning",
            f"{family} {modalities} repeated session learning adaptive decoder calibration transfer",
            "Repeated feedback sessions may reflect both neural learning and measurement drift.",
            ["session_adaptation", "models", "validation"],
        ),
        (
            "data_efficient_representation",
            f"{family} {modalities} self-supervised pretraining meta-learning transfer learning data efficient decoding",
            "Per-subject labeled data are limited even when the full dataset is large.",
            ["features", "models", "session_adaptation"],
        ),
        (
            "test_time_adaptation",
            f"{family} {modalities} test-time adaptation source-free domain adaptation cross-session",
            "Dataset sessions permit studying adaptation under explicit target-data access rules.",
            ["session_adaptation", "models", "validation"],
        ),
        (
            "riemannian_alignment",
            f"{family} {modalities} covariance geometry alignment recentering transfer learning session",
            "Multichannel repeated-session signals make covariance geometry a plausible robust representation family.",
            ["features", "models", "session_adaptation"],
        ),
        (
            "automated_pipeline_discovery",
            f"{family} {modalities} AutoML automated pipeline selection agent scientific discovery reproducible",
            "The research system must discover and compare whole pipelines, not only isolated classifiers.",
            ["pipeline_search", "validation", "evidence_explanation"],
        ),
        (
            "uncertainty_and_explainability",
            f"{family} {modalities} uncertainty calibration explainable decoding reliability subject-specific",
            "A scientific assistant must quantify uncertainty and failure modes, not only maximize a score.",
            ["models", "validation", "evidence_explanation"],
        ),
    ]
    repeated_session_ids = {
        "cross_session_nonstationarity",
        "adaptive_bci_learning",
        "test_time_adaptation",
    }
    if sessions < 2:
        raw = [item for item in raw if item[0] not in repeated_session_ids]
        raw.extend(
            [
                (
                    "within_session_nonstationarity",
                    f"{family} {modalities} within-session nonstationarity robust decoding calibration drift",
                    "Only one session per subject is available, so longitudinal claims are invalid but within-session drift may still matter.",
                    ["features", "models", "validation"],
                ),
                (
                    "calibration_efficient_decoding",
                    f"{family} {modalities} calibration-efficient few-shot subject-specific decoding",
                    "A single-session dataset motivates data-efficient calibration without pretending cross-session transfer can be measured.",
                    ["features", "models", "validation"],
                ),
            ]
        )
    return [
        {
            "query_id": query_id,
            "text": text,
            "rationale": rationale,
            "target_stages": stages,
            "derived_from_profile": context,
            "source_names": ["crossref", "openalex"],
            "limit_per_source": 20,
        }
        for query_id, text, rationale, stages in raw
    ]


def build_search_space_draft(
    *,
    dataset_profile_path: str,
    component_registry_path: str,
) -> dict[str, Any]:
    profile_path = Path(dataset_profile_path).expanduser().resolve()
    registry_path = Path(component_registry_path).expanduser().resolve()
    profile = _load_json(profile_path)
    validate_dataset_profile_provenance(profile)
    registry = ComponentRegistry.load(registry_path)
    signal = profile["signal"]

    applicable: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded: list[dict[str, Any]] = []
    deferred_component_decisions: list[dict[str, Any]] = []
    for component in registry.components:
        failures = _requirement_failures(component, profile)
        if failures:
            excluded.append(
                {
                    "component_id": component["id"],
                    "category": component["category"],
                    "reasons": failures,
                }
            )
            continue
        normalized = {
            "component_id": component["id"],
            "family": component["family"],
            "description": component["description"],
            "maturity": component["maturity"],
            "cost_tier": component["cost_tier"],
            "parameters": _resolve_parameters(component, profile),
            "dataset_requirements": component.get("requirements") or {},
            "execution_status": "not_activated_at_dataset_level",
        }
        conditional_fields: list[str] = []
        if component["id"] == "notch_powerline" and signal.get(
            "power_line_frequency_hz"
        ) is None:
            conditional_fields.append("signal.power_line_frequency_hz")
            normalized["dataset_condition"] = {
                "status": "deferred_missing_dataset_metadata",
                "requirement": (
                    "A downstream Research Design Agent must establish an evidence-backed "
                    "50/60 Hz policy before this component can be activated."
                ),
            }
        if component["id"] == "epoch_crop" and profile["events"].get(
            "cue_onset_s"
        ) is None:
            conditional_fields.extend(
                ["events.time_axis_reference", "events.cue_onset_s"]
            )
            normalized["dataset_condition"] = {
                "status": "stored_epoch_relative_only",
                "requirement": (
                    "Windows are offsets inside the validated stored epoch and must not be "
                    "described as cue-relative until cue timing is independently verified."
                ),
            }
        applicable[component["category"]].append(normalized)
        downstream_fields = component.get("downstream_decision_fields") or component.get(
            "human_approval_fields"
        )
        if downstream_fields:
            deferred_component_decisions.append(
                {
                    "component_id": component["id"],
                    "fields": downstream_fields,
                    "owner_agent": "research_design_or_method_engineering",
                }
            )
        if conditional_fields:
            deferred_component_decisions.append(
                {
                    "component_id": component["id"],
                    "fields": conditional_fields,
                    "owner_agent": "research_design",
                }
            )

    canonical_dimensions = {key: value for key, value in sorted(applicable.items())}
    return {
        "schema_version": "2.0",
        "contract_id": f"{profile['dataset']['id']}-dataset-coarse-space-v1",
        "status": "dataset_coarse_space_awaiting_network_discovery",
        "dataset_id": profile["dataset"]["id"],
        "scope_policy": {
            "name": "broad_dataset_level_coarse_search",
            "include_if": "compatible with dataset physics, observed modalities, paradigm, and acquisition structure",
            "exclude_only_if": "a hard data, physical, modality, paradigm, or acquisition incompatibility is demonstrated",
            "never_exclude_for": [
                "implementation maturity",
                "compute cost",
                "being conventional",
                "being newly discovered",
            ],
            "frontier_network_discovery_required": True,
            "execution_activation_owner": "downstream_method_and_research_design_agents",
        },
        "stage_boundary": {
            "session_roles_assigned": False,
            "evaluation_metrics_selected": False,
            "experiment_budget_allocated": False,
            "subject_data_accessed": False,
            "confirmation_data_accessed": False,
            "execution_activation_performed": False,
        },
        "canonical_space": {
            "status": "broad_non_executable_draft",
            "dimensions": canonical_dimensions,
            "component_count": sum(len(items) for items in canonical_dimensions.values()),
            "interpretation": (
                "A broad traditional and established-method universe. Maturity and cost "
                "are annotations for later scheduling, not dataset-level exclusion criteria."
            ),
        },
        "dimensions": canonical_dimensions,
        "excluded_components": excluded,
        "compatibility_rules": list(registry.compatibility_rules),
        "dataset_hard_constraints": profile["constraints"],
        "deferred_to_downstream_agents": {
            "component_decisions": deferred_component_decisions,
            "research_design_decisions": list(
                profile["constraints"].get("requires_research_design_decision")
                or profile["constraints"].get("requires_human_decision")
                or []
            )
            + [
                "evaluation metrics and aggregation",
                "per-subject experiment budget",
                "quality-control thresholds",
                "deep-model activation and compute allocation",
            ],
            "external_authority_blockers": list(
                profile["constraints"].get("external_authority_blockers") or []
            ),
        },
        "frontier_discovery": {
            "status": "required_not_yet_executed",
            "query_plan": _literature_query_plan(profile),
            "completion_requirement": (
                "Every planned query/source pair must be attempted and the evidence "
                "ledger must record results or failures; each query also needs at least "
                "one successful source."
            ),
            "rule": (
                "Network literature discovery may add method families absent from the local "
                "registry. Dataset-level directions remain non-executable hypotheses; "
                "method engineering and research-design agents own later implementation, "
                "license, test, budget, and protocol gates."
            ),
        },
        "provenance": {
            "dataset_profile": {"path": str(profile_path), "sha256": _hash(profile_path)},
            "component_registry": {
                "id": registry.registry_id,
                "status": registry.status,
                "path": str(registry_path),
                "sha256": _hash(registry_path),
            },
        },
    }
