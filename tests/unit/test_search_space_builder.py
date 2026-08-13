from __future__ import annotations

import json
from pathlib import Path

from bci_autodiscovery.agents.providers import SearchSpaceBuilderMockProvider
from bci_autodiscovery.agents.runtime import AgentRuntime
from bci_autodiscovery.agents.search_space_builder import (
    SearchSpaceBuilderAgent,
    create_search_space_builder_tools,
)
from bci_autodiscovery.search import build_search_space_draft


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    profile_path = tmp_path / "profile.json"
    profile = {
        "schema_version": "1.0",
        "dataset": {"id": "fixture_mi"},
        "paradigm": {
            "family": "motor_imagery",
            "actions": [{"label": "left_hand"}, {"label": "right_hand"}],
        },
        "resting_state": {"present": False},
        "signal": {
            "modalities": ["EEG"],
            "sampling_frequency_hz": 512.0,
            "channel_count": 22,
            "placement_scheme": "10-20 system",
            "eog_channel_count": 0,
        },
        "equipment": {},
        "events": {"common_analysis_window_s": [0.0, 4.0]},
        "sessions": {"sessions_per_subject": 6, "session_indices": [0, 1, 2, 3, 4, 5]},
        "volume": {"trials": 7200},
        "quality": {},
        "constraints": {
            "allowed": [],
            "forbidden": [],
            "requires_research_design_decision": [],
            "external_authority_blockers": [],
        },
        "evidence": [{"source": "fixture-profile"}],
    }
    _write_json(profile_path, profile)
    registry_path = Path("configs/component_registry.v0.json").resolve()
    return profile_path, registry_path


def test_builder_excludes_incompatible_methods_and_binds_dataset_ranges(tmp_path: Path) -> None:
    profile, registry = _inputs(tmp_path)
    draft = build_search_space_draft(
        dataset_profile_path=str(profile),
        component_registry_path=str(registry),
    )

    excluded = {item["component_id"] for item in draft["excluded_components"]}
    assert {"artifact_eog_regression", "feature_resting_iaf", "feature_xdawn"} <= excluded
    segmentation = draft["dimensions"]["segmentation"][0]
    windows = segmentation["parameters"]["window_s"]["values"]
    assert [0.0, 4.0] in windows
    assert all(0 <= start < stop <= 4 for start, stop in windows)
    sampling = draft["dimensions"]["sampling"][0]
    assert sampling["parameters"]["target_hz"]["values"] == [512.0, 256.0, 128.0, 100.0]
    assert draft["status"] == "dataset_coarse_space_awaiting_network_discovery"
    assert "protocol" not in draft
    assert "human_approval_required" not in draft
    assert all(value is False for value in draft["stage_boundary"].values())
    assert draft["scope_policy"]["frontier_network_discovery_required"] is True
    assert draft["canonical_space"]["component_count"] >= 20
    assert len(draft["frontier_discovery"]["query_plan"]) >= 8
    assert all(
        query["derived_from_profile"]["sessions_per_subject"] == 6
        for query in draft["frontier_discovery"]["query_plan"]
    )


def test_search_space_agent_runs_the_deterministic_builder_offline(tmp_path: Path) -> None:
    profile, registry = _inputs(tmp_path)
    runtime = AgentRuntime(
        provider=SearchSpaceBuilderMockProvider(),
        tools=create_search_space_builder_tools(),
        run_id="search-space-fixture",
    )
    result = SearchSpaceBuilderAgent(runtime).run(
        dataset_profile_path=profile,
        component_registry_path=registry,
    )

    assert result.status == "completed"
    draft = result.latest_tool_result("build_search_space_draft")
    assert draft["dataset_id"] == "fixture_mi"
    assert draft["status"] == "dataset_coarse_space_awaiting_network_discovery"
    assert draft["stage_boundary"]["execution_activation_performed"] is False


def test_single_session_profile_does_not_invent_cross_session_search_questions(
    tmp_path: Path,
) -> None:
    profile_path, registry = _inputs(tmp_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["sessions"] = {"sessions_per_subject": 1, "session_indices": [0]}
    _write_json(profile_path, profile)

    draft = build_search_space_draft(
        dataset_profile_path=str(profile_path),
        component_registry_path=str(registry),
    )

    query_ids = {
        item["query_id"] for item in draft["frontier_discovery"]["query_plan"]
    }
    assert "cross_session_nonstationarity" not in query_ids
    assert "test_time_adaptation" not in query_ids
    assert "within_session_nonstationarity" in query_ids
    assert "calibration_efficient_decoding" in query_ids


def test_unknown_timing_and_geometry_are_deferred_without_overclaiming(
    tmp_path: Path,
) -> None:
    profile_path, registry = _inputs(tmp_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["signal"].update(
        {
            "power_line_frequency_hz": None,
            "montage_verified": False,
        }
    )
    profile["events"].update(
        {
            "time_axis_reference": "stored_epoch_start; cue onset unverified",
            "cue_onset_s": None,
        }
    )
    _write_json(profile_path, profile)

    draft = build_search_space_draft(
        dataset_profile_path=str(profile_path),
        component_registry_path=str(registry),
    )

    excluded = {item["component_id"] for item in draft["excluded_components"]}
    assert "reference_surface_laplacian" in excluded
    components = {
        component["component_id"]: component
        for items in draft["dimensions"].values()
        for component in items
    }
    assert components["notch_powerline"]["dataset_condition"]["status"] == (
        "deferred_missing_dataset_metadata"
    )
    assert components["epoch_crop"]["dataset_condition"]["status"] == (
        "stored_epoch_relative_only"
    )
    assert {
        "model_kernel_svm",
        "model_tree_ensemble",
        "model_shallow_convnet",
    } <= set(components)
    rule_ids = {item["id"] for item in draft["compatibility_rules"]}
    assert "single_band_csp_requires_bandpass" in rule_ids
