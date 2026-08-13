from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from bci_autodiscovery.profiling import (
    DatasetProfileError,
    MatEpochSource,
    create_default_adapter_registry,
    validate_semantic_dataset,
)
from bci_autodiscovery.profiling.subject_measurements import SubjectMeasurementError


os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")


def _profile(*, dataset_id: str, runs: int = 1) -> dict:
    return {
        "dataset": {
            "id": dataset_id,
            "name": "Semantic adapter fixture",
            "version": "1",
            "dataset_type": "raw EEG",
            "license": "test-only",
            "format": "fixture",
            "doi": None,
        },
        "paradigm": {
            "family": "motor_imagery",
            "task_name": "left versus right imagery",
            "actions": [
                {"label": "left", "values": ["L"], "trial_count": 1},
                {"label": "right", "values": ["R"], "trial_count": 1},
            ],
            "cue_based": True,
            "feedback_present": False,
            "feedback_note": "fixture",
        },
        "resting_state": {
            "present": False,
            "matching_files": [],
            "interpretation": "No rest recording declared.",
        },
        "signal": {
            "modalities": ["EEG"],
            "sampling_frequency_hz": 100.0,
            "power_line_frequency_hz": 50.0,
            "recording_type": "continuous",
            "channel_count": 2,
            "channel_names": ["C3", "C4"],
            "channel_units": ["V"],
            "declared_channel_status_counts": {"good": 2},
            "reference": "unknown",
            "ground": "unknown",
            "placement_scheme": "10-20 labels",
            "montage_verified": False,
            "eog_channel_count": 0,
            "software_filters": "none",
            "hardware_filters": "unknown",
        },
        "equipment": {
            "manufacturer": None,
            "model": None,
            "cap_manufacturer": None,
            "cap_model": None,
        },
        "events": {
            "labels": ["left", "right"],
            "nominal_durations_s": [1.0],
            "total": 2,
            "class_counts": {"left": 1, "right": 1},
            "common_analysis_window_s": [0.0, 1.0],
            "common_window_reason": "Explicit fixture semantics.",
            "time_axis_reference": "annotation onset",
        },
        "sessions": {
            "subjects": 1,
            "sessions_per_subject": 1,
            "total_subject_sessions": 1,
            "session_indices": ["01"],
            "runs_per_session": {"01": runs},
            "longitudinal": False,
            "separate_days_declared": False,
        },
        "volume": {
            "subjects": 1,
            "subject_sessions": 1,
            "runs": runs,
            "trials": 2,
            "signal_bytes": 1,
        },
        "quality": {
            "core_valid_runs": runs,
            "core_invalid_runs": 0,
            "nonfinite_runs": 0,
            "flat_channel_runs": 0,
            "automatic_exclusions": 0,
        },
        "constraints": {
            "allowed": ["EEG classification within the declared one-second window"],
            "forbidden": ["silent exclusion", "confirmation data use during search"],
            "requires_research_design_decision": ["search/confirmation role policy"],
            "external_authority_blockers": [],
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_real_fif_semantic_adapter_validates_headers_events_and_profiles(
    tmp_path: Path,
) -> None:
    import mne

    root = tmp_path / "fif_dataset"
    root.mkdir()
    raw = mne.io.RawArray(
        np.zeros((2, 400)),
        mne.create_info(["C3", "C4"], 100.0, ch_types=["eeg", "eeg"]),
        verbose="ERROR",
    )
    raw.info["line_freq"] = 50.0
    raw.set_annotations(mne.Annotations([1.0, 2.5], [1.0, 1.0], ["L", "R"]))
    fif_path = root / "sub-01_ses-01_task-mi_raw.fif"
    raw.save(fif_path, overwrite=False, verbose="ERROR")
    semantics_path = root / "dataset_semantics.json"
    _write_json(
        semantics_path,
        {
            "schema_version": "1.0",
            "adapter_kind": "mne_raw",
            "source_globs": ["*.fif"],
            "mne_mapping": {
                "subject_regex": r"sub-(?P<subject>[^_/]+)",
                "session_regex": r"ses-(?P<session>[^_/]+)",
                "event_mapping": {"L": "left", "R": "right"},
            },
            "profile": _profile(dataset_id="fif_fixture"),
        },
    )
    validation_path = tmp_path / "fif_validation.json"
    validate_semantic_dataset(
        dataset_root=root,
        semantics_path=semantics_path,
        output_path=validation_path,
    )

    registry = create_default_adapter_registry()
    inspection = registry.inspect(dataset_root=root, validation_path=validation_path)
    profile = registry.profile(
        adapter_id=inspection["selected_adapter_id"],
        dataset_root=root,
        validation_path=validation_path,
        dataset_id_hint="auto",
    )

    assert inspection["status"] == "adapter_selected"
    assert inspection["selected_adapter_id"] == "mne_raw_semantic_v1"
    assert profile["events"]["class_counts"] == {"left": 1, "right": 1}
    assert profile["dataset"]["adapter_id"] == "mne_raw_semantic_v1"


def test_mne_validation_rejects_declared_event_count_disagreement(tmp_path: Path) -> None:
    import mne

    root = tmp_path / "bad_fif"
    root.mkdir()
    raw = mne.io.RawArray(
        np.zeros((2, 300)),
        mne.create_info(["C3", "C4"], 100.0, ch_types="eeg"),
        verbose="ERROR",
    )
    raw.info["line_freq"] = 50.0
    raw.set_annotations(mne.Annotations([1.0], [1.0], ["L"]))
    raw.save(root / "sub-01_ses-01_raw.fif", verbose="ERROR")
    semantics = {
        "schema_version": "1.0",
        "adapter_kind": "mne_raw",
        "source_globs": ["*.fif"],
        "mne_mapping": {
            "subject_regex": r"sub-(?P<subject>[^_/]+)",
            "session_regex": r"ses-(?P<session>[^_/]+)",
            "event_mapping": {"L": "left"},
        },
        "profile": _profile(dataset_id="bad_fixture"),
    }
    semantics_path = root / "dataset_semantics.json"
    _write_json(semantics_path, semantics)

    with pytest.raises(DatasetProfileError, match="class_counts|event_total"):
        validate_semantic_dataset(
            dataset_root=root,
            semantics_path=semantics_path,
            output_path=tmp_path / "should_not_exist.json",
        )


def test_declarative_npz_adapter_profiles_and_rejects_source_tampering(
    tmp_path: Path,
) -> None:
    root = tmp_path / "npz_dataset"
    root.mkdir()
    source = root / "sub-01_ses-01_epochs.npz"
    np.savez(source, data=np.zeros((2, 100, 2)), labels=np.array([1, 2]))
    semantics_path = root / "dataset_semantics.json"
    _write_json(
        semantics_path,
        {
            "schema_version": "1.0",
            "adapter_kind": "declarative",
            "source_globs": ["*.npz"],
            "container_mapping": {
                "signal_object_or_stream": "data",
                "axis_order": ["channel", "sample", "trial"],
                "event_source": "labels",
                "event_code_mapping": {"1": "left", "2": "right"},
                "data_role_policy": "roles are assigned only by frozen ResearchProtocol",
            },
            "profile": _profile(dataset_id="npz_fixture"),
        },
    )
    validation_path = tmp_path / "npz_validation.json"
    validate_semantic_dataset(
        dataset_root=root,
        semantics_path=semantics_path,
        output_path=validation_path,
    )
    registry = create_default_adapter_registry()
    inspection = registry.inspect(dataset_root=root, validation_path=validation_path)
    profile = registry.profile(
        adapter_id=inspection["selected_adapter_id"],
        dataset_root=root,
        validation_path=validation_path,
        dataset_id_hint="renamed_fixture",
    )

    assert inspection["selected_adapter_id"] == "declarative_semantic_v1"
    assert profile["dataset"]["id"] == "renamed_fixture"

    source.write_bytes(source.read_bytes() + b"tampered")
    rejected = registry.inspect(dataset_root=root, validation_path=validation_path)
    assert rejected["selected_adapter_id"] is None
    assert rejected["status"] == "recognized_format_requires_semantic_mapping"


def test_declarative_semantics_require_explicit_axes_events_and_role_policy(
    tmp_path: Path,
) -> None:
    root = tmp_path / "incomplete"
    root.mkdir()
    np.save(root / "signal.npy", np.zeros((2, 100)))
    semantics = {
        "schema_version": "1.0",
        "adapter_kind": "declarative",
        "source_globs": ["*.npy"],
        "container_mapping": {
            "signal_object_or_stream": "signal",
            "axis_order": ["channel", "sample"],
            "event_source": "missing",
            "event_code_mapping": {"1": "left"}
        },
        "profile": _profile(dataset_id="incomplete"),
    }
    semantics_path = root / "dataset_semantics.json"
    _write_json(semantics_path, semantics)

    with pytest.raises(DatasetProfileError, match="data_role_policy"):
        validate_semantic_dataset(
            dataset_root=root,
            semantics_path=semantics_path,
            output_path=tmp_path / "incomplete_validation.json",
        )


def test_registry_rejects_validation_with_empty_source_inventory(tmp_path: Path) -> None:
    root = tmp_path / "empty_inventory"
    root.mkdir()
    np.save(root / "signal.npy", np.zeros((2, 100)))
    semantics_path = root / "dataset_semantics.json"
    _write_json(
        semantics_path,
        {
            "schema_version": "1.0",
            "adapter_kind": "declarative",
            "source_globs": ["*.npy"],
            "container_mapping": {
                "signal_object_or_stream": "signal",
                "axis_order": ["channel", "sample"],
                "event_source": "external_annotations",
                "event_code_mapping": {"1": "left", "2": "right"},
                "data_role_policy": "assigned by frozen protocol",
            },
            "profile": _profile(dataset_id="empty_inventory"),
        },
    )
    validation_path = tmp_path / "empty_validation.json"
    validate_semantic_dataset(
        dataset_root=root,
        semantics_path=semantics_path,
        output_path=validation_path,
    )
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["sources"] = []
    _write_json(validation_path, validation)

    inspection = create_default_adapter_registry().inspect(
        dataset_root=root, validation_path=validation_path
    )
    assert inspection["selected_adapter_id"] is None
    assert inspection["status"] == "recognized_format_requires_semantic_mapping"


def test_generic_mat_epoch_adapter_derives_counts_quality_and_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mat_epochs"
    for subject in ("alpha", "beta"):
        for session in ("day-a", "day-b"):
            folder = root / f"participant-{subject}" / f"visit-{session}"
            folder.mkdir(parents=True, exist_ok=True)
            trial_count = 3 if (subject, session) == ("beta", "day-b") else 4
            labels = np.array([1, 2] * 2, dtype=np.int64)[:trial_count]
            savemat(
                folder / "epochs.mat",
                {
                    "epochs": np.ones((2, 20, trial_count), dtype=np.float32),
                    "targets": labels.reshape(1, -1),
                },
            )
    semantics_path = root / "dataset_semantics.json"
    _write_json(
        semantics_path,
        {
            "schema_version": "1.0",
            "adapter_kind": "mat_epochs",
            "source_globs": ["participant-*/visit-*/*.mat"],
            "mat_mapping": {
                "signal_key": "epochs",
                "label_key": "targets",
                "axis_order": ["channel", "sample", "trial"],
                "subject_regex": r"participant-(?P<subject>[^/]+)",
                "session_regex": r"visit-(?P<session>[^/]+)",
                "event_code_mapping": {"1": "left", "2": "right"},
                "sampling_frequency_hz": 100.0,
                "channel_names": ["C3", "C4"],
                "channel_units": "uV",
                "data_role_policy": "roles are assigned only by a frozen protocol",
            },
            "profile_hints": {
                "dataset": {
                    "id": "mat_fixture",
                    "name": "Generic MAT fixture",
                    "version": "1",
                    "dataset_type": "processed EEG arrays",
                    "license": "test-only",
                    "format": "MAT epoch arrays",
                    "doi": None,
                },
                "paradigm": {
                    "family": "motor_imagery",
                    "task_name": "generic binary imagery",
                    "cue_based": True,
                    "feedback_present": None,
                    "feedback_note": "not declared",
                },
                "resting_state": {
                    "present": False,
                    "matching_files": [],
                    "interpretation": "No rest recording declared.",
                },
                "signal": {
                    "modalities": ["EEG"],
                    "power_line_frequency_hz": None,
                    "declared_channel_status_counts": {},
                    "reference": "unknown",
                    "ground": "unknown",
                    "placement_scheme": "channel labels only",
                    "montage_verified": False,
                    "eog_channel_count": 0,
                    "software_filters": "unknown",
                    "hardware_filters": "unknown",
                },
                "equipment": {
                    "manufacturer": None,
                    "model": None,
                    "cap_manufacturer": None,
                    "cap_model": None,
                },
                "events": {
                    "time_axis_reference": "stored epoch start",
                    "cue_onset_s": None,
                    "pre_cue_baseline_present": None,
                },
                "sessions": {"separate_days_declared": True},
                "constraints": {
                    "allowed": ["analysis within the validated stored epoch"],
                    "forbidden": ["silent exclusion", "confirmation use during search"],
                    "requires_research_design_decision": ["session role assignment"],
                    "external_authority_blockers": [],
                },
            },
        },
    )
    validation_path = tmp_path / "mat_validation.json"
    validate_semantic_dataset(
        dataset_root=root,
        semantics_path=semantics_path,
        output_path=validation_path,
    )
    registry = create_default_adapter_registry()
    inspection = registry.inspect(dataset_root=root, validation_path=validation_path)
    profile = registry.profile(
        adapter_id=inspection["selected_adapter_id"],
        dataset_root=root,
        validation_path=validation_path,
        dataset_id_hint="renamed_mat_fixture",
    )

    assert inspection["selected_adapter_id"] == "mat_epoch_semantic_v1"
    assert profile["dataset"]["id"] == "renamed_mat_fixture"
    assert profile["volume"]["subjects"] == 2
    assert profile["volume"]["runs"] == 4
    assert profile["volume"]["trials"] == 15
    assert profile["events"]["class_counts"] == {"left": 8, "right": 7}
    assert profile["quality"]["missing_trials"] == 1
    assert profile["quality"]["trial_count_anomalies"][0]["session_id"] == "day-b"


def test_generic_mat_epoch_adapter_reports_missing_subject_session(
    tmp_path: Path,
) -> None:
    root = tmp_path / "incomplete_grid"
    for subject, session in (("one", "a"), ("one", "b"), ("two", "a")):
        folder = root / f"person-{subject}" / f"visit-{session}"
        folder.mkdir(parents=True, exist_ok=True)
        savemat(
            folder / "epochs.mat",
            {
                "x": np.ones((2, 10, 2), dtype=np.float32),
                "y": np.array([[1, 2]]),
            },
        )
    semantics = {
        "schema_version": "1.0",
        "adapter_kind": "mat_epochs",
        "source_globs": ["person-*/visit-*/*.mat"],
        "mat_mapping": {
            "signal_key": "x",
            "label_key": "y",
            "axis_order": ["channel", "sample", "trial"],
            "subject_regex": r"person-(?P<subject>[^/]+)",
            "session_regex": r"visit-(?P<session>[^/]+)",
            "event_code_mapping": {"1": "a", "2": "b"},
            "sampling_frequency_hz": 100.0,
            "channel_names": ["C3", "C4"],
            "channel_units": "uV",
            "data_role_policy": "assigned later",
        },
        "profile_hints": {
            "dataset": {"id": "grid", "name": "grid", "version": "1", "dataset_type": "epochs", "license": "test", "format": "MAT", "doi": None},
            "paradigm": {"family": "fixture", "task_name": "fixture", "cue_based": True, "feedback_present": None, "feedback_note": "unknown"},
            "resting_state": {"present": False, "matching_files": [], "interpretation": "none"},
            "signal": {"modalities": ["EEG"], "power_line_frequency_hz": None, "declared_channel_status_counts": {}, "reference": "unknown", "ground": "unknown", "placement_scheme": "labels", "montage_verified": False, "eog_channel_count": 0, "software_filters": "unknown", "hardware_filters": "unknown"},
            "equipment": {"manufacturer": None, "model": None, "cap_manufacturer": None, "cap_model": None},
            "events": {"time_axis_reference": "epoch start"},
            "sessions": {"separate_days_declared": False},
            "constraints": {"allowed": [], "forbidden": [], "requires_research_design_decision": [], "external_authority_blockers": []},
        },
    }
    semantics_path = root / "dataset_semantics.json"
    _write_json(semantics_path, semantics)
    validation_path = tmp_path / "grid_validation.json"
    validate_semantic_dataset(
        dataset_root=root,
        semantics_path=semantics_path,
        output_path=validation_path,
    )
    registry = create_default_adapter_registry()
    profile = registry.profile(
        adapter_id="mat_epoch_semantic_v1",
        dataset_root=root,
        validation_path=validation_path,
    )
    assert profile["quality"]["missing_subject_sessions"] == [
        {"subject_id": "two", "session_id": "b"}
    ]


def test_generic_mat_epoch_adapter_reports_identity_conflicts_and_blocks_subject_loading(
    tmp_path: Path,
) -> None:
    root = tmp_path / "identity_conflict"
    folder = root / "person-one" / "visit-a"
    folder.mkdir(parents=True)
    savemat(
        folder / "person-two_visit-a_epochs.mat",
        {"x": np.ones((2, 10, 2), dtype=np.float32), "y": np.array([[1, 2]])},
    )
    semantics = {
        "schema_version": "1.0",
        "adapter_kind": "mat_epochs",
        "source_globs": ["person-*/*/*.mat"],
        "mat_mapping": {
            "signal_key": "x", "label_key": "y",
            "axis_order": ["channel", "sample", "trial"],
            "subject_regex": r"(?:^|/)person-(?P<subject>[^/]+)(?:/|$)",
            "session_regex": r"(?:^|/)visit-(?P<session>[^/]+)(?:/|$)",
            "subject_assertion_regex": r"(?:^|/)person-(?P<subject>[^/_]+)_visit-",
            "session_assertion_regex": r"_visit-(?P<session>[^_]+)_",
            "event_code_mapping": {"1": "a", "2": "b"},
            "sampling_frequency_hz": 100.0,
            "channel_names": ["C3", "C4"], "channel_units": "uV",
            "data_role_policy": "assigned later",
        },
        "profile_hints": {
            "dataset": {"id": "identity", "name": "identity", "version": "1", "dataset_type": "epochs", "license": "test", "format": "MAT", "doi": None},
            "paradigm": {"family": "fixture", "task_name": "fixture", "cue_based": True, "feedback_present": None, "feedback_note": "unknown"},
            "resting_state": {"present": False, "matching_files": [], "interpretation": "none"},
            "signal": {"modalities": ["EEG"], "power_line_frequency_hz": None, "declared_channel_status_counts": {}, "reference": "unknown", "ground": "unknown", "placement_scheme": "labels", "montage_verified": False, "eog_channel_count": 0, "software_filters": "unknown", "hardware_filters": "unknown"},
            "equipment": {"manufacturer": None, "model": None, "cap_manufacturer": None, "cap_model": None},
            "events": {"time_axis_reference": "epoch start"},
            "sessions": {"separate_days_declared": False},
            "constraints": {"allowed": [], "forbidden": [], "requires_research_design_decision": [], "external_authority_blockers": []},
        },
    }
    semantics_path = root / "dataset_semantics.json"
    _write_json(semantics_path, semantics)
    validation_path = tmp_path / "identity_validation.json"
    validate_semantic_dataset(
        dataset_root=root, semantics_path=semantics_path, output_path=validation_path
    )
    profile = create_default_adapter_registry().profile(
        adapter_id="mat_epoch_semantic_v1",
        dataset_root=root,
        validation_path=validation_path,
    )

    assert profile["quality"]["identity_conflict_count"] == 1
    assert profile["quality"]["core_invalid_runs"] == 1
    assert profile["quality"]["identity_conflicts"][0]["conflicts"] == [
        {"field": "subject", "authoritative_value": "one", "asserted_value": "two"}
    ]
    assert any(
        "identity declarations" in item
        for item in profile["constraints"]["external_authority_blockers"]
    )
    with pytest.raises(SubjectMeasurementError, match="conflicting identity"):
        MatEpochSource(dataset_root=root, validation_path=validation_path)


def test_mat_epoch_subject_source_uses_validated_identity_and_standard_axes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "subject_source"
    folder = root / "person-x" / "visit-y"
    folder.mkdir(parents=True)
    data = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
    savemat(folder / "epochs.mat", {"x": data, "y": np.array([[1, 2, 1]])})
    semantics = {
        "schema_version": "1.0",
        "adapter_kind": "mat_epochs",
        "source_globs": ["person-*/*/*.mat"],
        "mat_mapping": {
            "signal_key": "x", "label_key": "y",
            "axis_order": ["channel", "sample", "trial"],
            "subject_regex": r"person-(?P<subject>[^/]+)",
            "session_regex": r"visit-(?P<session>[^/]+)",
            "event_code_mapping": {"1": "left", "2": "right"},
            "sampling_frequency_hz": 100.0,
            "channel_names": ["C3", "C4"], "channel_units": "uV",
            "data_role_policy": "assigned later",
        },
        "profile_hints": {
            "dataset": {"id": "source", "name": "source", "version": "1", "dataset_type": "epochs", "license": "test", "format": "MAT", "doi": None},
            "paradigm": {"family": "fixture", "task_name": "fixture", "cue_based": True, "feedback_present": None, "feedback_note": "unknown"},
            "resting_state": {"present": False, "matching_files": [], "interpretation": "none"},
            "signal": {"modalities": ["EEG"], "power_line_frequency_hz": None, "declared_channel_status_counts": {}, "reference": "unknown", "ground": "unknown", "placement_scheme": "labels", "montage_verified": False, "eog_channel_count": 0, "software_filters": "unknown", "hardware_filters": "unknown"},
            "equipment": {"manufacturer": None, "model": None, "cap_manufacturer": None, "cap_model": None},
            "events": {"time_axis_reference": "epoch start"},
            "sessions": {"separate_days_declared": False},
            "constraints": {"allowed": [], "forbidden": [], "requires_research_design_decision": [], "external_authority_blockers": []},
        },
    }
    semantics_path = root / "dataset_semantics.json"
    _write_json(semantics_path, semantics)
    validation_path = tmp_path / "subject_validation.json"
    validate_semantic_dataset(
        dataset_root=root, semantics_path=semantics_path, output_path=validation_path
    )
    session = MatEpochSource(
        dataset_root=root, validation_path=validation_path
    ).load_session(subject_id="x", session_id="y")
    assert session.data.shape == (3, 2, 5)
    assert session.labels.tolist() == [1, 2, 1]
    assert session.channel_names == ("C3", "C4")
