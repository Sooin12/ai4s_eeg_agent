from __future__ import annotations

import json
from pathlib import Path

from bci_autodiscovery.agents.dataset_profiler import (
    DatasetProfilerAgent,
    create_dataset_profiler_tools,
)
from bci_autodiscovery.agents.contracts import ModelResponse, ToolCall
from bci_autodiscovery.agents.providers import (
    DatasetProfilerMockProvider,
    ScriptedProvider,
)
from bci_autodiscovery.agents.runtime import AgentRuntime


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _dataset_fixture(
    tmp_path: Path, *, signal_extension: str = ".edf"
) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    eeg = root / "sub-1" / "ses-0" / "eeg"
    eeg.mkdir(parents=True)
    prefix = eeg / "sub-1_ses-0_task-imagery_run-0"
    _write_json(
        root / "dataset_description.json",
        {
            "Name": "Fixture MI",
            "Version": "1",
            "BIDSVersion": "1.9.0",
            "DatasetType": "derivative",
            "License": "CC-BY-4.0",
            "DatasetDOI": "fixture",
        },
    )
    _write_json(
        prefix.with_name(prefix.name + "_eeg.json"),
        {
            "TaskName": "imagery",
            "Manufacturer": "Fixture",
            "ManufacturersModelName": "Device",
            "CapManufacturer": "Fixture",
            "CapManufacturersModelName": "Cap",
            "SamplingFrequency": 512,
            "PowerLineFrequency": 50,
            "RecordingType": "continuous",
            "EEGReference": "CPz",
            "EEGGround": "AFz",
            "EEGPlacementScheme": "10-20",
            "EEGChannelCount": 2,
            "EOGChannelCount": 0,
            "SoftwareFilters": "n/a",
            "HardwareFilters": "n/a",
            "TaskDescription": "Cue-based left/right hand motor imagery with feedback on separate days.",
            "Instructions": "Imagine moving the instructed hand.",
        },
    )
    prefix.with_name(prefix.name + f"_eeg{signal_extension}").write_bytes(b"fixture")
    prefix.with_name(prefix.name + "_channels.tsv").write_text(
        "name\ttype\tunits\tstatus\nC3\tEEG\tuV\tgood\nC4\tEEG\tuV\tgood\n",
        encoding="utf-8",
    )
    prefix.with_name(prefix.name + "_events.tsv").write_text(
        "onset\tduration\ttrial_type\tvalue\n"
        "0\t5\tleft_hand\t1\n6\t5\tright_hand\t2\n",
        encoding="utf-8",
    )
    validation_path = tmp_path / "validation.json"
    _write_json(
        validation_path,
        {
            "dataset_root": str(root.resolve()),
            "subject_count": 1,
            "session_count": 1,
            "validation": {
                "valid_runs": 1,
                "invalid_runs": 0,
                "nonfinite_runs": 0,
                "flat_channel_runs": 0,
                "truncated_event_runs": 0,
                "isolated_diagnostic_transient_runs": 0,
            },
            "quality_review": {"candidate_count": 0},
            "diagnostic_filtered_review": {"candidate_count": 0},
        },
    )
    return root, validation_path


def test_dataset_profiler_agent_runs_offline_and_returns_evidence(tmp_path: Path) -> None:
    root, validation_path = _dataset_fixture(tmp_path)
    runtime = AgentRuntime(
        provider=DatasetProfilerMockProvider(),
        tools=create_dataset_profiler_tools(),
        run_id="dataset-fixture",
    )
    result = DatasetProfilerAgent(runtime).run(
        dataset_id="fixture_mi",
        dataset_root=root,
        validation_path=validation_path,
    )

    assert result.status == "completed"
    inspection = result.latest_tool_result("inspect_dataset")
    assert inspection["status"] == "adapter_selected"
    assert inspection["selected_adapter_id"] == "bids_eeg_v1"
    profile = result.latest_tool_result("profile_dataset")
    assert profile["paradigm"]["family"] == "motor_imagery"
    assert profile["events"]["class_counts"] == {"left_hand": 1, "right_hand": 1}
    assert profile["resting_state"]["present"] is False
    assert profile["signal"]["channel_names"] == ["C3", "C4"]
    assert profile["dataset"]["adapter_id"] == "bids_eeg_v1"
    assert profile["evidence"]


def test_generic_bids_adapter_accepts_non_edf_official_container(tmp_path: Path) -> None:
    root, validation_path = _dataset_fixture(tmp_path, signal_extension=".bdf")
    tools = create_dataset_profiler_tools()

    inspection = tools.execute(
        "inspect_dataset",
        {
            "dataset_root": str(root),
            "validation_path": str(validation_path),
        },
    )
    profile = tools.execute(
        "profile_dataset",
        {
            "dataset_id": "bdf_fixture",
            "adapter_id": inspection["selected_adapter_id"],
            "dataset_root": str(root),
            "validation_path": str(validation_path),
        },
    )

    assert inspection["selected_adapter_id"] == "bids_eeg_v1"
    assert profile["dataset"]["format"] == "BIDS EEG with bdf"


def test_generic_bids_adapter_does_not_force_motor_imagery_semantics(
    tmp_path: Path,
) -> None:
    root, validation_path = _dataset_fixture(tmp_path)
    eeg_json = next(root.rglob("*_eeg.json"))
    metadata = json.loads(eeg_json.read_text(encoding="utf-8"))
    metadata["TaskName"] = "oddball"
    metadata["TaskDescription"] = "Auditory target detection task."
    metadata["Instructions"] = "Press a button for target tones."
    _write_json(eeg_json, metadata)
    events = next(root.rglob("*_events.tsv"))
    events.write_text(
        "onset\tduration\ttrial_type\tvalue\n0\t1\tstandard\t1\n2\t1\ttarget\t2\n",
        encoding="utf-8",
    )

    tools = create_dataset_profiler_tools()
    inspection = tools.execute(
        "inspect_dataset",
        {"dataset_root": str(root), "validation_path": str(validation_path)},
    )
    profile = tools.execute(
        "profile_dataset",
        {
            "dataset_id": "oddball_fixture",
            "adapter_id": inspection["selected_adapter_id"],
            "dataset_root": str(root),
            "validation_path": str(validation_path),
        },
    )

    assert profile["paradigm"]["family"] == "bids_task_oddball"
    assert profile["events"]["labels"] == ["standard", "target"]


def test_dataset_profiler_rejects_premature_final_and_repairs_itself(
    tmp_path: Path,
) -> None:
    root, validation_path = _dataset_fixture(tmp_path)
    tools = create_dataset_profiler_tools()
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="inspect",
                        name="inspect_dataset",
                        arguments={
                            "dataset_root": str(root.resolve()),
                            "validation_path": str(validation_path.resolve()),
                        },
                    ),
                )
            ),
            ModelResponse(content="Premature completion."),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="profile",
                        name="profile_dataset",
                        arguments={
                            "dataset_id": "fixture_mi",
                            "adapter_id": "bids_eeg_v1",
                            "dataset_root": str(root.resolve()),
                            "validation_path": str(validation_path.resolve()),
                        },
                    ),
                )
            ),
            ModelResponse(content="Dataset profiling completed after repair."),
        ]
    )
    result = DatasetProfilerAgent(
        AgentRuntime(provider=provider, tools=tools, run_id="profiler-self-repair")
    ).run(
        dataset_id="fixture_mi",
        dataset_root=root,
        validation_path=validation_path,
    )

    assert result.status == "completed"
    assert result.latest_tool_result("profile_dataset")["dataset"]["id"] == (
        "fixture_mi"
    )
    assert tools.execute("inspect_dataset_profiler_status", {})["complete"] is True
