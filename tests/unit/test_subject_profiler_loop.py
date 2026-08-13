from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bci_autodiscovery.agents.contracts import ModelResponse, ToolCall
from bci_autodiscovery.agents.providers import ScriptedProvider
from bci_autodiscovery.agents.runtime import AgentRuntime
from bci_autodiscovery.agents.subject_profiler import (
    SubjectProfilerAgent,
    create_subject_profiler_tools,
)
from bci_autodiscovery.agents.tools import ToolExecutionError
from bci_autodiscovery.profiling.subject_measurements import (
    EpochSession,
    SubjectMeasurementEngine,
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class _SyntheticSource:
    source_id = "synthetic-subject-source"

    def __init__(self) -> None:
        self.sessions: dict[str, EpochSession] = {}
        for session_index, session_id in enumerate(("1", "2"), start=1):
            rng = np.random.default_rng(100 + session_index)
            labels = np.repeat(np.asarray([1, 2]), 20)
            samples = 256
            sfreq = 128.0
            time = np.arange(samples) / sfreq
            data = rng.normal(scale=0.7, size=(40, 4, samples))
            motor = np.sin(2 * np.pi * 10.0 * time)
            data[labels == 2, 1, :] += (1.0 + 0.1 * session_index) * motor
            data[labels == 1, 2, :] += 0.8 * motor
            self.sessions[session_id] = EpochSession(
                subject_id="sub-001",
                session_id=session_id,
                data=data,
                labels=labels,
                sampling_frequency_hz=sfreq,
                channel_names=("C3", "Cz", "C4", "Pz"),
                provenance={
                    "source_id": self.source_id,
                    "session_id": session_id,
                    "sha256": f"synthetic-{session_id}",
                },
            )

    def load_session(self, *, subject_id: str, session_id: str) -> EpochSession:
        assert subject_id == "sub-001"
        return self.sessions[session_id]


def _dataset_profile() -> dict:
    return {
        "schema_version": "1.0",
        "dataset": {"id": "subject-fixture"},
        "paradigm": {
            "family": "motor_imagery",
            "actions": [{"label": "left"}, {"label": "right"}],
        },
        "resting_state": {"present": False},
        "signal": {
            "channel_count": 4,
            "sampling_frequency_hz": 128,
            "modalities": ["EEG"],
        },
        "equipment": {},
        "events": {"common_analysis_window_s": [0, 2]},
        "sessions": {"session_indices": [1, 2, 3], "sessions_per_subject": 3},
        "volume": {"trials": 120},
        "quality": {},
        "constraints": {"allowed": [], "forbidden": [], "requires_human_decision": []},
        "evidence": [{"source": "synthetic"}],
    }


def _frozen_protocol() -> dict:
    return {
        "schema_version": "2.0",
        "protocol_id": "subject-fixture-protocol",
        "dataset_id": "subject-fixture",
        "status": "frozen_autonomous",
        "split_unit": "session",
        "data_roles": {
            "profiling_and_calibration": ["1", "2"],
            "pipeline_search_and_lock": ["3-search"],
            "frozen_confirmation": ["3-confirmation"],
        },
    }


def test_deterministic_measurements_are_raw_free_and_evidence_linked() -> None:
    engine = SubjectMeasurementEngine(
        source=_SyntheticSource(),
        subject_id="sub-001",
        allowed_session_ids=["1", "2"],
    )
    quality = engine.quality("1")
    spectral = engine.spectral("1")
    separability = engine.separability("1")
    stability = engine.stability(["1", "2"])
    encoded = json.dumps([quality, spectral, separability, stability])
    assert "measurement_id" in encoded
    assert "raw" not in encoded.lower()
    assert quality["payload"]["finite_fraction"] == 1.0
    assert spectral["payload"]["mu_peak"]["frequency_hz"] is not None
    assert separability["payload"]["max_absolute_standardized_effect"] > 0
    assert stability["payload"]["cross_session_status"] == "measured"


def test_subject_profiler_agent_completes_measure_interpret_loop(tmp_path: Path) -> None:
    profile_path = tmp_path / "dataset_profile.json"
    protocol_path = tmp_path / "frozen_protocol.json"
    _write(profile_path, _dataset_profile())
    _write(protocol_path, _frozen_protocol())
    expected_engine = SubjectMeasurementEngine(
        source=_SyntheticSource(),
        subject_id="sub-001",
        allowed_session_ids=["1", "2"],
    )
    expected = [
        expected_engine.quality("1"),
        expected_engine.spectral("1"),
        expected_engine.separability("1"),
        expected_engine.stability(["1", "2"]),
    ]
    measurement_ids = [item["measurement_id"] for item in expected]
    spectral_id = expected[1]["measurement_id"]
    separability_id = expected[2]["measurement_id"]
    subject_profile = {
        "schema_version": "1.0",
        "subject_id": "sub-001",
        "status": "complete",
        "measurement_ids": measurement_ids,
        "quality_summary": ["No deterministic flat-channel flag was observed."],
        "spectral_summary": ["A candidate mu peak was measured on profiling data."],
        "erd_ers_summary": {
            "status": "unavailable_without_pre_event_baseline_contract",
            "statement": "Do not claim ERD/ERS from epochs without a baseline.",
            "evidence_measurement_ids": [spectral_id],
        },
        "stability_summary": ["Cross-session class-effect stability was measured."],
        "class_separability_summary": [
            "Training-side 8-30 Hz effects motivate, but do not prove, spatial features."
        ],
        "hypotheses": [
            {
                "hypothesis_id": "subject-mu-spatial-v1",
                "statement": "Subject-centered spectral and spatial branches merit testing.",
                "evidence_measurement_ids": [spectral_id, separability_id],
                "confidence": 0.75,
            }
        ],
        "uncertainties": ["No independent baseline supports an ERD/ERS estimate."],
        "search_implications": [
            {
                "target_stage": "features",
                "recommendation": "Prioritize but do not force CSP and subject-centered bands.",
                "evidence_measurement_ids": [spectral_id, separability_id],
                "status": "proposed_for_budgeted_search",
            }
        ],
    }
    engine = SubjectMeasurementEngine(
        source=_SyntheticSource(),
        subject_id="sub-001",
        allowed_session_ids=["1", "2"],
    )
    tools, context = create_subject_profiler_tools(
        engine=engine,
        dataset_profile_path=profile_path,
        frozen_protocol_path=protocol_path,
    )
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(ToolCall("context", "read_subject_profile_context", {}),)
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "quality",
                        "measure_subject_signal_quality",
                        {"session_id": "1"},
                    ),
                    ToolCall(
                        "spectral",
                        "measure_subject_spectral_profile",
                        {"session_id": "1"},
                    ),
                    ToolCall(
                        "separability",
                        "measure_subject_class_separability",
                        {"session_id": "1"},
                    ),
                    ToolCall(
                        "stability",
                        "measure_subject_stability",
                        {"session_ids": ["1", "2"]},
                    ),
                )
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "record",
                        "record_subject_profile",
                        {"subject_profile": subject_profile},
                    ),
                )
            ),
            ModelResponse(content="Evidence-linked subject profile completed."),
        ]
    )
    result = SubjectProfilerAgent(
        runtime=AgentRuntime(provider=provider, tools=tools, run_id="subject-loop"),
        context=context,
    ).run()
    recorded = result.latest_tool_result("record_subject_profile")
    assert result.status == "completed"
    assert recorded["profile_complete"] is True
    assert recorded["raw_arrays_exposed_to_model"] is False
    assert len(recorded["measurements"]) == 4


def test_subject_measurement_cannot_cross_authorized_role(tmp_path: Path) -> None:
    profile_path = tmp_path / "dataset_profile.json"
    protocol_path = tmp_path / "frozen_protocol.json"
    _write(profile_path, _dataset_profile())
    _write(protocol_path, _frozen_protocol())
    engine = SubjectMeasurementEngine(
        source=_SyntheticSource(),
        subject_id="sub-001",
        allowed_session_ids=["1", "2"],
    )
    tools, _ = create_subject_profiler_tools(
        engine=engine,
        dataset_profile_path=profile_path,
        frozen_protocol_path=protocol_path,
    )
    tools.execute("read_subject_profile_context", {})
    with pytest.raises(ToolExecutionError, match="outside the authorized profiling role"):
        tools.execute("measure_subject_signal_quality", {"session_id": "3"})
