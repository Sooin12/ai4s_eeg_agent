from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bci_autodiscovery.agents.pipeline_lock_critic import (
    PipelineLockCriticError,
    validate_pipeline_lock,
)
from bci_autodiscovery.agents.pipeline_search import create_pipeline_search_tools
from bci_autodiscovery.agents.tools import ToolExecutionError
from bci_autodiscovery.pipelines import DeterministicPipelineExecutor, PipelineSpec
from bci_autodiscovery.profiling.subject_measurements import EpochSession
from bci_autodiscovery.search import (
    DatasetIncumbentError,
    build_dataset_incumbent,
    validate_dataset_incumbent,
)
from tests.unit.test_pipeline_search_loop import _contracts


def _session(subject_id: str, *, seed: int) -> EpochSession:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.asarray([1, 2]), 30)
    sfreq = 128.0
    samples = 256
    time = np.arange(samples) / sfreq
    data = rng.normal(scale=0.8, size=(60, 6, samples))
    rhythm = np.sin(2 * np.pi * 11.0 * time)
    data[labels == 1, 1, :] += 1.1 * rhythm
    data[labels == 2, 4, :] += 1.1 * rhythm
    return EpochSession(
        subject_id=subject_id,
        session_id="2",
        data=data,
        labels=labels,
        sampling_frequency_hz=sfreq,
        channel_names=("F3", "C3", "Cz", "C4", "P4", "Oz"),
        provenance={"path": f"synthetic-{subject_id}", "sha256": f"sha-{subject_id}"},
    )


def _candidate(pipeline_id: str, band: tuple[float, float]) -> PipelineSpec:
    return PipelineSpec(
        pipeline_id=pipeline_id,
        family="bandpower_lda",
        bandpass_hz=band,
        spatial_filter="none",
        csp_components=0,
        feature="log_bandpower",
        model="shrinkage_lda",
        lda_shrinkage=0.1,
        cv_folds=3,
        random_seed=5,
    )


def _incumbent() -> dict:
    executors = {
        subject_id: DeterministicPipelineExecutor(
            sessions=[_session(subject_id, seed=seed)]
        )
        for subject_id, seed in (("sub-001", 1), ("sub-002", 2), ("sub-003", 3))
    }
    return build_dataset_incumbent(
        dataset_id="search-fixture",
        executors=executors,
        candidates=[
            _candidate("dataset-wide", (8.0, 30.0)),
            _candidate("narrow-mu", (8.0, 13.0)),
        ],
        minimum_personalization_gain=1.0,
    )


def test_dataset_incumbent_is_subject_balanced_and_tamper_evident() -> None:
    artifact = _incumbent()
    validate_dataset_incumbent(artifact)
    assert artifact["status"] == "frozen_dataset_pipeline_incumbent"
    assert artifact["selection_policy"]["subject_weighting"] == "equal_subject_macro"
    assert artifact["confirmation_data_accessed"] is False
    assert len(artifact["cohort_subject_ids"]) == 3

    artifact["selected_score_summary"]["macro_mean"] += 0.01
    with pytest.raises(DatasetIncumbentError, match="score summary is inconsistent"):
        validate_dataset_incumbent(artifact)


def test_subject_lock_must_follow_frozen_personalization_fallback(tmp_path: Path) -> None:
    subject_path, protocol_path, envelope_path = _contracts(tmp_path)
    incumbent = _incumbent()
    incumbent_path = tmp_path / "dataset_incumbent.json"
    incumbent_path.write_text(
        json.dumps(incumbent, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    executor = DeterministicPipelineExecutor(sessions=[_session("sub-001", seed=66)])
    tools, _ = create_pipeline_search_tools(
        executor=executor,
        subject_profile_path=subject_path,
        frozen_protocol_path=protocol_path,
        autonomy_envelope_path=envelope_path,
        capability_registry_path=Path("configs/executable_pipeline_capabilities.v0.json"),
        dataset_incumbent_path=incumbent_path,
    )
    context = tools.execute("read_pipeline_search_context", {})
    assert context["dataset_pipeline_incumbent"]["mandatory_control"] is True
    incumbent_band = tuple(incumbent["selected_pipeline"]["bandpass_hz"])
    alternative_band = (8.0, 13.0) if incumbent_band != (8.0, 13.0) else (8.0, 30.0)
    alternative = _candidate("subject-alternative", alternative_band).to_dict()
    with pytest.raises(ToolExecutionError, match="first control"):
        tools.execute("evaluate_pipeline_candidate", {"pipeline": alternative})
    incumbent_result = tools.execute(
        "evaluate_pipeline_candidate",
        {"pipeline": incumbent["selected_pipeline"]},
    )
    alternative_result = tools.execute(
        "evaluate_pipeline_candidate", {"pipeline": alternative}
    )
    common = {
        "evidence_experiment_ids": [
            incumbent_result["experiment_id"],
            alternative_result["experiment_id"],
        ],
        "selection_rationale": ["Apply the frozen selective personalization gate."],
        "rejected_alternatives": ["Insufficient search gain over the incumbent."],
        "uncertainty": ["Frozen confirmation remains unseen."],
        "stop_reason": "The controlled comparison is complete.",
    }
    with pytest.raises(ToolExecutionError, match="selective personalization gate"):
        tools.execute(
            "lock_pipeline",
            {
                "selected_experiment_id": alternative_result["experiment_id"],
                **common,
            },
        )
    lock = tools.execute(
        "lock_pipeline",
        {
            "selected_experiment_id": incumbent_result["experiment_id"],
            **common,
        },
    )
    assert lock["route_decision"]["mode"] == "fallback_to_dataset_incumbent"
    assert lock["route_decision"]["confirmation_outcomes_observed"] is False
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    validate_pipeline_lock(lock, protocol=protocol, envelope=envelope)
    lock["route_decision"]["mode"] = "personalized"
    with pytest.raises(PipelineLockCriticError, match="differs from frozen policy"):
        validate_pipeline_lock(lock, protocol=protocol, envelope=envelope)
