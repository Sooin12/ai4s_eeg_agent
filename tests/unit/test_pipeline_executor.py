from __future__ import annotations

import numpy as np
import pytest

from bci_autodiscovery.pipelines import (
    CandidateExecutionError,
    DeterministicPipelineExecutor,
)
from bci_autodiscovery.profiling.subject_measurements import EpochSession


def _session(seed: int = 7) -> EpochSession:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.asarray([0, 1]), 40)
    sfreq = 128.0
    samples = 256
    time = np.arange(samples) / sfreq
    data = rng.normal(scale=0.8, size=(80, 6, samples))
    rhythm = np.sin(2 * np.pi * 11.0 * time)
    data[labels == 0, 1, :] += 1.5 * rhythm
    data[labels == 1, 4, :] += 1.5 * rhythm
    return EpochSession(
        subject_id="sub-executor",
        session_id="search-1",
        data=data,
        labels=labels,
        sampling_frequency_hz=sfreq,
        channel_names=("F3", "C3", "Cz", "C4", "P4", "Oz"),
        provenance={"path": "synthetic", "sha256": "fixture-search-1"},
    )


def _spec(family: str) -> dict:
    return {
        "pipeline_id": f"{family}-fixture",
        "family": family,
        "bandpass_hz": [8.0, 30.0],
        "spatial_filter": "csp" if family == "csp_lda" else "none",
        "csp_components": 4 if family == "csp_lda" else 0,
        "feature": "csp_log_variance" if family == "csp_lda" else "log_bandpower",
        "model": "shrinkage_lda",
        "lda_shrinkage": 0.1,
        "cv_folds": 4,
        "random_seed": 42,
    }


@pytest.mark.parametrize("family", ["bandpower_lda", "csp_lda"])
def test_complete_pipeline_is_deterministic_and_training_fold_safe(family: str) -> None:
    executor = DeterministicPipelineExecutor(sessions=[_session()])
    first = executor.evaluate(_spec(family))
    second = executor.evaluate(_spec(family))
    assert first["experiment_id"] == second["experiment_id"]
    assert first["metrics"] == second["metrics"]
    assert first["metrics"]["balanced_accuracy"] >= 0.8
    assert first["confirmation_data_accessed"] is False
    assert all(
        fold["training_only_fits"]["feature_standardization"] is True
        and fold["training_only_fits"]["classifier"] is True
        for fold in first["validation"]["fold_results"]
    )
    if family == "csp_lda":
        assert all(
            fold["training_only_fits"]["spatial_filter_fitted"] is True
            for fold in first["validation"]["fold_results"]
        )


def test_executor_rejects_invalid_or_nonfinite_candidate() -> None:
    session = _session()
    executor = DeterministicPipelineExecutor(sessions=[session])
    invalid = _spec("csp_lda")
    invalid["csp_components"] = 3
    with pytest.raises(CandidateExecutionError, match="even integer"):
        executor.evaluate(invalid)
    nonfinite = np.asarray(session.data).copy()
    nonfinite[0, 0, 0] = np.nan
    bad_session = EpochSession(
        subject_id=session.subject_id,
        session_id=session.session_id,
        data=nonfinite,
        labels=session.labels,
        sampling_frequency_hz=session.sampling_frequency_hz,
        channel_names=session.channel_names,
        provenance=session.provenance,
    )
    with pytest.raises(CandidateExecutionError, match="non-finite"):
        DeterministicPipelineExecutor(sessions=[bad_session]).evaluate(_spec("bandpower_lda"))


@pytest.mark.parametrize("family", ["bandpower_lda", "csp_lda"])
def test_locked_model_is_serializable_and_never_refits_confirmation(family: str) -> None:
    search = _session(seed=7)
    confirmation = _session(seed=8)
    confirmation = EpochSession(
        subject_id=search.subject_id,
        session_id="confirmation-1",
        data=confirmation.data,
        labels=confirmation.labels,
        sampling_frequency_hz=confirmation.sampling_frequency_hz,
        channel_names=confirmation.channel_names,
        provenance={"path": "synthetic-confirmation", "sha256": "fixture-confirmation"},
    )
    executor = DeterministicPipelineExecutor(sessions=[search])
    fitted = executor.fit(_spec(family))
    result = executor.evaluate_fitted(
        fitted,
        sessions=[confirmation],
        data_role="frozen_confirmation",
    )
    assert fitted.model_sha256
    assert fitted.to_dict()["training_session_ids"] == ["search-1"]
    assert result["data_role"] == "frozen_confirmation"
    assert result["fitting_performed_on_evaluation_data"] is False
    assert result["metrics"]["balanced_accuracy"] >= 0.75
