"""Deterministic, leakage-safe execution of declarative EEG pipeline candidates."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt

from bci_autodiscovery.profiling.subject_measurements import EpochSession


class CandidateExecutionError(ValueError):
    pass


@dataclass(frozen=True)
class PipelineSpec:
    pipeline_id: str
    family: str
    bandpass_hz: tuple[float, float]
    spatial_filter: str
    csp_components: int
    feature: str
    model: str
    lda_shrinkage: float
    cv_folds: int
    random_seed: int
    channel_strategy: str = "all"
    selected_channels: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PipelineSpec":
        try:
            band = tuple(float(item) for item in value["bandpass_hz"])
            if len(band) != 2:
                raise ValueError
            spec = cls(
                pipeline_id=str(value["pipeline_id"]),
                family=str(value["family"]),
                bandpass_hz=(band[0], band[1]),
                spatial_filter=str(value["spatial_filter"]),
                csp_components=int(value["csp_components"]),
                feature=str(value["feature"]),
                model=str(value["model"]),
                lda_shrinkage=float(value["lda_shrinkage"]),
                cv_folds=int(value["cv_folds"]),
                random_seed=int(value["random_seed"]),
                channel_strategy=str(value.get("channel_strategy", "all")),
                selected_channels=tuple(
                    str(item) for item in value.get("selected_channels", [])
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CandidateExecutionError(f"Malformed pipeline specification: {exc}") from exc
        validate_pipeline_spec(spec)
        return spec

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["bandpass_hz"] = list(self.bandpass_hz)
        value["selected_channels"] = list(self.selected_channels)
        return value


@dataclass(frozen=True)
class FittedPipeline:
    """Serializable training-side state used exactly once on confirmation data."""

    spec: dict[str, Any]
    sampling_frequency_hz: float
    channel_count: int
    input_channel_names: list[str]
    selected_channel_names: list[str]
    spatial_filters: list[list[float]] | None
    feature_mean: list[float]
    feature_scale: list[float]
    classes: list[Any]
    class_means: list[list[float]]
    precision: list[list[float]]
    priors: list[float]
    training_subject_id: str
    training_session_ids: list[str]
    training_provenance: list[dict[str, Any]]
    model_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_pipeline_spec(spec: PipelineSpec) -> None:
    if not spec.pipeline_id.strip():
        raise CandidateExecutionError("pipeline_id must be non-empty")
    if spec.family not in {"bandpower_lda", "csp_lda"}:
        raise CandidateExecutionError(f"Unsupported pipeline family: {spec.family}")
    low, high = spec.bandpass_hz
    if not 0 < low < high:
        raise CandidateExecutionError("bandpass_hz must satisfy 0 < low < high")
    if spec.family == "bandpower_lda":
        if spec.spatial_filter != "none" or spec.feature != "log_bandpower":
            raise CandidateExecutionError(
                "bandpower_lda requires spatial_filter=none and feature=log_bandpower"
            )
    if spec.family == "csp_lda":
        if spec.spatial_filter != "csp" or spec.feature != "csp_log_variance":
            raise CandidateExecutionError(
                "csp_lda requires spatial_filter=csp and feature=csp_log_variance"
            )
        if spec.csp_components < 2 or spec.csp_components % 2:
            raise CandidateExecutionError("CSP components must be an even integer >= 2")
    if spec.model != "shrinkage_lda":
        raise CandidateExecutionError("Only shrinkage_lda is currently executable")
    if not 0 <= spec.lda_shrinkage <= 1:
        raise CandidateExecutionError("lda_shrinkage must be in [0, 1]")
    if spec.cv_folds < 2:
        raise CandidateExecutionError("cv_folds must be >= 2")
    if spec.random_seed < 0:
        raise CandidateExecutionError("random_seed must be non-negative")
    if spec.channel_strategy not in {"all", "named"}:
        raise CandidateExecutionError("channel_strategy must be all or named")
    if spec.channel_strategy == "all" and spec.selected_channels:
        raise CandidateExecutionError("all-channel pipelines cannot name selected_channels")
    if spec.channel_strategy == "named":
        if not spec.selected_channels:
            raise CandidateExecutionError("named channel strategy requires selected_channels")
        if len(spec.selected_channels) != len(set(spec.selected_channels)):
            raise CandidateExecutionError("selected_channels must be unique")


def pipeline_spec_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "pipeline_id": {"type": "string"},
            "family": {"type": "string", "enum": ["bandpower_lda", "csp_lda"]},
            "bandpass_hz": {"type": "array", "items": {"type": "number"}},
            "spatial_filter": {"type": "string", "enum": ["none", "csp"]},
            "csp_components": {"type": "integer"},
            "feature": {
                "type": "string",
                "enum": ["log_bandpower", "csp_log_variance"],
            },
            "model": {"type": "string", "enum": ["shrinkage_lda"]},
            "lda_shrinkage": {"type": "number"},
            "cv_folds": {"type": "integer"},
            "random_seed": {"type": "integer"},
            "channel_strategy": {"type": "string", "enum": ["all", "named"]},
            "selected_channels": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
        },
        "required": [
            "pipeline_id",
            "family",
            "bandpass_hz",
            "spatial_filter",
            "csp_components",
            "feature",
            "model",
            "lda_shrinkage",
            "cv_folds",
            "random_seed",
        ],
        "additionalProperties": False,
    }


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pipeline_configuration_hash(spec_value: dict[str, Any] | PipelineSpec) -> str:
    """Identify an executable configuration independently of its human-readable ID."""

    spec = spec_value if isinstance(spec_value, PipelineSpec) else PipelineSpec.from_dict(spec_value)
    value = spec.to_dict()
    value.pop("pipeline_id", None)
    return _canonical_hash(value)


def _stratified_folds(labels: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    labels = np.asarray(labels).squeeze()
    classes = np.unique(labels)
    if classes.size != 2:
        raise CandidateExecutionError("Current executor requires binary labels")
    rng = np.random.default_rng(seed)
    fold_parts: list[list[np.ndarray]] = [[] for _ in range(folds)]
    for label in classes:
        indices = np.flatnonzero(labels == label)
        if indices.size < folds:
            raise CandidateExecutionError(
                f"Class {label!r} has fewer trials than cv_folds={folds}"
            )
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        for fold_index, part in enumerate(np.array_split(shuffled, folds)):
            fold_parts[fold_index].append(part)
    return [np.sort(np.concatenate(parts)) for parts in fold_parts]


def _bandpass(data: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    low, high = band
    nyquist = sfreq / 2.0
    if high >= nyquist:
        raise CandidateExecutionError(
            f"Bandpass high edge {high} must be below Nyquist {nyquist}"
        )
    sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    try:
        return sosfiltfilt(sos, data, axis=-1)
    except ValueError as exc:
        raise CandidateExecutionError(f"Bandpass failed: {exc}") from exc


def _select_channels(
    data: np.ndarray,
    channel_names: Sequence[str],
    spec: PipelineSpec,
) -> tuple[np.ndarray, tuple[str, ...]]:
    names = tuple(str(item) for item in channel_names)
    if len(names) != len(set(names)):
        raise CandidateExecutionError("Signal contract contains duplicate channel names")
    if spec.channel_strategy == "all":
        return data, names
    missing = [name for name in spec.selected_channels if name not in names]
    if missing:
        raise CandidateExecutionError(f"Selected channels are unavailable: {missing}")
    indices = [names.index(name) for name in spec.selected_channels]
    return data[:, indices, :], spec.selected_channels


def _reorder_channels(
    data: np.ndarray,
    actual_names: Sequence[str],
    expected_names: Sequence[str],
) -> np.ndarray:
    actual = tuple(str(item) for item in actual_names)
    expected = tuple(str(item) for item in expected_names)
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise CandidateExecutionError("Evaluation channel contract differs from fitted model")
    return data[:, [actual.index(name) for name in expected], :]


def _normalized_covariances(data: np.ndarray) -> np.ndarray:
    centered = data - np.mean(data, axis=-1, keepdims=True)
    covariances = centered @ np.transpose(centered, (0, 2, 1))
    traces = np.trace(covariances, axis1=1, axis2=2)
    traces = np.maximum(traces, np.finfo(float).eps)
    return covariances / traces[:, None, None]


def _fit_csp(data: np.ndarray, labels: np.ndarray, components: int) -> np.ndarray:
    classes = np.unique(labels)
    if classes.size != 2:
        raise CandidateExecutionError("CSP requires exactly two classes")
    if components > data.shape[1]:
        raise CandidateExecutionError("CSP components exceed channel count")
    covariances = _normalized_covariances(data)
    class_cov = [np.mean(covariances[labels == label], axis=0) for label in classes]
    regularizer = np.eye(data.shape[1]) * 1e-8
    composite = class_cov[0] + class_cov[1] + regularizer
    try:
        eigenvalues, vectors = eigh(class_cov[0] + regularizer, composite)
    except np.linalg.LinAlgError as exc:
        raise CandidateExecutionError(f"CSP eigendecomposition failed: {exc}") from exc
    order = np.argsort(eigenvalues)
    half = components // 2
    selected = np.concatenate([order[:half], order[-half:]])
    return vectors[:, selected].T


def _features(
    train_data: np.ndarray,
    evaluation_data: np.ndarray,
    train_labels: np.ndarray,
    spec: PipelineSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_features, state = _fit_feature_state(train_data, train_labels, spec)
    evaluation_features = _apply_feature_state(evaluation_data, spec, state)
    filters = state.get("spatial_filters")
    return train_features, evaluation_features, {
        "spatial_filter_fitted": filters is not None,
        **({"csp_filter_shape": list(filters.shape)} if filters is not None else {}),
    }


def _fit_feature_state(
    data: np.ndarray,
    labels: np.ndarray,
    spec: PipelineSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    if spec.family == "bandpower_lda":
        return np.log(np.var(data, axis=-1) + 1e-12), {"spatial_filters": None}
    filters = _fit_csp(data, labels, spec.csp_components)
    projected = np.einsum("kc,tcs->tks", filters, data)
    variance = np.var(projected, axis=-1)
    features = np.log(variance / np.sum(variance, axis=1, keepdims=True) + 1e-12)
    return features, {"spatial_filters": filters}


def _apply_feature_state(
    data: np.ndarray,
    spec: PipelineSpec,
    state: dict[str, Any],
) -> np.ndarray:
    filters = state.get("spatial_filters")
    if spec.family == "bandpower_lda":
        if filters is not None:
            raise CandidateExecutionError("Bandpower pipeline received unexpected spatial filters")
        return np.log(np.var(data, axis=-1) + 1e-12)
    if filters is None:
        raise CandidateExecutionError("CSP pipeline lacks fitted spatial filters")
    projected = np.einsum("kc,tcs->tks", np.asarray(filters), data)
    variance = np.var(projected, axis=-1)
    return np.log(variance / np.sum(variance, axis=1, keepdims=True) + 1e-12)


def _standardize(
    train: np.ndarray, evaluation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(scale <= 1e-12, 1.0, scale)
    return (train - mean) / scale, (evaluation - mean) / scale


def _fit_predict_lda(
    train: np.ndarray,
    labels: np.ndarray,
    evaluation: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    state = _fit_lda_state(train, labels, shrinkage)
    return _predict_lda_state(evaluation, state)


def _fit_lda_state(
    train: np.ndarray,
    labels: np.ndarray,
    shrinkage: float,
) -> dict[str, np.ndarray]:
    classes = np.unique(labels)
    means = np.stack([np.mean(train[labels == label], axis=0) for label in classes])
    residuals = np.concatenate(
        [train[labels == label] - means[index] for index, label in enumerate(classes)],
        axis=0,
    )
    denominator = max(1, train.shape[0] - classes.size)
    covariance = residuals.T @ residuals / denominator
    target = np.eye(covariance.shape[0]) * np.trace(covariance) / covariance.shape[0]
    covariance = (1.0 - shrinkage) * covariance + shrinkage * target
    precision = np.linalg.pinv(covariance, hermitian=True)
    priors = np.asarray([np.mean(labels == label) for label in classes])
    return {
        "classes": classes,
        "means": means,
        "precision": precision,
        "priors": priors,
    }


def _predict_lda_state(
    evaluation: np.ndarray,
    state: dict[str, np.ndarray],
) -> np.ndarray:
    classes = state["classes"]
    means = state["means"]
    precision = state["precision"]
    priors = state["priors"]
    scores = np.stack(
        [
            evaluation @ precision @ mean
            - 0.5 * mean @ precision @ mean
            + np.log(max(prior, 1e-12))
            for mean, prior in zip(means, priors, strict=True)
        ],
        axis=1,
    )
    return classes[np.argmax(scores, axis=1)]


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    classes = np.unique(labels)
    recalls = [float(np.mean(predictions[labels == label] == label)) for label in classes]
    accuracy = float(np.mean(predictions == labels))
    expected = float(
        sum(
            np.mean(labels == label) * np.mean(predictions == label)
            for label in np.union1d(labels, predictions)
        )
    )
    kappa = (accuracy - expected) / (1.0 - expected) if expected < 1.0 else 0.0
    return {
        "balanced_accuracy": float(np.mean(recalls)),
        "accuracy": accuracy,
        "kappa": float(kappa),
    }


class DeterministicPipelineExecutor:
    """Evaluate complete candidates using only authorized search sessions."""

    def __init__(self, *, sessions: Sequence[EpochSession]) -> None:
        if not sessions:
            raise CandidateExecutionError("At least one search session is required")
        subject_ids = {session.subject_id for session in sessions}
        if len(subject_ids) != 1:
            raise CandidateExecutionError("One executor cannot mix subjects")
        for session in sessions:
            session.validate()
        self.sessions = tuple(sessions)
        self.subject_id = sessions[0].subject_id

    def evaluate(self, spec_value: dict[str, Any] | PipelineSpec) -> dict[str, Any]:
        spec = (
            spec_value
            if isinstance(spec_value, PipelineSpec)
            else PipelineSpec.from_dict(spec_value)
        )
        validate_pipeline_spec(spec)
        started = time.perf_counter()
        if any(
            session.data.shape[1:] != self.sessions[0].data.shape[1:]
            or session.sampling_frequency_hz != self.sessions[0].sampling_frequency_hz
            or tuple(session.channel_names) != tuple(self.sessions[0].channel_names)
            for session in self.sessions
        ):
            raise CandidateExecutionError("Search sessions have incompatible signal contracts")
        data = np.concatenate([np.asarray(session.data) for session in self.sessions], axis=0)
        labels = np.concatenate(
            [np.asarray(session.labels).squeeze() for session in self.sessions], axis=0
        )
        if not np.all(np.isfinite(data)):
            raise CandidateExecutionError("Executor refuses non-finite search data")
        data, selected_channel_names = _select_channels(
            data,
            self.sessions[0].channel_names,
            spec,
        )
        if spec.family == "csp_lda" and spec.csp_components > data.shape[1]:
            raise CandidateExecutionError("CSP components exceed selected channel count")
        filtered = _bandpass(
            np.asarray(data, dtype=np.float64),
            self.sessions[0].sampling_frequency_hz,
            spec.bandpass_hz,
        )
        folds = _stratified_folds(labels, spec.cv_folds, spec.random_seed)
        fold_results: list[dict[str, Any]] = []
        all_predictions = np.empty_like(labels)
        for fold_index, validation_indices in enumerate(folds):
            train_mask = np.ones(labels.size, dtype=bool)
            train_mask[validation_indices] = False
            train_indices = np.flatnonzero(train_mask)
            train_features, validation_features, fitted = _features(
                filtered[train_indices],
                filtered[validation_indices],
                labels[train_indices],
                spec,
            )
            train_features, validation_features = _standardize(
                train_features,
                validation_features,
            )
            predictions = _fit_predict_lda(
                train_features,
                labels[train_indices],
                validation_features,
                spec.lda_shrinkage,
            )
            all_predictions[validation_indices] = predictions
            fold_results.append(
                {
                    "fold_index": fold_index,
                    "train_trials": int(train_indices.size),
                    "validation_trials": int(validation_indices.size),
                    "metrics": _metrics(labels[validation_indices], predictions),
                    "training_only_fits": {
                        "bandpass_fixed_from_spec": True,
                        "feature_standardization": True,
                        "classifier": True,
                        **fitted,
                    },
                }
            )
        overall = _metrics(labels, all_predictions)
        spec_dict = spec.to_dict()
        provenance = [session.provenance for session in self.sessions]
        identity = {
            "subject_id": self.subject_id,
            "pipeline": spec_dict,
            "session_provenance": provenance,
        }
        return {
            "schema_version": "1.0",
            "experiment_id": f"experiment-{_canonical_hash(identity)[:16]}",
            "subject_id": self.subject_id,
            "pipeline": spec_dict,
            "pipeline_sha256": _canonical_hash(spec_dict),
            "status": "completed",
            "data_role": "pipeline_search_and_lock",
            "session_ids": [session.session_id for session in self.sessions],
            "trial_count": int(labels.size),
            "class_counts": {
                str(label): int(np.count_nonzero(labels == label))
                for label in np.unique(labels)
            },
            "selected_channel_names": list(selected_channel_names),
            "validation": {
                "scheme": "deterministic_stratified_kfold_within_search_role",
                "folds": spec.cv_folds,
                "random_seed": spec.random_seed,
                "fold_results": fold_results,
            },
            "metrics": overall,
            "elapsed_seconds": float(time.perf_counter() - started),
            "provenance": provenance,
            "confirmation_data_accessed": False,
        }

    def fit(self, spec_value: dict[str, Any] | PipelineSpec) -> FittedPipeline:
        """Fit all learned state on search-role data after candidate selection."""

        spec = spec_value if isinstance(spec_value, PipelineSpec) else PipelineSpec.from_dict(spec_value)
        validate_pipeline_spec(spec)
        data, labels, sfreq = self._combined_sessions(self.sessions)
        input_channel_names = tuple(self.sessions[0].channel_names)
        data, selected_channel_names = _select_channels(data, input_channel_names, spec)
        if spec.family == "csp_lda" and spec.csp_components > data.shape[1]:
            raise CandidateExecutionError("CSP components exceed selected channel count")
        filtered = _bandpass(data, sfreq, spec.bandpass_hz)
        features, feature_state = _fit_feature_state(filtered, labels, spec)
        feature_mean = np.mean(features, axis=0)
        feature_scale = np.std(features, axis=0)
        feature_scale = np.where(feature_scale <= 1e-12, 1.0, feature_scale)
        standardized = (features - feature_mean) / feature_scale
        lda = _fit_lda_state(standardized, labels, spec.lda_shrinkage)
        filters = feature_state.get("spatial_filters")
        state_without_hash = {
            "spec": spec.to_dict(),
            "sampling_frequency_hz": sfreq,
            "channel_count": int(data.shape[1]),
            "input_channel_names": list(input_channel_names),
            "selected_channel_names": list(selected_channel_names),
            "spatial_filters": filters.tolist() if filters is not None else None,
            "feature_mean": feature_mean.tolist(),
            "feature_scale": feature_scale.tolist(),
            "classes": lda["classes"].tolist(),
            "class_means": lda["means"].tolist(),
            "precision": lda["precision"].tolist(),
            "priors": lda["priors"].tolist(),
            "training_subject_id": self.subject_id,
            "training_session_ids": [session.session_id for session in self.sessions],
            "training_provenance": [session.provenance for session in self.sessions],
        }
        return FittedPipeline(
            **state_without_hash,
            model_sha256=_canonical_hash(state_without_hash),
        )

    def evaluate_fitted(
        self,
        fitted: FittedPipeline,
        *,
        sessions: Sequence[EpochSession],
        data_role: str,
    ) -> dict[str, Any]:
        """Apply immutable fitted state without refitting any confirmation-side parameter."""

        if not sessions:
            raise CandidateExecutionError("At least one evaluation session is required")
        if any(session.subject_id != fitted.training_subject_id for session in sessions):
            raise CandidateExecutionError("Fitted pipeline cannot be applied to another subject")
        data, labels, sfreq = self._combined_sessions(sessions)
        if sfreq != fitted.sampling_frequency_hz:
            raise CandidateExecutionError("Evaluation signal contract differs from fitted model")
        spec = PipelineSpec.from_dict(fitted.spec)
        data = _reorder_channels(
            data,
            sessions[0].channel_names,
            fitted.input_channel_names,
        )
        data, selected_channel_names = _select_channels(
            data,
            fitted.input_channel_names,
            spec,
        )
        if (
            data.shape[1] != fitted.channel_count
            or list(selected_channel_names) != fitted.selected_channel_names
        ):
            raise CandidateExecutionError("Evaluation channel selection differs from fitted model")
        filtered = _bandpass(data, sfreq, spec.bandpass_hz)
        feature_state = {
            "spatial_filters": (
                np.asarray(fitted.spatial_filters, dtype=float)
                if fitted.spatial_filters is not None
                else None
            )
        }
        features = _apply_feature_state(filtered, spec, feature_state)
        standardized = (
            features - np.asarray(fitted.feature_mean, dtype=float)
        ) / np.asarray(fitted.feature_scale, dtype=float)
        predictions = _predict_lda_state(
            standardized,
            {
                "classes": np.asarray(fitted.classes),
                "means": np.asarray(fitted.class_means, dtype=float),
                "precision": np.asarray(fitted.precision, dtype=float),
                "priors": np.asarray(fitted.priors, dtype=float),
            },
        )
        return {
            "schema_version": "1.0",
            "subject_id": fitted.training_subject_id,
            "data_role": data_role,
            "session_ids": [session.session_id for session in sessions],
            "trial_count": int(labels.size),
            "metrics": _metrics(labels, predictions),
            "model_sha256": fitted.model_sha256,
            "pipeline_sha256": _canonical_hash(fitted.spec),
            "fitting_performed_on_evaluation_data": False,
            "evaluation_provenance": [session.provenance for session in sessions],
        }

    @staticmethod
    def _combined_sessions(
        sessions: Sequence[EpochSession],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        first = sessions[0]
        for session in sessions:
            session.validate()
            if (
                session.data.shape[1:] != first.data.shape[1:]
                or session.sampling_frequency_hz != first.sampling_frequency_hz
                or tuple(session.channel_names) != tuple(first.channel_names)
            ):
                raise CandidateExecutionError("Sessions have incompatible signal contracts")
        data = np.concatenate(
            [np.asarray(session.data, dtype=np.float64) for session in sessions],
            axis=0,
        )
        labels = np.concatenate(
            [np.asarray(session.labels).squeeze() for session in sessions],
            axis=0,
        )
        if not np.all(np.isfinite(data)):
            raise CandidateExecutionError("Executor refuses non-finite data")
        return data, labels, float(first.sampling_frequency_hz)
