"""Dataset-neutral deterministic measurements for subject-level EEG profiles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch


class SubjectMeasurementError(ValueError):
    pass


@dataclass(frozen=True)
class EpochSession:
    """Standard in-memory epoch contract; arrays never leave deterministic tools."""

    subject_id: str
    session_id: str
    data: np.ndarray  # trial x channel x sample
    labels: np.ndarray  # trial
    sampling_frequency_hz: float
    channel_names: tuple[str, ...]
    provenance: dict[str, Any]

    def validate(self) -> None:
        data = np.asarray(self.data)
        labels = np.asarray(self.labels).squeeze()
        if data.ndim != 3:
            raise SubjectMeasurementError("Epoch data must be trial x channel x sample")
        if labels.ndim != 1 or labels.size != data.shape[0]:
            raise SubjectMeasurementError("Labels must contain one value per trial")
        if len(self.channel_names) != data.shape[1]:
            raise SubjectMeasurementError("channel_names do not match the channel axis")
        if self.sampling_frequency_hz <= 0:
            raise SubjectMeasurementError("sampling_frequency_hz must be positive")
        if not np.issubdtype(data.dtype, np.number):
            raise SubjectMeasurementError("Epoch data must be numeric")


class SubjectEpochSource(Protocol):
    source_id: str

    def load_session(self, *, subject_id: str, session_id: str) -> EpochSession: ...


def _jsonable_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _measurement(
    *,
    kind: str,
    subject_id: str,
    session_ids: Sequence[str],
    payload: dict[str, Any],
    provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    body = {
        "schema_version": "1.0",
        "kind": kind,
        "subject_id": subject_id,
        "session_ids": list(session_ids),
        "payload": payload,
        "provenance": provenance,
    }
    canonical = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    body["measurement_id"] = f"{kind}-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}"
    return body


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale <= np.finfo(float).eps:
        return np.zeros_like(values)
    return (values - median) / scale


def _welch_psd(session: EpochSession) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(session.data, dtype=np.float64)
    if not np.all(np.isfinite(data)):
        raise SubjectMeasurementError("Spectral measurement refuses non-finite epoch data")
    data = data - np.mean(data, axis=-1, keepdims=True)
    nperseg = min(data.shape[-1], max(64, int(round(session.sampling_frequency_hz))))
    freqs, psd = welch(
        data,
        fs=session.sampling_frequency_hz,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        axis=-1,
        detrend="constant",
    )
    return freqs, psd


def _bandpower(
    freqs: np.ndarray,
    psd: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    mask = (freqs >= low) & (freqs <= high)
    if np.count_nonzero(mask) < 2:
        raise SubjectMeasurementError(f"Insufficient frequency bins for {low}-{high} Hz")
    return trapezoid(psd[..., mask], freqs[mask], axis=-1)


def measure_signal_quality(session: EpochSession) -> dict[str, Any]:
    session.validate()
    data = np.asarray(session.data, dtype=np.float64)
    finite = np.isfinite(data)
    channel_std = np.nanstd(data, axis=(0, 2))
    trial_peak = np.nanmax(np.abs(data), axis=(1, 2))
    median_std = float(np.nanmedian(channel_std))
    flat_threshold = max(np.finfo(float).eps, median_std * 1e-4)
    flat_indices = np.flatnonzero(channel_std <= flat_threshold)
    outlier_indices = np.flatnonzero(np.abs(_robust_z(np.log(channel_std + 1e-12))) > 6.0)
    trial_outliers = np.flatnonzero(np.abs(_robust_z(np.log(trial_peak + 1e-12))) > 6.0)
    payload = {
        "trial_count": int(data.shape[0]),
        "channel_count": int(data.shape[1]),
        "sample_count": int(data.shape[2]),
        "finite_fraction": float(np.mean(finite)),
        "channel_std_median": _jsonable_float(median_std),
        "channel_std_iqr": _jsonable_float(
            float(np.nanpercentile(channel_std, 75) - np.nanpercentile(channel_std, 25))
        ),
        "flat_channel_indices": flat_indices.astype(int).tolist(),
        "flat_channel_names": [session.channel_names[index] for index in flat_indices],
        "robust_outlier_channel_indices": outlier_indices.astype(int).tolist(),
        "robust_outlier_channel_names": [
            session.channel_names[index] for index in outlier_indices
        ],
        "robust_outlier_trial_indices": trial_outliers.astype(int).tolist(),
        "automatic_exclusions": [],
        "interpretation_boundary": (
            "Flags are deterministic diagnostics, not scientific exclusion decisions."
        ),
    }
    return _measurement(
        kind="signal_quality",
        subject_id=session.subject_id,
        session_ids=[session.session_id],
        payload=payload,
        provenance=[session.provenance],
    )


def measure_spectral_profile(session: EpochSession) -> dict[str, Any]:
    session.validate()
    freqs, psd = _welch_psd(session)
    mean_psd = np.mean(psd, axis=(0, 1))
    total = float(_bandpower(freqs, mean_psd, 1.0, min(45.0, freqs[-1])))
    bands = {
        "theta_4_8": (4.0, 8.0),
        "mu_8_13": (8.0, 13.0),
        "beta_13_30": (13.0, 30.0),
        "motor_8_30": (8.0, 30.0),
    }
    bandpowers: dict[str, dict[str, float | None]] = {}
    for name, (low, high) in bands.items():
        power = float(_bandpower(freqs, mean_psd, low, high))
        bandpowers[name] = {
            "absolute": _jsonable_float(power),
            "relative_1_45": _jsonable_float(power / total if total > 0 else np.nan),
        }

    def peak(low: float, high: float) -> dict[str, float | None]:
        mask = (freqs >= low) & (freqs <= high)
        if not np.any(mask):
            return {"frequency_hz": None, "power": None}
        local_freqs = freqs[mask]
        local_psd = mean_psd[mask]
        index = int(np.argmax(local_psd))
        return {
            "frequency_hz": _jsonable_float(float(local_freqs[index])),
            "power": _jsonable_float(float(local_psd[index])),
        }

    payload = {
        "method": "Welch PSD over demeaned epochs; deterministic mean over trials/channels",
        "frequency_resolution_hz": _jsonable_float(float(freqs[1] - freqs[0])),
        "bandpowers": bandpowers,
        "mu_peak": peak(7.0, 15.0),
        "beta_peak": peak(15.0, 30.0),
        "individual_frequency_status": "measured_candidate_not_yet_validated",
        "erd_ers": {
            "status": "unavailable_without_pre_event_baseline_contract",
            "reason": (
                "ERD/ERS requires a documented baseline window; an event-only epoch cannot "
                "supply one by assumption."
            ),
        },
    }
    return _measurement(
        kind="spectral_profile",
        subject_id=session.subject_id,
        session_ids=[session.session_id],
        payload=payload,
        provenance=[session.provenance],
    )


def _class_effect_vector(
    session: EpochSession,
    band_hz: tuple[float, float] = (8.0, 30.0),
) -> tuple[np.ndarray, dict[str, Any]]:
    freqs, psd = _welch_psd(session)
    features = np.log(_bandpower(freqs, psd, band_hz[0], band_hz[1]) + 1e-12)
    labels = np.asarray(session.labels).squeeze()
    classes = np.unique(labels)
    if classes.size != 2:
        raise SubjectMeasurementError("Binary class-separability measurement requires two labels")
    left = features[labels == classes[0]]
    right = features[labels == classes[1]]
    pooled = np.sqrt((np.var(left, axis=0) + np.var(right, axis=0)) / 2.0 + 1e-12)
    effect = (np.mean(right, axis=0) - np.mean(left, axis=0)) / pooled
    return effect, {
        "classes": [str(value) for value in classes.tolist()],
        "class_counts": {
            str(classes[0]): int(left.shape[0]),
            str(classes[1]): int(right.shape[0]),
        },
    }


def measure_class_separability(session: EpochSession) -> dict[str, Any]:
    session.validate()
    effect, class_info = _class_effect_vector(session)
    ranked = np.argsort(np.abs(effect))[::-1]
    top = ranked[: min(10, ranked.size)]
    bandwise: dict[str, dict[str, Any]] = {}
    for name, band in {"mu_8_13": (8.0, 13.0), "beta_13_30": (13.0, 30.0)}.items():
        band_effect, _ = _class_effect_vector(session, band)
        band_ranked = np.argsort(np.abs(band_effect))[::-1]
        band_top = band_ranked[: min(3, band_ranked.size)]
        bandwise[name] = {
            "band_hz": list(band),
            "median_absolute_standardized_effect": _jsonable_float(
                float(np.median(np.abs(band_effect)))
            ),
            "max_absolute_standardized_effect": _jsonable_float(
                float(np.max(np.abs(band_effect)))
            ),
            "top_channels": [
                {
                    "channel_index": int(index),
                    "channel_name": session.channel_names[index],
                    "standardized_effect": _jsonable_float(float(band_effect[index])),
                }
                for index in band_top
            ],
        }
    payload = {
        "feature": "per-channel log 8-30 Hz bandpower",
        **class_info,
        "median_absolute_standardized_effect": _jsonable_float(
            float(np.median(np.abs(effect)))
        ),
        "max_absolute_standardized_effect": _jsonable_float(
            float(np.max(np.abs(effect)))
        ),
        "top_channels": [
            {
                "channel_index": int(index),
                "channel_name": session.channel_names[index],
                "standardized_effect": _jsonable_float(float(effect[index])),
            }
            for index in top
        ],
        "bandwise": bandwise,
        "interpretation_boundary": (
            "Univariate training-side diagnostic; not a cross-validated decoding result."
        ),
    }
    return _measurement(
        kind="class_separability",
        subject_id=session.subject_id,
        session_ids=[session.session_id],
        payload=payload,
        provenance=[session.provenance],
    )


def measure_stability(sessions: Sequence[EpochSession]) -> dict[str, Any]:
    if not sessions:
        raise SubjectMeasurementError("At least one session is required")
    for session in sessions:
        session.validate()
    subject_ids = {session.subject_id for session in sessions}
    if len(subject_ids) != 1:
        raise SubjectMeasurementError("Stability measurement cannot mix subjects")
    effects: list[np.ndarray] = []
    bandpower_means: list[np.ndarray] = []
    within_session: list[dict[str, Any]] = []
    for session in sessions:
        effect, _ = _class_effect_vector(session)
        effects.append(effect)
        freqs, psd = _welch_psd(session)
        feature = np.log(_bandpower(freqs, psd, 8.0, 30.0) + 1e-12)
        bandpower_means.append(np.mean(feature, axis=0))

        labels = np.asarray(session.labels).squeeze()
        split_effects: list[np.ndarray] = []
        for parity in (0, 1):
            selected: list[int] = []
            for label in np.unique(labels):
                indices = np.flatnonzero(labels == label)
                selected.extend(indices[parity::2].tolist())
            subset = EpochSession(
                subject_id=session.subject_id,
                session_id=session.session_id,
                data=session.data[np.asarray(sorted(selected), dtype=int)],
                labels=labels[np.asarray(sorted(selected), dtype=int)],
                sampling_frequency_hz=session.sampling_frequency_hz,
                channel_names=session.channel_names,
                provenance=session.provenance,
            )
            split_effects.append(_class_effect_vector(subset)[0])
        correlation = float(np.corrcoef(split_effects[0], split_effects[1])[0, 1])
        within_session.append(
            {
                "session_id": session.session_id,
                "split_half_effect_correlation": _jsonable_float(correlation),
            }
        )

    pairwise: list[dict[str, Any]] = []
    for left_index in range(len(sessions)):
        for right_index in range(left_index + 1, len(sessions)):
            effect_corr = float(
                np.corrcoef(effects[left_index], effects[right_index])[0, 1]
            )
            spectral_shift = float(
                np.linalg.norm(bandpower_means[left_index] - bandpower_means[right_index])
                / np.sqrt(bandpower_means[left_index].size)
            )
            pairwise.append(
                {
                    "left_session": sessions[left_index].session_id,
                    "right_session": sessions[right_index].session_id,
                    "class_effect_correlation": _jsonable_float(effect_corr),
                    "rms_log_bandpower_shift": _jsonable_float(spectral_shift),
                }
            )
    payload = {
        "within_session": within_session,
        "cross_session": pairwise,
        "cross_session_status": (
            "measured" if pairwise else "unavailable_insufficient_authorized_sessions"
        ),
    }
    return _measurement(
        kind="stability",
        subject_id=sessions[0].subject_id,
        session_ids=[session.session_id for session in sessions],
        payload=payload,
        provenance=[session.provenance for session in sessions],
    )


class SubjectMeasurementEngine:
    """Cache authorized sessions and expose raw-free deterministic measurements."""

    def __init__(
        self,
        *,
        source: SubjectEpochSource,
        subject_id: str,
        allowed_session_ids: Sequence[str],
    ) -> None:
        if not subject_id.strip():
            raise SubjectMeasurementError("subject_id must be non-empty")
        if not allowed_session_ids:
            raise SubjectMeasurementError("At least one authorized session is required")
        self.source = source
        self.subject_id = subject_id
        self.allowed_session_ids = tuple(str(item) for item in allowed_session_ids)
        self._cache: dict[str, EpochSession] = {}

    def _session(self, session_id: str) -> EpochSession:
        normalized = str(session_id)
        if normalized not in self.allowed_session_ids:
            raise SubjectMeasurementError(
                f"Session {normalized!r} is outside the authorized profiling role"
            )
        if normalized not in self._cache:
            session = self.source.load_session(
                subject_id=self.subject_id,
                session_id=normalized,
            )
            session.validate()
            self._cache[normalized] = session
        return self._cache[normalized]

    def quality(self, session_id: str) -> dict[str, Any]:
        return measure_signal_quality(self._session(session_id))

    def spectral(self, session_id: str) -> dict[str, Any]:
        return measure_spectral_profile(self._session(session_id))

    def separability(self, session_id: str) -> dict[str, Any]:
        return measure_class_separability(self._session(session_id))

    def stability(self, session_ids: Sequence[str] | None = None) -> dict[str, Any]:
        selected = tuple(session_ids or self.allowed_session_ids)
        if not selected:
            raise SubjectMeasurementError("At least one session is required for stability")
        return measure_stability([self._session(item) for item in selected])
