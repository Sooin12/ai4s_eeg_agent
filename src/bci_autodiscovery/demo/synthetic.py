"""Deterministic heterogeneous EEG-like fixtures for pipeline personalization demos."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bci_autodiscovery.profiling.subject_measurements import EpochSession


CHANNEL_NAMES = ("F3", "C3", "Cz", "C4", "P3", "P4", "O1", "O2")
SUBJECT_PHENOTYPES = {
    "subject-mu": "mu_power",
    "subject-csp": "beta_covariance",
    "subject-beta": "individual_beta_power",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generate_session(*, subject_id: str, phenotype: str, session_id: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    sfreq = 128.0
    sample_count = 128
    labels = np.repeat(np.asarray([0, 1], dtype=np.int64), 32)
    rng.shuffle(labels)
    time = np.arange(sample_count) / sfreq
    scale = {"profile": 1.0, "search": 0.95, "confirm": 0.88}[session_id]
    data = rng.normal(scale=0.72, size=(labels.size, len(CHANNEL_NAMES), sample_count))

    if phenotype == "mu_power":
        rhythm = np.sin(2 * np.pi * 10.0 * time)
        for trial, label in enumerate(labels):
            channel = 1 if label == 0 else 3
            data[trial, channel] += scale * 1.55 * rhythm
    elif phenotype == "beta_covariance":
        for trial, label in enumerate(labels):
            phase = rng.uniform(0, 2 * np.pi)
            rhythm = np.sin(2 * np.pi * 20.0 * time + phase)
            data[trial, 1] += scale * 1.35 * rhythm
            data[trial, 3] += scale * (1.35 if label == 0 else -1.35) * rhythm
    elif phenotype == "individual_beta_power":
        rhythm = np.sin(2 * np.pi * 17.0 * time)
        distractor = np.sin(2 * np.pi * 26.0 * time)
        for trial, label in enumerate(labels):
            data[trial, 4 if label == 0 else 5] += scale * 1.45 * rhythm
            data[trial, 0] += rng.normal(0.0, 0.8) * distractor
            data[trial, 7] += rng.normal(0.0, 0.8) * distractor
    else:
        raise ValueError(f"Unknown synthetic phenotype: {phenotype}")
    return data.astype(np.float32), labels


def write_synthetic_sessions(root: Path) -> dict[str, dict[str, Path]]:
    root = Path(root).resolve()
    result: dict[str, dict[str, Path]] = {}
    for subject_index, (subject_id, phenotype) in enumerate(SUBJECT_PHENOTYPES.items()):
        result[subject_id] = {}
        for session_index, session_id in enumerate(("profile", "search", "confirm")):
            path = root / subject_id / f"{session_id}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            data, labels = _generate_session(
                subject_id=subject_id,
                phenotype=phenotype,
                session_id=session_id,
                seed=9100 + subject_index * 100 + session_index,
            )
            np.savez_compressed(
                path,
                data=data,
                labels=labels,
                sampling_frequency_hz=np.asarray(128.0),
                channel_names=np.asarray(CHANNEL_NAMES),
            )
            result[subject_id][session_id] = path
    return result


@dataclass(frozen=True)
class DemoNpzEpochSource:
    root: Path
    source_id: str = "synthetic-demo-npz-v1"

    def load_session(self, *, subject_id: str, session_id: str) -> EpochSession:
        path = (Path(self.root) / subject_id / f"{session_id}.npz").resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Synthetic session is unavailable: {path}")
        with np.load(path, allow_pickle=False) as payload:
            data = np.asarray(payload["data"])
            labels = np.asarray(payload["labels"])
            sfreq = float(payload["sampling_frequency_hz"])
            channel_names = tuple(str(item) for item in payload["channel_names"].tolist())
        session = EpochSession(
            subject_id=subject_id,
            session_id=session_id,
            data=data,
            labels=labels,
            sampling_frequency_hz=sfreq,
            channel_names=channel_names,
            provenance={
                "path": str(path),
                "sha256": _sha256(path),
                "source_id": self.source_id,
                "engineering_fixture": True,
                "scientific_claim_authorized": False,
            },
        )
        session.validate()
        return session
