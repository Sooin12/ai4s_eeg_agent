"""Dataset-neutral sources for deterministic subject-level measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from .dataset import DatasetProfileError
from .semantic_validation import SEMANTICS_FILENAME, verify_semantic_validation
from .subject_measurements import EpochSession, SubjectMeasurementError


class MatEpochSource:
    """Expose validated MATLAB epoch arrays through the standard subject contract."""

    source_id = "mat_epoch_subject_source_v1"

    def __init__(self, *, dataset_root: Path, validation_path: Path) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.validation_path = Path(validation_path).expanduser().resolve()
        try:
            semantics, validation = verify_semantic_validation(
                dataset_root=self.dataset_root,
                semantics_path=self.dataset_root / SEMANTICS_FILENAME,
                validation_path=self.validation_path,
            )
        except DatasetProfileError as exc:
            raise SubjectMeasurementError(f"MAT epoch source is not validated: {exc}") from exc
        if semantics.get("adapter_kind") != "mat_epochs":
            raise SubjectMeasurementError("Semantic contract is not a MAT epoch contract")
        self.mapping = semantics["mat_mapping"]
        self.records = list((validation.get("observed") or {}).get("records") or [])
        conflict_count = sum(
            not bool(record.get("identity_consistent", True)) for record in self.records
        )
        if conflict_count:
            raise SubjectMeasurementError(
                "MAT epoch source has conflicting identity declarations in "
                f"{conflict_count} validated source(s); subject-level loading is blocked"
            )
        self.channel_names = tuple(str(item) for item in self.mapping["channel_names"])
        self.sampling_frequency_hz = float(self.mapping["sampling_frequency_hz"])

    def load_session(self, *, subject_id: str, session_id: str) -> EpochSession:
        matches = [
            item
            for item in self.records
            if str(item["subject_id"]) == str(subject_id)
            and str(item["session_id"]) == str(session_id)
        ]
        if not matches:
            raise SubjectMeasurementError(
                f"No validated MAT epochs for subject={subject_id}, session={session_id}"
            )
        epochs: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        provenance_sources: list[dict[str, Any]] = []
        axes = list(self.mapping["axis_order"])
        source_axes = (axes.index("trial"), axes.index("channel"), axes.index("sample"))
        for record in sorted(matches, key=lambda item: str(item.get("run_id", "default"))):
            path = (self.dataset_root / str(record["relative_path"])).resolve()
            payload = loadmat(
                path,
                variable_names=[self.mapping["signal_key"], self.mapping["label_key"]],
            )
            data = np.asarray(payload[self.mapping["signal_key"]])
            target = np.asarray(payload[self.mapping["label_key"]]).reshape(-1)
            standardized = np.moveaxis(data, source_axes, (0, 1, 2))
            if standardized.shape[0] != target.size:
                raise SubjectMeasurementError(f"Validated MAT source changed shape: {path}")
            epochs.append(standardized)
            labels.append(target)
            provenance_sources.append(
                {
                    "relative_path": str(record["relative_path"]),
                    "run_id": str(record.get("run_id", "default")),
                    "shape": list(data.shape),
                }
            )
        return EpochSession(
            subject_id=str(subject_id),
            session_id=str(session_id),
            data=np.concatenate(epochs, axis=0),
            labels=np.concatenate(labels),
            sampling_frequency_hz=self.sampling_frequency_hz,
            channel_names=self.channel_names,
            provenance={
                "source_id": self.source_id,
                "validation_path": str(self.validation_path),
                "sources": provenance_sources,
                "array_axes_standard": ["trial", "channel", "sample"],
            },
        )
