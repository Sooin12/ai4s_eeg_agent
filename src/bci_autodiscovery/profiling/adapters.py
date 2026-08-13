"""Pluggable dataset inspection and normalized-profile adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .dataset import (
    DatasetProfileError,
    profile_bids_eeg_dataset,
    validate_dataset_profile_provenance,
)
from .formats import DatasetFormatCatalog
from .semantic_validation import (
    SEMANTICS_FILENAME,
    profile_from_semantic_validation,
    verify_semantic_validation,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class AdapterProbe:
    adapter_id: str
    confidence: float
    reasons: tuple[str, ...]
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "limitations": list(self.limitations),
        }


class DatasetAdapter(Protocol):
    adapter_id: str

    def probe(self, *, dataset_root: Path, validation_path: Path) -> AdapterProbe: ...

    def profile(
        self,
        *,
        dataset_root: Path,
        validation_path: Path,
        dataset_id_hint: str | None,
    ) -> dict[str, Any]: ...


class DatasetAdapterRegistry:
    def __init__(self, *, format_catalog: DatasetFormatCatalog | None = None) -> None:
        self._adapters: dict[str, DatasetAdapter] = {}
        self._format_catalog = format_catalog or DatasetFormatCatalog()

    def register(self, adapter: DatasetAdapter) -> None:
        if adapter.adapter_id in self._adapters:
            raise ValueError(f"Duplicate dataset adapter: {adapter.adapter_id}")
        self._adapters[adapter.adapter_id] = adapter

    def inspect(self, *, dataset_root: Path, validation_path: Path) -> dict[str, Any]:
        root = Path(dataset_root).expanduser().resolve()
        validation = Path(validation_path).expanduser().resolve()
        probes = [
            adapter.probe(dataset_root=root, validation_path=validation)
            for adapter in self._adapters.values()
        ]
        ranked = sorted(probes, key=lambda item: (-item.confidence, item.adapter_id))
        viable = [item for item in ranked if item.confidence > 0]
        selected = viable[0].adapter_id if viable else None
        ambiguous = (
            len(viable) > 1
            and abs(viable[0].confidence - viable[1].confidence) < 0.05
        )
        format_candidates = self._format_catalog.inspect(root)
        if ambiguous:
            status = "adapter_ambiguous_requires_review"
        elif selected:
            status = "adapter_selected"
        elif format_candidates:
            status = "recognized_format_requires_semantic_mapping"
        else:
            status = "unsupported_requires_new_adapter"
        return {
            "schema_version": "2.0",
            "dataset_root": str(root),
            "validation_path": str(validation),
            "selected_adapter_id": None if ambiguous else selected,
            "status": status,
            "candidates": [item.to_dict() for item in ranked],
            "format_candidates": format_candidates,
            "semantic_gate": {
                "profile_ready": status == "adapter_selected",
                "recognition_is_not_semantic_acceptance": True,
                "required_when_not_ready": (
                    "validated_semantic_adapter_or_mapping_sidecar"
                    if format_candidates
                    else "new_format_and_semantic_adapter"
                ),
                "forbidden_inferences": [
                    "array_axes_from_file_extension",
                    "event_meaning_from_numeric_codes_without_evidence",
                    "search_or_confirmation_roles_from_directory_order",
                ],
            },
            "raw_signal_transmitted": False,
        }

    def profile(
        self,
        *,
        adapter_id: str,
        dataset_root: Path,
        validation_path: Path,
        dataset_id_hint: str | None = None,
    ) -> dict[str, Any]:
        try:
            adapter = self._adapters[adapter_id]
        except KeyError as exc:
            raise DatasetProfileError(f"Unknown dataset adapter: {adapter_id}") from exc
        profile = adapter.profile(
            dataset_root=Path(dataset_root).expanduser().resolve(),
            validation_path=Path(validation_path).expanduser().resolve(),
            dataset_id_hint=dataset_id_hint,
        )
        profile["dataset"]["adapter_id"] = adapter_id
        validate_dataset_profile_provenance(
            profile,
            require_hashed_evidence=True,
            require_current_constraints=True,
        )
        return profile

    @property
    def adapter_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


class BidsEegAdapter:
    """Generic BIDS EEG organization adapter."""

    adapter_id = "bids_eeg_v1"

    def probe(self, *, dataset_root: Path, validation_path: Path) -> AdapterProbe:
        reasons: list[str] = []
        def has_eeg_file(pattern: str) -> bool:
            return any("eeg" in path.parts for path in dataset_root.rglob(pattern))

        if (dataset_root / "dataset_description.json").is_file():
            reasons.append("dataset_description.json present")
        if has_eeg_file("*_events.tsv"):
            reasons.append("BIDS EEG event sidecars present")
        if has_eeg_file("*_eeg.json"):
            reasons.append("BIDS EEG acquisition sidecars present")
        if validation_path.is_file():
            reasons.append("validation artifact present")
        confidence = 0.90 if len(reasons) == 4 else 0.0
        return AdapterProbe(
            adapter_id=self.adapter_id,
            confidence=confidence,
            reasons=tuple(reasons),
            limitations=(
                "Task semantics are derived only from BIDS sidecars and event labels; missing semantics remain explicit.",
            ),
        )

    def profile(
        self,
        *,
        dataset_root: Path,
        validation_path: Path,
        dataset_id_hint: str | None,
    ) -> dict[str, Any]:
        description = json.loads(
            (dataset_root / "dataset_description.json").read_text(encoding="utf-8-sig")
        )
        inferred = str(description.get("Name") or "bids_eeg_dataset").lower()
        inferred = "_".join(part for part in inferred.replace("-", " ").split() if part)
        return profile_bids_eeg_dataset(
            dataset_root=str(dataset_root),
            validation_path=str(validation_path),
            dataset_id=(
                dataset_id_hint
                if dataset_id_hint and dataset_id_hint != "auto"
                else inferred
            ),
        )


class MneRawSemanticAdapter:
    """Non-BIDS raw formats read locally by MNE and governed by explicit semantics."""

    adapter_id = "mne_raw_semantic_v1"

    def probe(self, *, dataset_root: Path, validation_path: Path) -> AdapterProbe:
        semantics_path = dataset_root / SEMANTICS_FILENAME
        reasons: list[str] = []
        try:
            semantics, validation = verify_semantic_validation(
                dataset_root=dataset_root,
                semantics_path=semantics_path,
                validation_path=validation_path,
            )
        except DatasetProfileError:
            return AdapterProbe(self.adapter_id, 0.0, ())
        if semantics.get("adapter_kind") == "mne_raw":
            reasons.append("validated mne_raw semantic sidecar present")
        if validation.get("checks", {}).get("mne_header_and_event_match") is True:
            reasons.append("MNE headers and annotations match declared profile")
        return AdapterProbe(
            self.adapter_id,
            0.96 if len(reasons) == 2 else 0.0,
            tuple(reasons),
            ("MNE reads headers and annotations locally; raw signal arrays are not sent to the LLM.",),
        )

    def profile(
        self,
        *,
        dataset_root: Path,
        validation_path: Path,
        dataset_id_hint: str | None,
    ) -> dict[str, Any]:
        semantics, _validation = verify_semantic_validation(
            dataset_root=dataset_root,
            semantics_path=dataset_root / SEMANTICS_FILENAME,
            validation_path=validation_path,
        )
        if semantics["adapter_kind"] != "mne_raw":
            raise DatasetProfileError("Semantic sidecar is not for mne_raw adapter")
        return profile_from_semantic_validation(
            semantics=semantics,
            validation_path=validation_path,
            dataset_id_hint=dataset_id_hint,
        )


class DeclarativeSemanticAdapter:
    """Fail-closed bridge for flexible containers with an explicit full profile."""

    adapter_id = "declarative_semantic_v1"

    def probe(self, *, dataset_root: Path, validation_path: Path) -> AdapterProbe:
        try:
            semantics, validation = verify_semantic_validation(
                dataset_root=dataset_root,
                semantics_path=dataset_root / SEMANTICS_FILENAME,
                validation_path=validation_path,
            )
        except DatasetProfileError:
            return AdapterProbe(self.adapter_id, 0.0, ())
        if semantics.get("adapter_kind") != "declarative":
            return AdapterProbe(self.adapter_id, 0.0, ())
        checks = validation.get("checks") or {}
        confidence = 0.85 if checks.get("all_sources_hash_bound") is True else 0.0
        return AdapterProbe(
            self.adapter_id,
            confidence,
            ("validated declarative semantic profile and hash-bound sources present",),
            (
                "Flexible container internals are accepted from the explicit profile contract; "
                "this adapter does not infer axes, streams, series, events or roles.",
            ),
        )

    def profile(
        self,
        *,
        dataset_root: Path,
        validation_path: Path,
        dataset_id_hint: str | None,
    ) -> dict[str, Any]:
        semantics, _validation = verify_semantic_validation(
            dataset_root=dataset_root,
            semantics_path=dataset_root / SEMANTICS_FILENAME,
            validation_path=validation_path,
        )
        if semantics["adapter_kind"] != "declarative":
            raise DatasetProfileError("Semantic sidecar is not for declarative adapter")
        return profile_from_semantic_validation(
            semantics=semantics,
            validation_path=validation_path,
            dataset_id_hint=dataset_id_hint,
        )


class MatEpochSemanticAdapter:
    """Generic deterministic reader for epoched MATLAB arrays."""

    adapter_id = "mat_epoch_semantic_v1"

    def probe(self, *, dataset_root: Path, validation_path: Path) -> AdapterProbe:
        try:
            semantics, validation = verify_semantic_validation(
                dataset_root=dataset_root,
                semantics_path=dataset_root / SEMANTICS_FILENAME,
                validation_path=validation_path,
            )
        except DatasetProfileError:
            return AdapterProbe(self.adapter_id, 0.0, ())
        checks = validation.get("checks") or {}
        ready = (
            semantics.get("adapter_kind") == "mat_epochs"
            and checks.get("mat_arrays_and_events_match") is True
        )
        return AdapterProbe(
            self.adapter_id,
            0.97 if ready else 0.0,
            ("MAT arrays, labels, identities and source hashes validated locally",) if ready else (),
            ("Experimental semantics come only from the explicit sidecar; unknowns remain unknown.",),
        )

    def profile(
        self,
        *,
        dataset_root: Path,
        validation_path: Path,
        dataset_id_hint: str | None,
    ) -> dict[str, Any]:
        semantics, validation = verify_semantic_validation(
            dataset_root=dataset_root,
            semantics_path=dataset_root / SEMANTICS_FILENAME,
            validation_path=validation_path,
        )
        if semantics["adapter_kind"] != "mat_epochs":
            raise DatasetProfileError("Semantic sidecar is not for MAT epoch adapter")
        return profile_from_semantic_validation(
            semantics=semantics,
            validation_path=validation_path,
            dataset_id_hint=dataset_id_hint,
            validation=validation,
        )


def create_default_adapter_registry() -> DatasetAdapterRegistry:
    registry = DatasetAdapterRegistry()
    registry.register(BidsEegAdapter())
    registry.register(MneRawSemanticAdapter())
    registry.register(DeclarativeSemanticAdapter())
    registry.register(MatEpochSemanticAdapter())
    return registry
