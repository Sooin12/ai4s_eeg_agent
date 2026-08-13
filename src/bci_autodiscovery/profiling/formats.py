"""Read-only recognition of common EEG dataset containers and layouts.

Recognition is deliberately weaker than semantic adaptation: a suffix can identify a
container, but it cannot establish array axes, event meaning, or experimental roles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FormatDefinition:
    format_id: str
    display_name: str
    extensions: tuple[str, ...]
    interpretation_level: str
    official_reference: str
    companion_extensions: tuple[str, ...] = ()
    companion_files_required: bool = True
    directory_suffixes: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_id": self.format_id,
            "display_name": self.display_name,
            "extensions": list(self.extensions),
            "companion_extensions": list(self.companion_extensions),
            "companion_files_required": self.companion_files_required,
            "directory_suffixes": list(self.directory_suffixes),
            "interpretation_level": self.interpretation_level,
            "official_reference": self.official_reference,
            "note": self.note,
        }


MAINSTREAM_EEG_FORMATS: tuple[FormatDefinition, ...] = (
    FormatDefinition(
        "brainvision_core",
        "BrainVision Core Data Format",
        (".vhdr",),
        "container_recognized_requires_experiment_semantics",
        "https://www.brainproducts.com/support-resources/brainvision-core-data-format-1-0/",
        companion_extensions=(".vmrk", ".eeg"),
        note="Header, marker and binary signal files must share a basename.",
    ),
    FormatDefinition(
        "brainvision_recording",
        "BrainVision Recording Format",
        (".bvrh",),
        "container_recognized_requires_experiment_semantics",
        "https://www.brainproducts.com/support-resources/brainvision-recording-format/",
        companion_extensions=(".bvrm", ".bvrd"),
        note="JSON header, TSV marker and binary signal files must share a basename.",
    ),
    FormatDefinition(
        "edf_edfplus",
        "European Data Format / EDF+",
        (".edf",),
        "container_recognized_requires_experiment_semantics",
        "https://www.edfplus.info/specs/index.html",
    ),
    FormatDefinition(
        "biosemi_bdf",
        "BioSemi Data Format / BDF+",
        (".bdf",),
        "container_recognized_requires_experiment_semantics",
        "https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html",
    ),
    FormatDefinition(
        "eeglab_set",
        "EEGLAB SET",
        (".set",),
        "container_recognized_requires_experiment_semantics",
        "https://eeglab.org/tutorials/ConceptsGuide/Data_Structures.html",
        companion_extensions=(".fdt",),
        companion_files_required=False,
        note="The FDT companion is optional because SET may store data internally.",
    ),
    FormatDefinition(
        "mne_fif",
        "MNE FIF",
        (".fif", ".fif.gz"),
        "container_recognized_requires_experiment_semantics",
        "https://mne.tools/stable/documentation/implementation.html#supported-data-formats",
    ),
    FormatDefinition(
        "biosig_gdf",
        "General Data Format",
        (".gdf",),
        "container_recognized_requires_experiment_semantics",
        "https://mne.tools/stable/documentation/implementation.html#supported-data-formats",
    ),
    FormatDefinition(
        "neuroscan_cnt",
        "Neuroscan CNT",
        (".cnt",),
        "container_recognized_requires_experiment_semantics",
        "https://mne.tools/stable/documentation/implementation.html#supported-data-formats",
    ),
    FormatDefinition(
        "egi_mff",
        "EGI MFF",
        (),
        "container_recognized_requires_experiment_semantics",
        "https://mne.tools/stable/auto_tutorials/io/20_reading_eeg_data.html",
        directory_suffixes=(".mff",),
    ),
    FormatDefinition(
        "xdf",
        "Extensible Data Format",
        (".xdf", ".xdfz"),
        "container_recognized_requires_stream_mapping",
        "https://github.com/sccn/xdf/wiki/Specifications",
        note="Stream identity, clocks, markers and EEG selection require an explicit mapping.",
    ),
    FormatDefinition(
        "nwb",
        "Neurodata Without Borders",
        (".nwb",),
        "container_recognized_requires_series_mapping",
        "https://nwb-schema.readthedocs.io/en/stable/format.html",
        note="The ElectricalSeries and behavioral/event semantics must be selected explicitly.",
    ),
    FormatDefinition(
        "mat_container",
        "MATLAB / FieldTrip-compatible container",
        (".mat",),
        "generic_container_requires_semantic_sidecar",
        "https://mne.tools/stable/documentation/implementation.html#supported-data-formats",
        note="A MAT suffix never establishes FieldTrip conformance, array keys or axes.",
    ),
    FormatDefinition(
        "hdf5_container",
        "HDF5 container",
        (".h5", ".hdf5", ".hdf"),
        "generic_container_requires_semantic_sidecar",
        "https://www.hdfgroup.org/solutions/hdf5/",
        note="Dataset paths, axes, units, events and roles require a semantic sidecar.",
    ),
    FormatDefinition(
        "numpy_container",
        "NumPy array container",
        (".npy", ".npz"),
        "generic_container_requires_semantic_sidecar",
        "https://numpy.org/doc/stable/reference/routines.io.html",
        note="Array keys, axes, units, labels and split roles require a semantic sidecar.",
    ),
)


class DatasetFormatCatalog:
    """Inventory file signatures without opening large signal arrays."""

    def __init__(self, definitions: tuple[FormatDefinition, ...] = MAINSTREAM_EEG_FORMATS):
        self.definitions = definitions

    def inspect(self, dataset_root: Path) -> list[dict[str, Any]]:
        root = Path(dataset_root).expanduser().resolve()
        if not root.is_dir():
            return []
        files = [path for path in root.rglob("*") if path.is_file()]
        directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
        candidates: list[dict[str, Any]] = []
        for definition in self.definitions:
            matches = [
                path for path in files if _matches_extension(path.name, definition.extensions)
            ]
            directory_matches = [
                path
                for path in directories
                if path.name.lower().endswith(definition.directory_suffixes)
            ]
            if not matches and not directory_matches:
                continue
            companion_complete: bool | None = None
            limitations: list[str] = []
            if definition.companion_extensions and matches:
                complete = 0
                for primary in matches:
                    if all(primary.with_suffix(ext).is_file() for ext in definition.companion_extensions):
                        complete += 1
                all_present = complete == len(matches)
                companion_complete = all_present or not definition.companion_files_required
                if not all_present and definition.companion_files_required:
                    limitations.append("one_or_more_companion_sets_are_incomplete")
                elif not all_present:
                    limitations.append("optional_companion_files_are_absent")
            limitations.append("format_detection_does_not_establish_experiment_semantics")
            candidates.append(
                {
                    **definition.to_dict(),
                    "matched_file_count": len(matches),
                    "matched_directory_count": len(directory_matches),
                    "sample_paths": [
                        str(path.relative_to(root))
                        for path in (matches + directory_matches)[:5]
                    ],
                    "companion_complete": companion_complete,
                    "limitations": limitations,
                }
            )
        return sorted(
            candidates,
            key=lambda item: (
                item["interpretation_level"].startswith("generic_container"),
                item["format_id"],
            ),
        )


def _matches_extension(name: str, extensions: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(extension) for extension in extensions)
