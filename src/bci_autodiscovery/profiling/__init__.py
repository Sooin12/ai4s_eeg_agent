"""Subject profiling primitives."""
"""Dataset- and subject-level deterministic profiling tools."""

from .dataset import (
    DATASET_PROFILE_BINDING_ROOTS,
    DatasetProfileError,
    dataset_profile_field_catalog,
    profile_bids_eeg_dataset,
    validate_dataset_profile,
    validate_dataset_profile_provenance,
)
from .formats import DatasetFormatCatalog, FormatDefinition, MAINSTREAM_EEG_FORMATS
from .semantic_validation import (
    SEMANTICS_FILENAME,
    load_semantics,
    profile_from_semantic_validation,
    validate_semantic_dataset,
    verify_semantic_validation,
)
from .adapters import (
    AdapterProbe,
    DatasetAdapterRegistry,
    create_default_adapter_registry,
)
from .subject_measurements import (
    EpochSession,
    SubjectEpochSource,
    SubjectMeasurementEngine,
    SubjectMeasurementError,
)
from .subject_adapters import MatEpochSource

__all__ = [
    "DatasetProfileError",
    "DATASET_PROFILE_BINDING_ROOTS",
    "dataset_profile_field_catalog",
    "AdapterProbe",
    "DatasetAdapterRegistry",
    "create_default_adapter_registry",
    "DatasetFormatCatalog",
    "FormatDefinition",
    "MAINSTREAM_EEG_FORMATS",
    "SEMANTICS_FILENAME",
    "load_semantics",
    "profile_from_semantic_validation",
    "validate_semantic_dataset",
    "verify_semantic_validation",
    "EpochSession",
    "SubjectEpochSource",
    "SubjectMeasurementEngine",
    "SubjectMeasurementError",
    "MatEpochSource",
    "profile_bids_eeg_dataset",
    "validate_dataset_profile",
    "validate_dataset_profile_provenance",
]
