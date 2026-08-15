"""Budget-aware search and selection strategies."""
"""Dataset-constrained, budget-aware pipeline search."""

from .builder import SearchSpaceBuildError, build_search_space_draft
from .components import ComponentRegistry, ComponentRegistryError
from .dataset_incumbent import (
    DatasetIncumbentError,
    build_dataset_incumbent,
    expected_selective_route,
    validate_dataset_incumbent,
)
from .frontier import (
    FrontierMergeError,
    build_combined_search_space_review,
    validate_dataset_level_review_draft,
)

__all__ = [
    "ComponentRegistry",
    "ComponentRegistryError",
    "DatasetIncumbentError",
    "FrontierMergeError",
    "SearchSpaceBuildError",
    "build_search_space_draft",
    "build_combined_search_space_review",
    "build_dataset_incumbent",
    "expected_selective_route",
    "validate_dataset_incumbent",
    "validate_dataset_level_review_draft",
]
