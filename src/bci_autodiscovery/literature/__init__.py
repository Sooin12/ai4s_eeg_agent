"""Auditable scholarly discovery and local evidence storage."""

from .contracts import DatasetDirectionCandidate, DirectionCandidate, LiteratureQuery, PaperRecord
from .sources import CrossrefSource, LiteratureSourceError, OpenAlexSource
from .store import LiteratureEvidenceError, LiteratureStore

__all__ = [
    "CrossrefSource",
    "DatasetDirectionCandidate",
    "DirectionCandidate",
    "LiteratureEvidenceError",
    "LiteratureQuery",
    "LiteratureSourceError",
    "LiteratureStore",
    "OpenAlexSource",
    "PaperRecord",
]
