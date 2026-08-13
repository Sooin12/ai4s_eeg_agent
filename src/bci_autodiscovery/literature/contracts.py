"""Provider-neutral records for literature discovery and evidence extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LiteratureQuery:
    query_id: str
    text: str
    rationale: str
    source_names: tuple[str, ...] = ("crossref", "openalex")
    year_from: int | None = None
    year_to: int | None = None
    limit_per_source: int = 20

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_names"] = list(self.source_names)
        return result


@dataclass(frozen=True)
class PaperRecord:
    source: str
    source_id: str
    title: str
    year: int | None = None
    doi: str | None = None
    authors: tuple[str, ...] = ()
    venue: str | None = None
    url: str | None = None
    abstract: str | None = None
    work_type: str | None = None
    citation_count: int | None = None
    is_retracted: bool | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def stable_id(self) -> str:
        return self.doi.lower() if self.doi else f"{self.source}:{self.source_id}"

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        result = asdict(self)
        result["authors"] = list(self.authors)
        if not include_raw:
            result.pop("raw_metadata", None)
        return result


@dataclass(frozen=True)
class DirectionCandidate:
    """Evidence-linked research direction; proposal is not executable activation."""

    candidate_id: str
    method_family: str
    pipeline_stages: tuple[str, ...]
    claim: str
    applicability: tuple[str, ...]
    limitations: tuple[str, ...]
    supporting_papers: tuple[str, ...]
    novelty_level: str
    proposed_validation: str
    status: str = "proposed_requires_review"
    protocol_binding: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "pipeline_stages",
            "applicability",
            "limitations",
            "supporting_papers",
        ):
            result[key] = list(result[key])
        return result


@dataclass(frozen=True)
class DatasetDirectionCandidate:
    """Dataset-bound frontier hypothesis; never an executable pipeline activation."""

    candidate_id: str
    method_family: str
    pipeline_stages: tuple[str, ...]
    claim: str
    applicability: tuple[str, ...]
    limitations: tuple[str, ...]
    supporting_papers: tuple[str, ...]
    novelty_level: str
    future_protocol_requirements: tuple[str, ...]
    proposed_validation: str
    dataset_binding: dict[str, Any]
    evidence_scope: str = "scholarly_metadata_or_abstract_discovery_only"
    status: str = "dataset_frontier_hypothesis"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "pipeline_stages",
            "applicability",
            "limitations",
            "supporting_papers",
            "future_protocol_requirements",
        ):
            result[key] = list(result[key])
        return result
