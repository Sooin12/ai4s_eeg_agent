from __future__ import annotations

import io
import json
from pathlib import Path

from bci_autodiscovery.literature import (
    CrossrefSource,
    DirectionCandidate,
    LiteratureEvidenceError,
    LiteratureQuery,
    LiteratureStore,
    OpenAlexSource,
)
import pytest


class _Response:
    def __init__(self, value: object) -> None:
        self._buffer = io.BytesIO(json.dumps(value).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self._buffer.read()


def test_crossref_records_can_be_persisted_with_search_provenance(tmp_path: Path) -> None:
    payload = {
        "message": {
            "items": [
                {
                    "DOI": "10.1234/example",
                    "title": ["<jats:title>Motor imagery pipeline</jats:title>"],
                    "abstract": "<jats:p>Cross-session evidence.</jats:p>",
                    "author": [{"given": "Ada", "family": "Researcher"}],
                    "published": {"date-parts": [[2026, 1, 2]]},
                    "container-title": ["BCI Journal"],
                    "URL": "https://doi.org/10.1234/example",
                    "type": "journal-article",
                    "relation": {},
                }
            ]
        }
    }
    source = CrossrefSource(opener=lambda *_args, **_kwargs: _Response(payload))
    query = LiteratureQuery(
        query_id="q1",
        text="motor imagery individualized pipeline",
        rationale="test",
        limit_per_source=5,
    )
    papers = source.search(query)

    assert papers[0].doi == "10.1234/example"
    assert papers[0].title == "Motor imagery pipeline"
    assert papers[0].abstract == "Cross-session evidence."

    store = LiteratureStore(tmp_path / "evidence.sqlite")
    store.record_search(
        search_run_id="literature-test",
        query=query,
        source=source.name,
        papers=papers,
    )
    assert store.summary() == {
        "searches": 1,
        "papers": 1,
        "search_results": 1,
        "direction_candidates": 0,
        "dataset_direction_candidates": 0,
    }

    candidate = DirectionCandidate(
        candidate_id="session-adaptation",
        method_family="test_time_adaptation",
        pipeline_stages=("session_adaptation", "models"),
        claim="May reduce session shift and should be tested under target-access constraints.",
        applicability=("repeated sessions",),
        limitations=("abstract-only evidence",),
        supporting_papers=("10.1234/example",),
        novelty_level="frontier_for_registry",
        proposed_validation="Fit on search sessions and lock before frozen confirmation.",
    )
    store.record_direction_candidates(
        search_run_id="literature-test", candidates=[candidate]
    )
    assert store.list_direction_candidates(search_run_id="literature-test")[0][
        "status"
    ] == "proposed_requires_review"


def test_direction_candidate_must_cite_a_paper_from_the_same_run(tmp_path: Path) -> None:
    store = LiteratureStore(tmp_path / "evidence.sqlite")
    candidate = DirectionCandidate(
        candidate_id="unsupported",
        method_family="unknown",
        pipeline_stages=("models",),
        claim="Unsupported claim",
        applicability=("EEG",),
        limitations=("no evidence",),
        supporting_papers=("10.9999/not-stored",),
        novelty_level="unknown",
        proposed_validation="Not yet defined",
    )
    with pytest.raises(LiteratureEvidenceError, match="unstored paper IDs"):
        store.record_direction_candidates(search_run_id="run", candidates=[candidate])


def test_openalex_records_are_normalized_to_provider_neutral_papers() -> None:
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "doi": "https://doi.org/10.1234/openalex",
                "display_name": "Cross-session EEG adaptation",
                "publication_year": 2025,
                "authorships": [{"author": {"display_name": "Ada Researcher"}}],
                "primary_location": {
                    "landing_page_url": "https://example.org/work",
                    "source": {"display_name": "BCI Journal"},
                },
                "abstract_inverted_index": {
                    "Session": [0],
                    "adaptation": [1],
                    "helps": [2],
                },
                "type": "article",
                "cited_by_count": 7,
                "is_retracted": False,
            }
        ]
    }
    source = OpenAlexSource(opener=lambda *_args, **_kwargs: _Response(payload))
    query = LiteratureQuery(query_id="oa", text="EEG adaptation", rationale="test")

    papers = source.search(query)

    assert papers[0].stable_id == "10.1234/openalex"
    assert papers[0].abstract == "Session adaptation helps"
    assert papers[0].citation_count == 7
    assert papers[0].is_retracted is False
