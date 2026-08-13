"""Scholarly metadata sources; web search is supplemental, not authoritative."""

from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from .contracts import LiteratureQuery, PaperRecord


class LiteratureSourceError(RuntimeError):
    pass


def _first(value: Any) -> Any | None:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _clean_markup(value: str | None) -> str | None:
    if not value:
        return None
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return " ".join(html.unescape(without_tags).split()) or None


class CrossrefSource:
    """Crossref REST metadata search with an injectable transport for tests."""

    name = "crossref"

    def __init__(
        self,
        *,
        mailto: str | None = None,
        timeout_seconds: float = 30.0,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.mailto = mailto
        self.timeout_seconds = timeout_seconds
        self._opener = opener or urllib.request.urlopen

    def search(self, query: LiteratureQuery) -> list[PaperRecord]:
        params: dict[str, Any] = {
            "query.bibliographic": query.text,
            "rows": max(1, min(query.limit_per_source, 100)),
            "select": "DOI,title,abstract,author,published,container-title,URL,type,relation",
        }
        filters: list[str] = []
        if query.year_from:
            filters.append(f"from-pub-date:{query.year_from}-01-01")
        if query.year_to:
            filters.append(f"until-pub-date:{query.year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        if self.mailto:
            params["mailto"] = self.mailto
        url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "bci-autodiscovery/0.1 (mailto: optional)"},
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise LiteratureSourceError(f"Crossref search failed: {exc}") from exc
        try:
            items = payload["message"]["items"]
        except (KeyError, TypeError) as exc:
            raise LiteratureSourceError("Crossref response has no message.items") from exc
        records: list[PaperRecord] = []
        for index, item in enumerate(items):
            title = _clean_markup(_first(item.get("title")))
            if not title:
                continue
            date_parts = ((item.get("published") or {}).get("date-parts") or [[]])[0]
            year = int(date_parts[0]) if date_parts else None
            authors = tuple(
                " ".join(
                    part for part in (author.get("given", ""), author.get("family", "")) if part
                )
                for author in item.get("author") or []
            )
            authors = tuple(author for author in authors if author)
            doi = item.get("DOI")
            records.append(
                PaperRecord(
                    source=self.name,
                    source_id=str(doi or item.get("URL") or index),
                    doi=str(doi) if doi else None,
                    title=title,
                    year=year,
                    authors=authors,
                    venue=_clean_markup(_first(item.get("container-title"))),
                    url=item.get("URL"),
                    abstract=_clean_markup(item.get("abstract")),
                    work_type=item.get("type"),
                    is_retracted=(
                        True
                        if "is-retracted-by" in (item.get("relation") or {})
                        else None
                    ),
                    raw_metadata=item,
                )
            )
        return records


def _openalex_abstract(inverted: Any) -> str | None:
    if not isinstance(inverted, dict):
        return None
    positioned: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int):
                positioned.append((position, str(word)))
    if not positioned:
        return None
    positioned.sort()
    return " ".join(word for _, word in positioned)


class OpenAlexSource:
    """OpenAlex public works search with an injectable transport for tests."""

    name = "openalex"

    def __init__(
        self,
        *,
        mailto: str | None = None,
        timeout_seconds: float = 30.0,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.mailto = mailto
        self.timeout_seconds = timeout_seconds
        self._opener = opener or urllib.request.urlopen

    def search(self, query: LiteratureQuery) -> list[PaperRecord]:
        params: dict[str, Any] = {
            "search": query.text,
            "per-page": max(1, min(query.limit_per_source, 100)),
        }
        filters: list[str] = []
        if query.year_from:
            filters.append(f"from_publication_date:{query.year_from}-01-01")
        if query.year_to:
            filters.append(f"to_publication_date:{query.year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        if self.mailto:
            params["mailto"] = self.mailto
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "bci-autodiscovery/0.1 (mailto: optional)"},
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise LiteratureSourceError(f"OpenAlex search failed: {exc}") from exc
        results = payload.get("results")
        if not isinstance(results, list):
            raise LiteratureSourceError("OpenAlex response has no results array")
        records: list[PaperRecord] = []
        for index, item in enumerate(results):
            title = _clean_markup(item.get("display_name") or item.get("title"))
            if not title:
                continue
            raw_doi = item.get("doi")
            doi = None
            if raw_doi:
                doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", str(raw_doi), flags=re.I)
            authors = tuple(
                str((entry.get("author") or {}).get("display_name") or "").strip()
                for entry in item.get("authorships") or []
            )
            authors = tuple(author for author in authors if author)
            source = ((item.get("primary_location") or {}).get("source") or {})
            records.append(
                PaperRecord(
                    source=self.name,
                    source_id=str(item.get("id") or index),
                    doi=doi,
                    title=title,
                    year=(
                        int(item["publication_year"])
                        if item.get("publication_year") is not None
                        else None
                    ),
                    authors=authors,
                    venue=_clean_markup(source.get("display_name")),
                    url=(item.get("primary_location") or {}).get("landing_page_url")
                    or item.get("id"),
                    abstract=_openalex_abstract(item.get("abstract_inverted_index")),
                    work_type=item.get("type"),
                    citation_count=(
                        int(item["cited_by_count"])
                        if item.get("cited_by_count") is not None
                        else None
                    ),
                    is_retracted=(
                        bool(item["is_retracted"])
                        if item.get("is_retracted") is not None
                        else None
                    ),
                    raw_metadata=item,
                )
            )
        return records
