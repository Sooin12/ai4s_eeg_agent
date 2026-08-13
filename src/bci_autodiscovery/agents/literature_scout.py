"""Auditable literature scout for dataset-specific frontier directions."""

from __future__ import annotations

import json
import html
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci_autodiscovery.literature import (
    CrossrefSource,
    DatasetDirectionCandidate,
    LiteratureQuery,
    LiteratureSourceError,
    LiteratureStore,
    OpenAlexSource,
)
from bci_autodiscovery.profiling import (
    DATASET_PROFILE_BINDING_ROOTS,
    DatasetProfileError,
    dataset_profile_field_catalog,
    validate_dataset_profile_provenance,
)

from .contracts import AgentRunResult
from .language import DEFAULT_LANGUAGE_INSTRUCTION
from .runtime import AgentRuntime
from .tools import ToolDefinition, ToolRegistry


LITERATURE_SCOUT_SYSTEM_PROMPT = f"""{DEFAULT_LANGUAGE_INSTRUCTION}
You are the Literature Scout Agent in an auditable BCI research system. Your task is to
look beyond the local component registry and discover research directions matched to the
current dataset. First call inspect_frontier_discovery_status. Call
search_scholarly_metadata for every missing query_id/source_name pair derived from the
dataset profile. Evidence copied into a revision run is an audited local cache and must not
trigger duplicate network requests. Do not record directions until every planned pair has
an audit record and every query has at least one successful source. If a tool call is
rejected because its arguments were malformed, retry only the rejected/missing pair with
one valid JSON object; do not abandon the run or repeat pairs already recorded in the
status ledger. Batch independent missing search tool calls into as few model turns as the
tool interface permits; do not spend a separate reasoning turn on each pair.

Review paper titles, publication types, abstracts, limitations, and dataset compatibility.
Ignore retractions, supplements, corrections, and clearly irrelevant results. A method not
yet implemented locally may be proposed as a research candidate, but distinguish literature
evidence from a hypothesis that still needs testing. Each direction must cite one or more
stable_id values returned by tools, state applicability and limitations, and propose a
leakage-safe validation plan.

Bind every direction to the exact DatasetProfile SHA. For supporting_profile_fields, copy
only exact dotted paths from allowed_supporting_profile_fields in the context/tool schema;
never invent semantic aliases. Cite fields that actually support dataset-level plausibility.
State future protocol requirements explicitly, but do not
assign profiling/search/confirmation roles, metrics, thresholds, budgets, or subject-level
choices. Those belong to downstream agents.

record_frontier_directions may only record dataset_frontier_hypothesis candidates and cannot
activate methods. Finally call inspect_frontier_discovery_status to confirm completion.
If the context contains revision_request, explicitly repair every required revision in the
new direction synthesis; do not reproduce the criticized draft unchanged. When
revision_evidence_policy.search_tool_available is false, the prior audited search coverage
and cited paper IDs are already authoritative for this revision. Do not attempt to reread
every cached query; revise directly from prior_frontier_directions and the Critic request.
Do not claim abstract-level evidence proves effectiveness, and do not access frozen-
confirmation data."""


class LiteratureScoutConfigurationError(ValueError):
    pass


def _load_bound_profile(
    draft: dict[str, Any], *, dataset_id: str
) -> tuple[dict[str, Any], dict[str, str]]:
    ref = (draft.get("provenance") or {}).get("dataset_profile") or {}
    profile_path = Path(str(ref.get("path") or "")).expanduser().resolve()
    expected_hash = str(ref.get("sha256") or "")
    if not profile_path.is_file() or not expected_hash:
        raise LiteratureScoutConfigurationError(
            "Canonical coarse space lacks a readable DatasetProfile provenance binding"
        )
    observed_hash = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    if observed_hash != expected_hash:
        raise LiteratureScoutConfigurationError(
            "Canonical coarse space DatasetProfile provenance failed integrity verification"
        )
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiteratureScoutConfigurationError(
            f"Cannot load bound DatasetProfile: {exc}"
        ) from exc
    if not isinstance(profile, dict):
        raise LiteratureScoutConfigurationError("Bound DatasetProfile must be an object")
    try:
        validate_dataset_profile_provenance(
            profile,
            require_hashed_evidence=True,
            require_current_constraints=True,
        )
    except DatasetProfileError as exc:
        raise LiteratureScoutConfigurationError(
            f"Bound DatasetProfile is not valid for a formal run: {exc}"
        ) from exc
    if profile["dataset"]["id"] != dataset_id:
        raise LiteratureScoutConfigurationError(
            "Canonical coarse space and DatasetProfile use different dataset IDs"
        )
    return profile, {"path": str(profile_path), "sha256": observed_hash}


def _load_query_plan(path: Path) -> tuple[str, dict[str, LiteratureQuery], dict[str, Any]]:
    try:
        draft = json.loads(path.read_text(encoding="utf-8"))
        raw_queries = draft["frontier_discovery"]["query_plan"]
        dataset_id = str(draft["dataset_id"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise LiteratureScoutConfigurationError(
            f"Cannot load frontier query plan from {path}: {exc}"
        ) from exc
    queries: dict[str, LiteratureQuery] = {}
    for item in raw_queries:
        query = LiteratureQuery(
            query_id=str(item["query_id"]),
            text=str(item["text"]),
            rationale=str(item["rationale"]),
            source_names=tuple(str(value) for value in item.get("source_names") or ["crossref"]),
            year_from=item.get("year_from"),
            year_to=item.get("year_to"),
            limit_per_source=int(item.get("limit_per_source", 20)),
        )
        if query.query_id in queries:
            raise LiteratureScoutConfigurationError(
                f"Duplicate frontier query_id: {query.query_id}"
            )
        queries[query.query_id] = query
    if not queries:
        raise LiteratureScoutConfigurationError("Frontier query plan is empty")
    return dataset_id, queries, draft


def _compact_stored_paper(paper: dict[str, Any]) -> dict[str, Any]:
    abstract = str(paper.get("abstract") or "").strip()
    return {
        "stable_id": str(paper["stable_id"]),
        "title": str(paper.get("title") or ""),
        "year": paper.get("year"),
        "venue": paper.get("venue"),
        "work_type": paper.get("work_type"),
        "citation_count": paper.get("citation_count"),
        "evidence_level": paper.get("evidence_level"),
        "fulltext_status": paper.get("fulltext_status", "not_verified"),
        "abstract": abstract[:320] if abstract else None,
    }


_INELIGIBLE_WORK_TYPES = {
    "component",
    "supplementary-material",
    "correction",
    "retraction",
    "peer-review",
}
_INELIGIBLE_TITLE = re.compile(
    r"\b(supplement(?:ary)?|supporting information|correction|corrigendum|erratum|retraction)\b|_supp\d*",
    re.IGNORECASE,
)


def _normalized_title(value: str) -> str:
    clean = re.sub(r"<[^>]+>", " ", html.unescape(value)).lower()
    return re.sub(r"[^a-z0-9]+", " ", clean).strip()


def _evidence_level(work_type: str | None) -> str:
    if work_type in {
        "article",
        "journal-article",
        "proceedings-article",
        "book-chapter",
    }:
        return "published_metadata_only"
    if work_type == "posted-content":
        return "preprint_metadata_only"
    return "other_metadata_only"


def _filter_papers(
    papers: list[dict[str, Any]], *, paradigm_family: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    del paradigm_family  # Scientific relevance is assessed by the Agent, not hard-coded here.
    eligible: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    seen_titles: set[str] = set()
    for paper in papers:
        reason: str | None = None
        title = str(paper.get("title") or "")
        if paper.get("is_retracted") is True:
            reason = "retracted"
        elif paper.get("work_type") in _INELIGIBLE_WORK_TYPES:
            reason = "ineligible_work_type"
        elif _INELIGIBLE_TITLE.search(title):
            reason = "supplement_correction_or_retraction_title"
        normalized = _normalized_title(title)
        if reason is None and normalized in seen_titles:
            reason = "duplicate_normalized_title"
        if reason is not None:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            continue
        seen_titles.add(normalized)
        enriched = dict(paper)
        enriched["evidence_level"] = _evidence_level(paper.get("work_type"))
        enriched["fulltext_status"] = "not_verified"
        eligible.append(enriched)
    return eligible, dict(sorted(reason_counts.items()))


def create_literature_scout_tools(
    *,
    search_space_path: Path,
    evidence_db_path: Path,
    search_run_id: str,
    source: Any | None = None,
    sources: tuple[Any, ...] | None = None,
    revision_critique_path: Path | None = None,
) -> tuple[ToolRegistry, dict[str, Any]]:
    search_space_path = Path(search_space_path).expanduser().resolve()
    dataset_id, queries, draft = _load_query_plan(search_space_path)
    dataset_profile, profile_ref = _load_bound_profile(draft, dataset_id=dataset_id)
    profile_field_catalog = dataset_profile_field_catalog(dataset_profile)
    allowed_profile_fields = tuple(profile_field_catalog)
    if not allowed_profile_fields:
        raise LiteratureScoutConfigurationError(
            "DatasetProfile exposes no fields for frontier direction binding"
        )
    store = LiteratureStore(Path(evidence_db_path))
    revision_request: dict[str, Any] | None = None
    if revision_critique_path is not None:
        critique_path = Path(revision_critique_path).expanduser().resolve()
        try:
            critique = json.loads(critique_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LiteratureScoutConfigurationError(
                f"Cannot load Dataset Critic revision request: {exc}"
            ) from exc
        if (
            critique.get("dataset_id") != dataset_id
            or critique.get("verdict") != "revise"
            or not critique.get("required_revisions")
        ):
            raise LiteratureScoutConfigurationError(
                "Revision request must be a revise verdict for the same dataset"
            )
        revision_request = {
            "path": str(critique_path),
            "sha256": hashlib.sha256(critique_path.read_bytes()).hexdigest(),
            "findings": critique.get("findings"),
            "required_revisions": critique["required_revisions"],
        }
        source_draft_ref = critique.get("source_draft") or {}
        source_draft_path = Path(
            str(source_draft_ref.get("path") or "")
        ).expanduser().resolve()
        expected_draft_hash = str(source_draft_ref.get("sha256") or "")
        if (
            not source_draft_path.is_file()
            or hashlib.sha256(source_draft_path.read_bytes()).hexdigest()
            != expected_draft_hash
        ):
            raise LiteratureScoutConfigurationError(
                "Revision request source draft failed integrity verification"
            )
        source_draft = json.loads(source_draft_path.read_text(encoding="utf-8"))
        revision_request["prior_frontier_directions"] = (
            (source_draft.get("frontier_space") or {}).get("directions") or []
        )
    if source is not None and sources is not None:
        raise LiteratureScoutConfigurationError("Use source or sources, not both")
    configured_sources = sources or (
        (source,) if source is not None else (CrossrefSource(), OpenAlexSource())
    )
    source_map = {str(item.name): item for item in configured_sources}
    if len(source_map) != len(configured_sources):
        raise LiteratureScoutConfigurationError("Scholarly source names must be unique")
    declared_source_names = {
        source_name
        for query in queries.values()
        for source_name in query.source_names
    }
    unsupported_sources = sorted(declared_source_names.difference(source_map))
    if unsupported_sources:
        raise LiteratureScoutConfigurationError(
            f"Query plan requires unavailable scholarly sources: {unsupported_sources}"
        )
    planned_search_keys = {
        (query.query_id, source_name)
        for query in queries.values()
        for source_name in query.source_names
    }
    registry = ToolRegistry()
    raw_query_plan = {
        str(item["query_id"]): item
        for item in draft["frontier_discovery"]["query_plan"]
    }
    eligible_paper_ids: set[str] = set()
    for query in queries.values():
        for source_name in query.source_names:
            cached = store.list_query_papers(
                search_run_id=search_run_id,
                query_id=query.query_id,
                source=source_name,
            )
            eligible_cached, _ = _filter_papers(
                cached,
                paradigm_family=str(
                    (raw_query_plan[query.query_id].get("derived_from_profile") or {}).get(
                        "paradigm", "unknown"
                    )
                ),
            )
            eligible_paper_ids.update(
                str(paper["stable_id"]) for paper in eligible_cached
            )

    def search_handler(query_id: str, source_name: str) -> dict[str, Any]:
        if query_id not in queries:
            raise LiteratureScoutConfigurationError(
                f"query_id must come from the dataset-derived plan: {query_id}"
            )
        query = queries[query_id]
        if source_name not in query.source_names:
            raise LiteratureScoutConfigurationError(
                f"source_name {source_name!r} is not planned for query {query_id!r}"
            )
        scholarly_source = source_map[source_name]
        attempts = store.list_search_attempts(search_run_id=search_run_id)
        prior_attempt = next(
            (
                item
                for item in attempts
                if item["query_id"] == query_id and item["source"] == source_name
            ),
            None,
        )
        if prior_attempt is not None and prior_attempt["status"] == "failed":
            return {
                "query_id": query_id,
                "query": query.text,
                "rationale": query.rationale,
                "source": scholarly_source.name,
                "source_status": "cached_failure",
                "stored_result_count": 0,
                "eligible_result_count": 0,
                "error": prior_attempt["error"],
                "warning": "The failed request remains in the immutable run ledger.",
            }
        if prior_attempt is not None:
            cached = store.list_query_papers(
                search_run_id=search_run_id, query_id=query_id, source=source_name
            )
            paradigm = str(
                (raw_query_plan[query_id].get("derived_from_profile") or {}).get(
                    "paradigm", "unknown"
                )
            )
            eligible_cached, excluded_reasons = _filter_papers(
                cached, paradigm_family=paradigm
            )
            eligible_paper_ids.update(
                str(paper["stable_id"]) for paper in eligible_cached
            )
            return {
                "query_id": query_id,
                "query": query.text,
                "rationale": query.rationale,
                "source": scholarly_source.name,
                "source_status": "cached_evidence",
                "stored_result_count": len(cached),
                "eligible_result_count": len(eligible_cached),
                "excluded_reason_counts": excluded_reasons,
                "papers": [
                    _compact_stored_paper(paper) for paper in eligible_cached[:4]
                ],
                "warning": "Cached metadata are discovery evidence, not proof of efficacy.",
            }
        try:
            papers = scholarly_source.search(query)
        except LiteratureSourceError as exc:
            store.record_search_failure(
                search_run_id=search_run_id,
                query=query,
                source=scholarly_source.name,
                error=str(exc),
            )
            raise
        store.record_search(
            search_run_id=search_run_id,
            query=query,
            source=scholarly_source.name,
            papers=papers,
        )
        paper_dicts = [paper.to_dict() | {"stable_id": paper.stable_id} for paper in papers]
        paradigm = str(
            (raw_query_plan[query_id].get("derived_from_profile") or {}).get(
                "paradigm", "unknown"
            )
        )
        eligible_dicts, excluded_reasons = _filter_papers(
            paper_dicts, paradigm_family=paradigm
        )
        eligible_paper_ids.update(str(paper["stable_id"]) for paper in eligible_dicts)
        return {
            "query_id": query_id,
            "query": query.text,
            "rationale": query.rationale,
            "source": scholarly_source.name,
            "stored_result_count": len(papers),
            "eligible_result_count": len(eligible_dicts),
            "excluded_metadata_only_count": len(papers) - len(eligible_dicts),
            "excluded_reason_counts": excluded_reasons,
            "source_status": "network_fetched",
            "papers": [_compact_stored_paper(paper) for paper in eligible_dicts[:4]],
            "warning": (
                "Metadata and abstracts are discovery evidence, not proof of method efficacy; "
                "full-text verification remains a later review task."
            ),
        }

    def record_handler(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        attempts = store.list_search_attempts(search_run_id=search_run_id)
        attempted_keys = {
            (item["query_id"], item["source"]) for item in attempts
        }
        missing_searches = sorted(planned_search_keys.difference(attempted_keys))
        if missing_searches:
            raise LiteratureScoutConfigurationError(
                f"All planned query/source pairs must be attempted first; missing: {missing_searches}"
            )
        successful_queries = {
            item["query_id"] for item in attempts if item["status"] == "completed"
        }
        queries_without_success = sorted(set(queries).difference(successful_queries))
        if queries_without_success:
            raise LiteratureScoutConfigurationError(
                f"Every query requires one successful scholarly source: {queries_without_success}"
            )
        profile_hash = profile_ref["sha256"]
        candidate_ids: set[str] = set()
        for item in candidates:
            candidate_id = str(item["candidate_id"]).strip()
            if not candidate_id or candidate_id in candidate_ids:
                raise LiteratureScoutConfigurationError(
                    "Frontier candidate IDs must be non-empty and unique"
                )
            candidate_ids.add(candidate_id)
            for field in (
                "method_family",
                "claim",
                "novelty_level",
                "proposed_validation",
            ):
                if not str(item[field]).strip():
                    raise LiteratureScoutConfigurationError(
                        f"Candidate {candidate_id} has empty {field}"
                    )
            for field in (
                "pipeline_stages",
                "applicability",
                "limitations",
                "supporting_papers",
                "future_protocol_requirements",
            ):
                values = item[field]
                if not values or any(not str(value).strip() for value in values):
                    raise LiteratureScoutConfigurationError(
                        f"Candidate {candidate_id} has invalid {field}"
                    )
            cited = set(str(value) for value in item["supporting_papers"])
            rejected = sorted(cited.difference(eligible_paper_ids))
            if rejected:
                raise LiteratureScoutConfigurationError(
                    f"Candidate {item['candidate_id']} cites filtered or unread evidence IDs: {rejected}"
                )
            binding = item["dataset_binding"]
            if binding.get("dataset_id") != dataset_id:
                raise LiteratureScoutConfigurationError(
                    f"Candidate {item['candidate_id']} is bound to another dataset"
                )
            if binding.get("dataset_profile_sha256") != profile_hash:
                raise LiteratureScoutConfigurationError(
                    f"Candidate {item['candidate_id']} is not bound to exact DatasetProfile SHA"
                )
            profile_fields = binding.get("supporting_profile_fields")
            if not isinstance(profile_fields, list) or not profile_fields:
                raise LiteratureScoutConfigurationError(
                    f"Candidate {item['candidate_id']} lacks supporting DatasetProfile fields"
                )
            invalid_fields = sorted(
                str(field)
                for field in profile_fields
                if str(field) not in profile_field_catalog
            )
            if invalid_fields:
                raise LiteratureScoutConfigurationError(
                    f"Candidate {item['candidate_id']} cites invalid DatasetProfile fields: "
                    f"{invalid_fields}. Copy exact paths from allowed_supporting_profile_fields."
                )
        normalized = [
            DatasetDirectionCandidate(
                candidate_id=str(item["candidate_id"]),
                method_family=str(item["method_family"]),
                pipeline_stages=tuple(str(value) for value in item["pipeline_stages"]),
                claim=str(item["claim"]),
                applicability=tuple(str(value) for value in item["applicability"]),
                limitations=tuple(str(value) for value in item["limitations"]),
                supporting_papers=tuple(str(value) for value in item["supporting_papers"]),
                novelty_level=str(item["novelty_level"]),
                future_protocol_requirements=tuple(
                    str(value) for value in item["future_protocol_requirements"]
                ),
                proposed_validation=str(item["proposed_validation"]),
                dataset_binding=dict(item["dataset_binding"]),
            )
            for item in candidates
        ]
        store.record_dataset_direction_candidates(
            search_run_id=search_run_id,
            candidates=normalized,
        )
        return {
            "recorded": len(normalized),
            "status": "dataset_frontier_hypothesis",
            "activation_performed": False,
            "session_roles_assigned": False,
        }

    def status_handler() -> dict[str, Any]:
        attempts = store.list_search_attempts(search_run_id=search_run_id)
        attempted_keys = {(item["query_id"], item["source"]) for item in attempts}
        directions = store.list_dataset_direction_candidates(search_run_id=search_run_id)
        missing = sorted(planned_search_keys.difference(attempted_keys))
        successful_queries = {
            item["query_id"] for item in attempts if item["status"] == "completed"
        }
        queries_without_success = sorted(set(queries).difference(successful_queries))
        failed = [item for item in attempts if item["status"] == "failed"]
        return {
            "dataset_id": dataset_id,
            "search_run_id": search_run_id,
            "planned_query_count": len(queries),
            "planned_search_count": len(planned_search_keys),
            "attempted_search_count": len(attempted_keys.intersection(planned_search_keys)),
            "missing_searches": [
                {"query_id": query_id, "source": source_name}
                for query_id, source_name in missing
            ],
            "queries_without_success": queries_without_success,
            "failed_searches": failed,
            "search_attempts": attempts,
            "direction_count": len(directions),
            "directions": directions,
            "complete": not missing and not queries_without_success and bool(directions),
            "activation_performed": False,
            "session_roles_assigned": False,
            "next_gate": "independent Dataset Critic and deterministic freeze",
        }

    revision_attempts = store.list_search_attempts(search_run_id=search_run_id)
    revision_attempted_keys = {
        (item["query_id"], item["source"]) for item in revision_attempts
    }
    revision_successful_queries = {
        item["query_id"]
        for item in revision_attempts
        if item["status"] == "completed"
    }
    reused_revision_coverage_complete = bool(revision_request) and (
        not planned_search_keys.difference(revision_attempted_keys)
        and not set(queries).difference(revision_successful_queries)
    )
    if not reused_revision_coverage_complete:
        registry.register(
            ToolDefinition(
                name="search_scholarly_metadata",
                description=(
                    "Execute one dataset-derived scholarly-network query and store "
                    "the complete metadata result in the evidence ledger."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query_id": {"type": "string", "enum": sorted(queries)},
                        "source_name": {
                            "type": "string",
                            "enum": sorted(declared_source_names),
                        },
                    },
                    "required": ["query_id", "source_name"],
                    "additionalProperties": False,
                },
                approval="never",
                decision_kind="read_only_literature_discovery",
                tags=("network-read", "scholarly-metadata", "audited"),
            ),
            search_handler,
        )
    candidate_schema = {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string"},
            "method_family": {"type": "string"},
            "pipeline_stages": {"type": "array", "items": {"type": "string"}},
            "claim": {"type": "string"},
            "applicability": {"type": "array", "items": {"type": "string"}},
            "limitations": {"type": "array", "items": {"type": "string"}},
            "supporting_papers": {"type": "array", "items": {"type": "string"}},
            "novelty_level": {"type": "string"},
            "future_protocol_requirements": {
                "type": "array",
                "items": {"type": "string"},
            },
            "proposed_validation": {"type": "string"},
            "dataset_binding": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "dataset_profile_sha256": {"type": "string"},
                    "supporting_profile_fields": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(allowed_profile_fields),
                        },
                    },
                },
                "required": [
                    "dataset_id",
                    "dataset_profile_sha256",
                    "supporting_profile_fields",
                ],
                "additionalProperties": False,
            },
        },
        "required": [
            "candidate_id",
            "method_family",
            "pipeline_stages",
            "claim",
            "applicability",
            "limitations",
            "supporting_papers",
            "novelty_level",
            "future_protocol_requirements",
            "proposed_validation",
            "dataset_binding",
        ],
        "additionalProperties": False,
    }
    registry.register(
        ToolDefinition(
            name="record_frontier_directions",
            description=(
                "Record evidence-linked, non-executable research directions after every "
                "planned query has been attempted."
            ),
            input_schema={
                "type": "object",
                "properties": {"candidates": {"type": "array", "items": candidate_schema}},
                "required": ["candidates"],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="proposal_recording_only",
            tags=("local-write", "evidence-ledger", "non-executable"),
        ),
        record_handler,
    )
    registry.register(
        ToolDefinition(
            name="inspect_frontier_discovery_status",
            description="Inspect query coverage and recorded frontier directions.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="read_only_status",
            tags=("read-only", "completion-check"),
        ),
        status_handler,
    )
    context = {
        "dataset_id": dataset_id,
        "search_space_path": str(search_space_path),
        "search_space_contract_id": draft.get("contract_id"),
        "dataset_hard_constraints": draft.get("dataset_hard_constraints"),
        "dataset_profile_provenance": (draft.get("provenance") or {}).get(
            "dataset_profile"
        ),
        "dataset_profile_summary": {
            root: dataset_profile[root] for root in DATASET_PROFILE_BINDING_ROOTS
        },
        "allowed_supporting_profile_fields": profile_field_catalog,
        "revision_evidence_policy": {
            "coverage_reused": reused_revision_coverage_complete,
            "search_tool_available": not reused_revision_coverage_complete,
            "network_search_required": not reused_revision_coverage_complete,
        },
        "stage_boundary": draft.get("stage_boundary"),
        "revision_request": revision_request,
        "query_plan": [
            {
                "query_id": query.query_id,
                "text": query.text,
                "rationale": query.rationale,
                "source_names": list(query.source_names),
            }
            for query in queries.values()
        ],
    }
    return registry, context


@dataclass
class LiteratureScoutAgent:
    runtime: AgentRuntime
    context: dict[str, Any]

    def run(self) -> AgentRunResult:
        def completion_check() -> dict[str, Any]:
            status = self.runtime.tools.execute(
                "inspect_frontier_discovery_status", {}
            )
            return {
                "complete": status["complete"],
                "planned_search_count": status["planned_search_count"],
                "attempted_search_count": status["attempted_search_count"],
                "missing_searches": status["missing_searches"],
                "queries_without_success": status["queries_without_success"],
                "direction_count": status["direction_count"],
                "failed_searches": status["failed_searches"],
            }

        return self.runtime.run(
            system_prompt=LITERATURE_SCOUT_SYSTEM_PROMPT,
            user_prompt=json.dumps(self.context, ensure_ascii=False, sort_keys=True),
            completion_check=completion_check,
            complete_on_tool_state=True,
        )
