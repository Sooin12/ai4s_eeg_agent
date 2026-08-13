"""SQLite evidence ledger for repeatable literature searches."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import DatasetDirectionCandidate, DirectionCandidate, LiteratureQuery, PaperRecord


class LiteratureEvidenceError(ValueError):
    """Raised when a direction claim is not linked to stored evidence."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LiteratureStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS searches (
                    search_run_id TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    query_json TEXT NOT NULL,
                    executed_at_utc TEXT NOT NULL,
                    result_count INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'completed',
                    error TEXT,
                    PRIMARY KEY (search_run_id, query_id, source)
                );
                CREATE TABLE IF NOT EXISTS papers (
                    stable_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    doi TEXT,
                    title TEXT NOT NULL,
                    year INTEGER,
                    authors_json TEXT NOT NULL,
                    venue TEXT,
                    url TEXT,
                    abstract TEXT,
                    work_type TEXT,
                    citation_count INTEGER,
                    is_retracted INTEGER,
                    metadata_json TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS search_results (
                    search_run_id TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    stable_id TEXT NOT NULL,
                    PRIMARY KEY (search_run_id, query_id, source, rank),
                    FOREIGN KEY (stable_id) REFERENCES papers(stable_id)
                );
                CREATE TABLE IF NOT EXISTS direction_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    search_run_id TEXT NOT NULL,
                    method_family TEXT NOT NULL,
                    pipeline_stages_json TEXT NOT NULL DEFAULT '[]',
                    claim TEXT NOT NULL,
                    applicability_json TEXT NOT NULL,
                    limitations_json TEXT NOT NULL,
                    supporting_papers_json TEXT NOT NULL,
                    novelty_level TEXT NOT NULL DEFAULT 'unspecified',
                    proposed_validation TEXT NOT NULL DEFAULT '',
                    protocol_binding_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dataset_direction_candidates (
                    search_run_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    method_family TEXT NOT NULL,
                    pipeline_stages_json TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    applicability_json TEXT NOT NULL,
                    limitations_json TEXT NOT NULL,
                    supporting_papers_json TEXT NOT NULL,
                    novelty_level TEXT NOT NULL,
                    future_protocol_requirements_json TEXT NOT NULL,
                    proposed_validation TEXT NOT NULL,
                    dataset_binding_json TEXT NOT NULL,
                    evidence_scope TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    PRIMARY KEY (search_run_id, candidate_id)
                );
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(direction_candidates)")
            }
            migrations = {
                "pipeline_stages_json": "TEXT NOT NULL DEFAULT '[]'",
                "novelty_level": "TEXT NOT NULL DEFAULT 'unspecified'",
                "proposed_validation": "TEXT NOT NULL DEFAULT ''",
                "protocol_binding_json": "TEXT NOT NULL DEFAULT '{}'",
            }
            for name, declaration in migrations.items():
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE direction_candidates ADD COLUMN {name} {declaration}"
                    )
            search_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(searches)")
            }
            if "status" not in search_columns:
                connection.execute(
                    "ALTER TABLE searches ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'"
                )
            if "error" not in search_columns:
                connection.execute("ALTER TABLE searches ADD COLUMN error TEXT")

    def record_search(
        self,
        *,
        search_run_id: str,
        query: LiteratureQuery,
        source: str,
        papers: list[PaperRecord],
    ) -> None:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO searches
                (search_run_id, query_id, source, query_json, executed_at_utc, result_count,
                 status, error)
                VALUES (?, ?, ?, ?, ?, ?, 'completed', NULL)
                """,
                (
                    search_run_id,
                    query.query_id,
                    source,
                    json.dumps(query.to_dict(), ensure_ascii=False, sort_keys=True),
                    now,
                    len(papers),
                ),
            )
            for rank, paper in enumerate(papers, start=1):
                connection.execute(
                    """
                    INSERT INTO papers
                    (stable_id, source, source_id, doi, title, year, authors_json, venue,
                     url, abstract, work_type, citation_count, is_retracted, metadata_json,
                     updated_at_utc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(stable_id) DO UPDATE SET
                      title=excluded.title, year=excluded.year,
                      authors_json=excluded.authors_json, venue=excluded.venue,
                      url=excluded.url, abstract=excluded.abstract,
                      work_type=excluded.work_type,
                      citation_count=excluded.citation_count,
                      is_retracted=excluded.is_retracted,
                      metadata_json=excluded.metadata_json,
                      updated_at_utc=excluded.updated_at_utc
                    """,
                    (
                        paper.stable_id,
                        paper.source,
                        paper.source_id,
                        paper.doi,
                        paper.title,
                        paper.year,
                        json.dumps(list(paper.authors), ensure_ascii=False),
                        paper.venue,
                        paper.url,
                        paper.abstract,
                        paper.work_type,
                        paper.citation_count,
                        None if paper.is_retracted is None else int(paper.is_retracted),
                        json.dumps(paper.raw_metadata, ensure_ascii=False, sort_keys=True),
                        now,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO search_results
                    (search_run_id, query_id, source, rank, stable_id)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (search_run_id, query.query_id, source, rank, paper.stable_id),
                )

    def record_search_failure(
        self,
        *,
        search_run_id: str,
        query: LiteratureQuery,
        source: str,
        error: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO searches
                (search_run_id, query_id, source, query_json, executed_at_utc, result_count,
                 status, error)
                VALUES (?, ?, ?, ?, ?, 0, 'failed', ?)
                """,
                (
                    search_run_id,
                    query.query_id,
                    source,
                    json.dumps(query.to_dict(), ensure_ascii=False, sort_keys=True),
                    _utc_now(),
                    error[:2000],
                ),
            )

    def attempted_query_ids(self, *, search_run_id: str) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT query_id FROM searches WHERE search_run_id = ?",
                (search_run_id,),
            )
            return {str(row["query_id"]) for row in rows}

    def attempted_search_keys(self, *, search_run_id: str) -> set[tuple[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT query_id, source FROM searches WHERE search_run_id = ?",
                (search_run_id,),
            )
            return {(str(row["query_id"]), str(row["source"])) for row in rows}

    def clone_search_evidence(
        self,
        *,
        source_search_run_id: str,
        target_search_run_id: str,
    ) -> dict[str, int]:
        """Clone immutable search coverage for an outcome-blind synthesis revision."""

        if source_search_run_id == target_search_run_id:
            raise LiteratureEvidenceError("Source and target search run IDs must differ")
        with self._connect() as connection:
            existing = int(
                connection.execute(
                    "SELECT COUNT(*) FROM searches WHERE search_run_id = ?",
                    (target_search_run_id,),
                ).fetchone()[0]
            )
            if existing:
                raise LiteratureEvidenceError(
                    f"Target search run already contains attempts: {target_search_run_id}"
                )
            source_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM searches WHERE search_run_id = ?",
                    (source_search_run_id,),
                ).fetchone()[0]
            )
            if source_count == 0:
                raise LiteratureEvidenceError(
                    f"Source search run contains no attempts: {source_search_run_id}"
                )
            connection.execute(
                """
                INSERT INTO searches
                (search_run_id, query_id, source, query_json, executed_at_utc,
                 result_count, status, error)
                SELECT ?, query_id, source, query_json, executed_at_utc,
                       result_count, status, error
                FROM searches
                WHERE search_run_id = ?
                """,
                (target_search_run_id, source_search_run_id),
            )
            connection.execute(
                """
                INSERT INTO search_results
                (search_run_id, query_id, source, rank, stable_id)
                SELECT ?, query_id, source, rank, stable_id
                FROM search_results
                WHERE search_run_id = ?
                """,
                (target_search_run_id, source_search_run_id),
            )
            result_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM search_results WHERE search_run_id = ?",
                    (target_search_run_id,),
                ).fetchone()[0]
            )
        return {
            "search_attempt_count": source_count,
            "search_result_count": result_count,
        }

    def list_search_attempts(self, *, search_run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT query_id, source, executed_at_utc, result_count, status, error
                FROM searches
                WHERE search_run_id = ?
                ORDER BY query_id, source
                """,
                (search_run_id,),
            )
            return [
                {
                    "query_id": str(row["query_id"]),
                    "source": str(row["source"]),
                    "executed_at_utc": str(row["executed_at_utc"]),
                    "result_count": int(row["result_count"]),
                    "status": str(row["status"]),
                    "error": row["error"],
                }
                for row in rows
            ]

    def list_query_papers(
        self,
        *,
        search_run_id: str,
        query_id: str,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            source_clause = " AND r.source = ?" if source is not None else ""
            parameters: tuple[Any, ...] = (
                (search_run_id, query_id, source)
                if source is not None
                else (search_run_id, query_id)
            )
            rows = connection.execute(
                f"""
                SELECT r.rank, p.stable_id, p.title, p.year, p.authors_json,
                       p.venue, p.url, p.abstract, p.work_type, p.is_retracted
                FROM search_results AS r
                JOIN papers AS p ON p.stable_id = r.stable_id
                WHERE r.search_run_id = ? AND r.query_id = ?
                {source_clause}
                ORDER BY r.rank
                """,
                parameters,
            )
            return [
                {
                    "rank": int(row["rank"]),
                    "stable_id": row["stable_id"],
                    "title": row["title"],
                    "year": row["year"],
                    "authors": json.loads(row["authors_json"]),
                    "venue": row["venue"],
                    "url": row["url"],
                    "abstract": row["abstract"],
                    "work_type": row["work_type"],
                    "is_retracted": (
                        None if row["is_retracted"] is None else bool(row["is_retracted"])
                    ),
                }
                for row in rows
            ]

    def known_paper_ids(self, *, search_run_id: str | None = None) -> set[str]:
        with self._connect() as connection:
            if search_run_id is None:
                rows = connection.execute("SELECT stable_id FROM papers")
            else:
                rows = connection.execute(
                    "SELECT DISTINCT stable_id FROM search_results WHERE search_run_id = ?",
                    (search_run_id,),
                )
            return {str(row["stable_id"]) for row in rows}

    def record_direction_candidates(
        self,
        *,
        search_run_id: str,
        candidates: list[DirectionCandidate],
    ) -> None:
        known = self.known_paper_ids(search_run_id=search_run_id)
        with self._connect() as connection:
            for candidate in candidates:
                missing = sorted(set(candidate.supporting_papers).difference(known))
                if not candidate.supporting_papers:
                    raise LiteratureEvidenceError(
                        f"Direction {candidate.candidate_id} has no supporting paper IDs"
                    )
                if missing:
                    raise LiteratureEvidenceError(
                        f"Direction {candidate.candidate_id} cites unstored paper IDs: {missing}"
                    )
                if candidate.status != "proposed_requires_review":
                    raise LiteratureEvidenceError(
                        "Literature scout may only record proposed_requires_review directions"
                    )
                connection.execute(
                    """
                    INSERT INTO direction_candidates
                    (candidate_id, search_run_id, method_family, pipeline_stages_json,
                     claim, applicability_json, limitations_json, supporting_papers_json,
                     novelty_level, proposed_validation, status, created_at_utc,
                     protocol_binding_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.candidate_id,
                        search_run_id,
                        candidate.method_family,
                        json.dumps(list(candidate.pipeline_stages), ensure_ascii=False),
                        candidate.claim,
                        json.dumps(list(candidate.applicability), ensure_ascii=False),
                        json.dumps(list(candidate.limitations), ensure_ascii=False),
                        json.dumps(list(candidate.supporting_papers), ensure_ascii=False),
                        candidate.novelty_level,
                        candidate.proposed_validation,
                        candidate.status,
                        _utc_now(),
                        json.dumps(candidate.protocol_binding, ensure_ascii=False, sort_keys=True),
                    ),
                )

    def list_direction_candidates(self, *, search_run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT candidate_id, method_family, pipeline_stages_json, claim,
                       applicability_json, limitations_json, supporting_papers_json,
                       novelty_level, proposed_validation, status
                       , protocol_binding_json
                FROM direction_candidates
                WHERE search_run_id = ?
                ORDER BY candidate_id
                """,
                (search_run_id,),
            )
            return [
                {
                    "candidate_id": row["candidate_id"],
                    "method_family": row["method_family"],
                    "pipeline_stages": json.loads(row["pipeline_stages_json"]),
                    "claim": row["claim"],
                    "applicability": json.loads(row["applicability_json"]),
                    "limitations": json.loads(row["limitations_json"]),
                    "supporting_papers": json.loads(row["supporting_papers_json"]),
                    "novelty_level": row["novelty_level"],
                    "proposed_validation": row["proposed_validation"],
                    "status": row["status"],
                    "protocol_binding": json.loads(row["protocol_binding_json"]),
                }
                for row in rows
            ]

    def record_dataset_direction_candidates(
        self,
        *,
        search_run_id: str,
        candidates: list[DatasetDirectionCandidate],
    ) -> None:
        known = self.known_paper_ids(search_run_id=search_run_id)
        with self._connect() as connection:
            for candidate in candidates:
                missing = sorted(set(candidate.supporting_papers).difference(known))
                if not candidate.supporting_papers:
                    raise LiteratureEvidenceError(
                        f"Dataset direction {candidate.candidate_id} has no supporting paper IDs"
                    )
                if missing:
                    raise LiteratureEvidenceError(
                        f"Dataset direction {candidate.candidate_id} cites unstored IDs: {missing}"
                    )
                if candidate.status != "dataset_frontier_hypothesis":
                    raise LiteratureEvidenceError(
                        "Dataset literature discovery may only record non-executable hypotheses"
                    )
                connection.execute(
                    """
                    INSERT INTO dataset_direction_candidates
                    (search_run_id, candidate_id, method_family, pipeline_stages_json,
                     claim, applicability_json, limitations_json, supporting_papers_json,
                     novelty_level, future_protocol_requirements_json, proposed_validation,
                     dataset_binding_json, evidence_scope, status, created_at_utc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        search_run_id,
                        candidate.candidate_id,
                        candidate.method_family,
                        json.dumps(list(candidate.pipeline_stages), ensure_ascii=False),
                        candidate.claim,
                        json.dumps(list(candidate.applicability), ensure_ascii=False),
                        json.dumps(list(candidate.limitations), ensure_ascii=False),
                        json.dumps(list(candidate.supporting_papers), ensure_ascii=False),
                        candidate.novelty_level,
                        json.dumps(
                            list(candidate.future_protocol_requirements), ensure_ascii=False
                        ),
                        candidate.proposed_validation,
                        json.dumps(candidate.dataset_binding, ensure_ascii=False, sort_keys=True),
                        candidate.evidence_scope,
                        candidate.status,
                        _utc_now(),
                    ),
                )

    def list_dataset_direction_candidates(
        self, *, search_run_id: str
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT candidate_id, method_family, pipeline_stages_json, claim,
                       applicability_json, limitations_json, supporting_papers_json,
                       novelty_level, future_protocol_requirements_json,
                       proposed_validation, dataset_binding_json, evidence_scope, status
                FROM dataset_direction_candidates
                WHERE search_run_id = ?
                ORDER BY candidate_id
                """,
                (search_run_id,),
            )
            return [
                {
                    "candidate_id": row["candidate_id"],
                    "method_family": row["method_family"],
                    "pipeline_stages": json.loads(row["pipeline_stages_json"]),
                    "claim": row["claim"],
                    "applicability": json.loads(row["applicability_json"]),
                    "limitations": json.loads(row["limitations_json"]),
                    "supporting_papers": json.loads(row["supporting_papers_json"]),
                    "novelty_level": row["novelty_level"],
                    "future_protocol_requirements": json.loads(
                        row["future_protocol_requirements_json"]
                    ),
                    "proposed_validation": row["proposed_validation"],
                    "dataset_binding": json.loads(row["dataset_binding_json"]),
                    "evidence_scope": row["evidence_scope"],
                    "status": row["status"],
                }
                for row in rows
            ]

    def summary(self) -> dict[str, int]:
        with self._connect() as connection:
            return {
                "searches": int(connection.execute("SELECT COUNT(*) FROM searches").fetchone()[0]),
                "papers": int(connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]),
                "search_results": int(
                    connection.execute("SELECT COUNT(*) FROM search_results").fetchone()[0]
                ),
                "direction_candidates": int(
                    connection.execute("SELECT COUNT(*) FROM direction_candidates").fetchone()[0]
                ),
                "dataset_direction_candidates": int(
                    connection.execute(
                        "SELECT COUNT(*) FROM dataset_direction_candidates"
                    ).fetchone()[0]
                ),
            }
