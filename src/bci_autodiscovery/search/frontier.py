"""Merge canonical and network-discovered dataset spaces without activation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bci_autodiscovery.literature import LiteratureStore


class FrontierMergeError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_dataset_level_review_draft(value: dict[str, Any]) -> None:
    if value.get("schema_version") != "2.0":
        raise FrontierMergeError("Unsupported dataset-level review schema_version")
    if value.get("status") != "dataset_level_draft_awaiting_critic":
        raise FrontierMergeError("Dataset-level review is incomplete")
    boundary = value.get("stage_boundary") or {}
    required_false = {
        "session_roles_assigned",
        "evaluation_metrics_selected",
        "experiment_budget_allocated",
        "subject_data_accessed",
        "confirmation_data_accessed",
        "execution_activation_performed",
    }
    if any(boundary.get(field) is not False for field in required_false):
        raise FrontierMergeError("Dataset-level draft crossed a downstream stage boundary")
    coverage = (value.get("frontier_space") or {}).get("network_coverage") or {}
    if coverage.get("missing_searches") or coverage.get("queries_without_success"):
        raise FrontierMergeError("Scholarly network coverage is incomplete")
    if int(coverage.get("attempted_search_count", -1)) != int(
        coverage.get("planned_search_count", -2)
    ):
        raise FrontierMergeError("Not every planned scholarly search was attempted")
    directions = (value.get("frontier_space") or {}).get("directions")
    if not isinstance(directions, list) or not directions:
        raise FrontierMergeError("Dataset-level draft has no frontier hypotheses")
    dataset_id = value.get("dataset_id")
    profile_hash = (
        (value.get("provenance") or {}).get("dataset_profile") or {}
    ).get("sha256")
    if not profile_hash:
        raise FrontierMergeError("Dataset-level draft lacks DatasetProfile hash")
    candidate_ids: set[str] = set()
    for direction in directions:
        if not isinstance(direction, dict):
            raise FrontierMergeError("Frontier direction must be an object")
        candidate_id = str(direction.get("candidate_id") or "")
        if not candidate_id or candidate_id in candidate_ids:
            raise FrontierMergeError("Frontier direction IDs must be non-empty and unique")
        candidate_ids.add(candidate_id)
        if direction.get("status") != "dataset_frontier_hypothesis":
            raise FrontierMergeError("Dataset frontier direction was activated or mis-staged")
        binding = direction.get("dataset_binding") or {}
        if (
            binding.get("dataset_id") != dataset_id
            or binding.get("dataset_profile_sha256") != profile_hash
        ):
            raise FrontierMergeError("Frontier direction has invalid DatasetProfile binding")
        if not direction.get("supporting_papers"):
            raise FrontierMergeError("Frontier direction has no scholarly evidence IDs")


def build_combined_search_space_review(
    *,
    canonical_search_space_path: Path,
    evidence_db_path: Path,
    literature_run_id: str,
) -> dict[str, Any]:
    canonical_path = Path(canonical_search_space_path).expanduser().resolve()
    db_path = Path(evidence_db_path).expanduser().resolve()
    try:
        canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrontierMergeError(f"Cannot load canonical search space: {exc}") from exc
    if canonical.get("schema_version") != "2.0":
        raise FrontierMergeError("Canonical search space is not the protocol-free v2 contract")
    if canonical.get("status") != "dataset_coarse_space_awaiting_network_discovery":
        raise FrontierMergeError("Canonical search space has an invalid stage status")
    if not canonical.get("scope_policy", {}).get("frontier_network_discovery_required"):
        raise FrontierMergeError("Canonical contract does not require frontier discovery")
    forbidden_legacy = {"protocol", "human_approval_required"}.intersection(canonical)
    if forbidden_legacy:
        raise FrontierMergeError(
            f"Canonical dataset space retains legacy downstream fields: {sorted(forbidden_legacy)}"
        )

    store = LiteratureStore(db_path)
    query_plan = canonical["frontier_discovery"]["query_plan"]
    planned_keys = {
        (str(item["query_id"]), str(source_name))
        for item in query_plan
        for source_name in item.get("source_names") or ["crossref"]
    }
    attempts = store.list_search_attempts(search_run_id=literature_run_id)
    attempted_keys = {(item["query_id"], item["source"]) for item in attempts}
    missing = sorted(planned_keys.difference(attempted_keys))
    successful_queries = {
        item["query_id"] for item in attempts if item["status"] == "completed"
    }
    planned_queries = {str(item["query_id"]) for item in query_plan}
    queries_without_success = sorted(planned_queries.difference(successful_queries))
    directions = store.list_dataset_direction_candidates(
        search_run_id=literature_run_id
    )
    ready_for_critic = not missing and not queries_without_success and bool(directions)
    profile_ref = (canonical.get("provenance") or {}).get("dataset_profile") or {}
    result = {
        "schema_version": "2.0",
        "contract_id": f"{canonical['dataset_id']}-dataset-level-review-v1",
        "dataset_id": canonical["dataset_id"],
        "status": (
            "dataset_level_draft_awaiting_critic"
            if ready_for_critic
            else "dataset_level_discovery_incomplete"
        ),
        "canonical_space": canonical["canonical_space"],
        "excluded_components": canonical["excluded_components"],
        "compatibility_rules": canonical["compatibility_rules"],
        "dataset_hard_constraints": canonical["dataset_hard_constraints"],
        "deferred_to_downstream_agents": canonical["deferred_to_downstream_agents"],
        "frontier_space": {
            "literature_run_id": literature_run_id,
            "network_coverage": {
                "planned_query_count": len(planned_queries),
                "planned_search_count": len(planned_keys),
                "attempted_search_count": len(attempted_keys.intersection(planned_keys)),
                "missing_searches": [
                    {"query_id": query_id, "source": source_name}
                    for query_id, source_name in missing
                ],
                "queries_without_success": queries_without_success,
                "failed_searches": [
                    item for item in attempts if item["status"] == "failed"
                ],
                "search_attempts": attempts,
            },
            "directions": directions,
            "direction_count": len(directions),
            "evidence_boundary": (
                "Directions are metadata/abstract-backed hypotheses, not efficacy claims "
                "or executable pipeline activation."
            ),
        },
        "stage_boundary": canonical["stage_boundary"],
        "provenance": {
            "dataset_profile": profile_ref,
            "canonical_search_space": {
                "path": str(canonical_path),
                "sha256": _sha256(canonical_path),
            },
            "component_registry": (canonical.get("provenance") or {}).get(
                "component_registry"
            ),
            "evidence_db": {"path": str(db_path), "sha256": _sha256(db_path)},
        },
    }
    if ready_for_critic:
        validate_dataset_level_review_draft(result)
    return result
