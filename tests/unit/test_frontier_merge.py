from __future__ import annotations

import json
import hashlib
from pathlib import Path

from bci_autodiscovery.literature import (
    DatasetDirectionCandidate,
    LiteratureQuery,
    LiteratureStore,
    PaperRecord,
)
from bci_autodiscovery.search import build_combined_search_space_review


def test_combined_space_keeps_canonical_and_frontier_separate(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"dataset": {"id": "fixture_mi"}}), encoding="utf-8")
    profile_hash = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    canonical_path = tmp_path / "canonical.json"
    canonical_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "status": "dataset_coarse_space_awaiting_network_discovery",
                "dataset_id": "fixture_mi",
                "scope_policy": {"frontier_network_discovery_required": True},
                "canonical_space": {"dimensions": {"models": [{"component_id": "lda"}]}},
                "excluded_components": [],
                "compatibility_rules": [],
                "dataset_hard_constraints": {},
                "deferred_to_downstream_agents": {},
                "stage_boundary": {
                    "session_roles_assigned": False,
                    "evaluation_metrics_selected": False,
                    "experiment_budget_allocated": False,
                    "subject_data_accessed": False,
                    "confirmation_data_accessed": False,
                    "execution_activation_performed": False,
                },
                "provenance": {
                    "dataset_profile": {
                        "path": str(profile_path.resolve()),
                        "sha256": profile_hash,
                    }
                },
                "frontier_discovery": {
                    "query_plan": [{"query_id": "q1", "source_names": ["crossref"]}]
                },
            }
        ),
        encoding="utf-8",
    )
    store = LiteratureStore(tmp_path / "evidence.sqlite")
    query = LiteratureQuery(query_id="q1", text="query", rationale="dataset")
    paper = PaperRecord(
        source="crossref",
        source_id="paper",
        doi="10.1/paper",
        title="A frontier method",
    )
    store.record_search(search_run_id="lit", query=query, source="crossref", papers=[paper])
    store.record_dataset_direction_candidates(
        search_run_id="lit",
        candidates=[
            DatasetDirectionCandidate(
                candidate_id="new-method",
                method_family="new_method",
                pipeline_stages=("models",),
                claim="Test this hypothesis.",
                applicability=("multi-session",),
                limitations=("not implemented",),
                supporting_papers=("10.1/paper",),
                novelty_level="absent_from_registry",
                future_protocol_requirements=("leakage-safe downstream design",),
                proposed_validation="A downstream Agent must design validation.",
                dataset_binding={
                    "dataset_id": "fixture_mi",
                    "dataset_profile_sha256": profile_hash,
                    "supporting_profile_fields": ["sessions.sessions_per_subject"],
                },
            )
        ],
    )
    combined = build_combined_search_space_review(
        canonical_search_space_path=canonical_path,
        evidence_db_path=store.path,
        literature_run_id="lit",
    )
    assert combined["status"] == "dataset_level_draft_awaiting_critic"
    assert combined["canonical_space"]["dimensions"]["models"][0]["component_id"] == "lda"
    assert combined["frontier_space"]["directions"][0]["candidate_id"] == "new-method"
    assert combined["stage_boundary"]["execution_activation_performed"] is False
