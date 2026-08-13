from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from bci_autodiscovery.agents.literature_scout import (
    LiteratureScoutAgent,
    _filter_papers,
    create_literature_scout_tools,
)
from bci_autodiscovery.agents.contracts import ModelResponse, ToolCall
from bci_autodiscovery.agents.providers import ScriptedProvider
from bci_autodiscovery.agents.runtime import AgentRuntime
from bci_autodiscovery.agents.tools import (
    ToolArgumentError,
    ToolExecutionError,
    UnknownToolError,
)
from bci_autodiscovery.literature import (
    LiteratureSourceError,
    LiteratureStore,
    PaperRecord,
)


class _FakeSource:
    name = "crossref"

    def __init__(self):
        self.calls = 0

    def search(self, query):
        self.calls += 1
        return [
            PaperRecord(
                source="crossref",
                source_id=query.query_id,
                doi=f"10.1234/{query.query_id}",
                title=f"Evidence for {query.query_id}",
                abstract="A discovery abstract, not final proof.",
                work_type="journal-article",
            )
        ]


class _FailingSource:
    name = "crossref"

    def search(self, query):
        raise LiteratureSourceError(f"network unavailable for {query.query_id}")


def _draft(path: Path) -> str:
    profile_path = path.parent / "profile.json"
    validation_path = path.parent / "validation.json"
    validation_path.write_text('{"status":"valid"}', encoding="utf-8")
    validation_hash = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    profile_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset": {"id": "fixture_mi"},
                "paradigm": {
                    "family": "motor_imagery",
                    "actions": [{"label": "left"}, {"label": "right"}],
                },
                "resting_state": {"present": False},
                "signal": {
                    "modalities": ["EEG"],
                    "sampling_frequency_hz": 250.0,
                    "channel_count": 16,
                    "placement_scheme": "10-20",
                    "eog_channel_count": 0,
                },
                "equipment": {"amplifier": "fixture"},
                "events": {"common_analysis_window_s": [0.0, 4.0]},
                "sessions": {
                    "sessions_per_subject": 3,
                    "session_indices": [1, 2, 3],
                },
                "volume": {"trials": 300},
                "quality": {},
                "constraints": {
                    "allowed": [],
                    "forbidden": [],
                    "requires_research_design_decision": [],
                    "external_authority_blockers": [],
                },
                "evidence": [
                    {
                        "source": str(validation_path.resolve()),
                        "sha256": validation_hash,
                        "claim": "fixture validation",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    profile_hash = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "dataset_id": "fixture_mi",
                "contract_id": "fixture-mi-test",
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
                "dataset_hard_constraints": {},
                "frontier_discovery": {
                    "query_plan": [
                        {
                            "query_id": "q1",
                            "text": "first",
                            "rationale": "one",
                            "source_names": ["crossref"],
                        },
                        {
                            "query_id": "q2",
                            "text": "second",
                            "rationale": "two",
                            "source_names": ["crossref"],
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return profile_hash


def _candidate(profile_hash: str) -> dict:
    return {
        "candidate_id": "frontier-one",
        "method_family": "new_adaptation",
        "pipeline_stages": ["session_adaptation"],
        "claim": "A hypothesis worth testing.",
        "applicability": ["multi-session EEG"],
        "limitations": ["metadata evidence only"],
        "supporting_papers": ["10.1234/q1"],
        "novelty_level": "absent_from_registry",
        "future_protocol_requirements": ["downstream leakage-safe validation"],
        "proposed_validation": "A downstream Agent must design and test it.",
        "dataset_binding": {
            "dataset_id": "fixture_mi",
            "dataset_profile_sha256": profile_hash,
            "supporting_profile_fields": ["sessions.sessions_per_subject"],
        },
    }


def test_scout_requires_all_dataset_queries_before_recording_directions(tmp_path: Path) -> None:
    draft = tmp_path / "draft.json"
    profile_hash = _draft(draft)
    source = _FakeSource()
    tools, context = create_literature_scout_tools(
        search_space_path=draft,
        evidence_db_path=tmp_path / "evidence.sqlite",
        search_run_id="scout",
        source=source,
    )
    assert "sessions.sessions_per_subject" in context[
        "allowed_supporting_profile_fields"
    ]
    first = tools.execute(
        "search_scholarly_metadata", {"query_id": "q1", "source_name": "crossref"}
    )
    assert first["papers"][0]["stable_id"] == "10.1234/q1"
    cached = tools.execute(
        "search_scholarly_metadata", {"query_id": "q1", "source_name": "crossref"}
    )
    assert cached["source_status"] == "cached_evidence"
    assert source.calls == 1
    with pytest.raises(ToolExecutionError, match="missing.*q2"):
        tools.execute(
            "record_frontier_directions", {"candidates": [_candidate(profile_hash)]}
        )

    tools.execute(
        "search_scholarly_metadata", {"query_id": "q2", "source_name": "crossref"}
    )
    recorded = tools.execute(
        "record_frontier_directions", {"candidates": [_candidate(profile_hash)]}
    )
    assert recorded["activation_performed"] is False
    status = tools.execute("inspect_frontier_discovery_status", {})
    assert status["complete"] is True
    assert status["direction_count"] == 1


def test_scout_schema_rejects_semantic_aliases_and_exposes_exact_profile_paths(
    tmp_path: Path,
) -> None:
    draft = tmp_path / "draft.json"
    profile_hash = _draft(draft)
    tools, context = create_literature_scout_tools(
        search_space_path=draft,
        evidence_db_path=tmp_path / "evidence.sqlite",
        search_run_id="field-contract",
        source=_FakeSource(),
    )
    candidate = _candidate(profile_hash)
    candidate["dataset_binding"]["supporting_profile_fields"] = [
        "session_structure_repeated_sessions"
    ]

    with pytest.raises(ToolArgumentError, match="must be one of"):
        tools.execute("record_frontier_directions", {"candidates": [candidate]})

    assert context["allowed_supporting_profile_fields"][
        "sessions.sessions_per_subject"
    ] == 3


def test_scout_cannot_finish_after_invalid_record_and_repairs_itself(
    tmp_path: Path,
) -> None:
    draft = tmp_path / "draft.json"
    profile_hash = _draft(draft)
    tools, context = create_literature_scout_tools(
        search_space_path=draft,
        evidence_db_path=tmp_path / "evidence.sqlite",
        search_run_id="self-repair",
        source=_FakeSource(),
    )
    invalid = _candidate(profile_hash)
    invalid["dataset_binding"]["supporting_profile_fields"] = [
        "session_structure_repeated_sessions"
    ]
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="search-q1",
                        name="search_scholarly_metadata",
                        arguments={"query_id": "q1", "source_name": "crossref"},
                    ),
                    ToolCall(
                        id="search-q2",
                        name="search_scholarly_metadata",
                        arguments={"query_id": "q2", "source_name": "crossref"},
                    ),
                )
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="invalid-record",
                        name="record_frontier_directions",
                        arguments={"candidates": [invalid]},
                    ),
                )
            ),
            ModelResponse(content="I am done, incorrectly."),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="valid-record",
                        name="record_frontier_directions",
                        arguments={"candidates": [_candidate(profile_hash)]},
                    ),
                )
            ),
            ModelResponse(content="Discovery complete."),
        ]
    )

    result = LiteratureScoutAgent(
        runtime=AgentRuntime(provider=provider, tools=tools, run_id="self-repair"),
        context=context,
    ).run()

    assert result.status == "completed"
    assert result.final_text == "确定性完成门已确认所有必要工具状态。"
    assert tools.execute("inspect_frontier_discovery_status", {})["complete"] is True


def test_evidence_filter_only_applies_generic_metadata_hygiene() -> None:
    papers = [
        {
            "stable_id": "paper",
            "title": "Motor imagery EEG decoding with adaptation",
            "abstract": None,
            "work_type": "journal-article",
            "is_retracted": None,
        },
        {
            "stable_id": "duplicate",
            "title": "Motor Imagery EEG Decoding with Adaptation",
            "abstract": None,
            "work_type": "posted-content",
            "is_retracted": None,
        },
        {
            "stable_id": "supplement",
            "title": "Motor imagery EEG model_supp1.pdf",
            "abstract": None,
            "work_type": "component",
            "is_retracted": None,
        },
        {
            "stable_id": "vision",
            "title": "Self-supervised learning for image classification",
            "abstract": "STL-10 computer vision.",
            "work_type": "journal-article",
            "is_retracted": None,
        },
    ]
    eligible, reasons = _filter_papers(papers, paradigm_family="motor_imagery")
    assert [paper["stable_id"] for paper in eligible] == ["paper", "vision"]
    assert reasons == {
        "duplicate_normalized_title": 1,
        "ineligible_work_type": 1,
    }


def test_failed_network_attempt_is_cached_as_failure_not_evidence(tmp_path: Path) -> None:
    draft = tmp_path / "draft.json"
    _draft(draft)
    tools, _ = create_literature_scout_tools(
        search_space_path=draft,
        evidence_db_path=tmp_path / "failed.sqlite",
        search_run_id="failed-scout",
        source=_FailingSource(),
    )

    with pytest.raises(ToolExecutionError, match="network unavailable"):
        tools.execute(
            "search_scholarly_metadata",
            {"query_id": "q1", "source_name": "crossref"},
        )
    cached = tools.execute(
        "search_scholarly_metadata",
        {"query_id": "q1", "source_name": "crossref"},
    )

    assert cached["source_status"] == "cached_failure"
    assert cached["eligible_result_count"] == 0


def test_revision_reuses_search_evidence_without_repeating_network_calls(
    tmp_path: Path,
) -> None:
    draft_path = tmp_path / "draft.json"
    profile_hash = _draft(draft_path)
    evidence_path = tmp_path / "evidence.sqlite"
    source = _FakeSource()
    first_tools, _ = create_literature_scout_tools(
        search_space_path=draft_path,
        evidence_db_path=evidence_path,
        search_run_id="cycle-1",
        source=source,
    )
    for query_id in ("q1", "q2"):
        first_tools.execute(
            "search_scholarly_metadata",
            {"query_id": query_id, "source_name": "crossref"},
        )
    first_tools.execute(
        "record_frontier_directions",
        {"candidates": [_candidate(profile_hash)]},
    )
    assert source.calls == 2

    clone = LiteratureStore(evidence_path).clone_search_evidence(
        source_search_run_id="cycle-1",
        target_search_run_id="cycle-2",
    )
    assert clone == {"search_attempt_count": 2, "search_result_count": 2}
    prior_draft_path = tmp_path / "prior_dataset_draft.json"
    prior_draft_path.write_text(
        json.dumps(
            {"frontier_space": {"directions": [_candidate(profile_hash)]}}
        ),
        encoding="utf-8",
    )
    critique_path = tmp_path / "critique.json"
    critique_path.write_text(
        json.dumps(
            {
                "dataset_id": "fixture_mi",
                "verdict": "revise",
                "required_revisions": ["Narrow the frontier claim."],
                "findings": [],
                "source_draft": {
                    "path": str(prior_draft_path.resolve()),
                    "sha256": hashlib.sha256(prior_draft_path.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    revision_tools, revision_context = create_literature_scout_tools(
        search_space_path=draft_path,
        evidence_db_path=evidence_path,
        search_run_id="cycle-2",
        source=source,
        revision_critique_path=critique_path,
    )
    assert revision_context["revision_evidence_policy"] == {
        "coverage_reused": True,
        "search_tool_available": False,
        "network_search_required": False,
    }
    assert "search_scholarly_metadata" not in {
        item["function"]["name"] for item in revision_tools.definitions()
    }
    with pytest.raises(UnknownToolError, match="Unknown tool"):
        revision_tools.execute(
            "search_scholarly_metadata",
            {"query_id": "q1", "source_name": "crossref"},
        )
    assert source.calls == 2
    assert revision_context["revision_request"]["prior_frontier_directions"] == [
        _candidate(profile_hash)
    ]
    status = revision_tools.execute("inspect_frontier_discovery_status", {})
    assert status["attempted_search_count"] == 2
    assert status["missing_searches"] == []
    assert status["direction_count"] == 0
