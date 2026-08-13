from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from bci_autodiscovery.agents import dataset_critic_cli, dataset_revision_cli
from bci_autodiscovery.agents.contracts import ModelResponse, ToolCall
from bci_autodiscovery.agents.dataset_intelligence_critic import (
    DatasetIntelligenceCriticAgent,
    DatasetIntelligenceCriticError,
    create_dataset_intelligence_critic_tools,
    validate_dataset_critique,
    validate_dataset_level_sources,
)
from bci_autodiscovery.agents.dataset_level_agent import (
    DatasetLevelAgent,
    _frontier_semantic_hash,
)
from bci_autodiscovery.agents.providers import ScriptedProvider
from bci_autodiscovery.agents.runtime import AgentRuntime
from bci_autodiscovery.agents.run_recovery import (
    AgentRecoveryError,
    assert_source_run_recoverable,
    write_process_state,
)
from bci_autodiscovery.literature import (
    DatasetDirectionCandidate,
    LiteratureQuery,
    LiteratureStore,
    PaperRecord,
)
from bci_autodiscovery.search import (
    build_combined_search_space_review,
    build_search_space_draft,
)
from bci_autodiscovery.workflow import (
    DatasetIntelligenceLoop,
    DatasetIntelligenceWorkflowError,
    freeze_dataset_level_contract,
)
from bci_autodiscovery.workflow.autonomy import sha256_path


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _profile() -> dict:
    return {
        "schema_version": "1.0",
        "dataset": {"id": "unseen-dataset"},
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
        "equipment": {"amplifier": "unknown"},
        "events": {"common_analysis_window_s": [0.0, 4.0]},
        "sessions": {"sessions_per_subject": 3, "session_indices": [0, 1, 2]},
        "volume": {"trials": 300},
        "quality": {"limitations": ["fixture metadata only"]},
        "constraints": {
            "allowed": [],
            "forbidden": [],
            "requires_research_design_decision": ["future session roles"],
            "external_authority_blockers": [],
        },
        "evidence": [{"source": "fixture-validation", "claim": "normalized profile"}],
    }


def _profile_with_hashed_evidence(tmp_path: Path) -> dict:
    evidence_path = tmp_path / "fixture_validation.json"
    evidence_path.write_text('{"status":"valid"}', encoding="utf-8")
    profile = _profile()
    profile["evidence"] = [
        {
            "source": str(evidence_path.resolve()),
            "sha256": sha256_path(evidence_path),
            "claim": "normalized profile fixture",
        }
    ]
    return profile


def _complete_review(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    profile_path = tmp_path / "dataset_profile.json"
    canonical_path = tmp_path / "canonical.json"
    evidence_path = tmp_path / "evidence.sqlite"
    review_path = tmp_path / "dataset_review.json"
    _write(profile_path, _profile_with_hashed_evidence(tmp_path))
    registry_path = Path("configs/component_registry.v0.json").resolve()
    canonical = build_search_space_draft(
        dataset_profile_path=str(profile_path),
        component_registry_path=str(registry_path),
    )
    _write(canonical_path, canonical)

    store = LiteratureStore(evidence_path)
    first_paper_id = ""
    for planned in canonical["frontier_discovery"]["query_plan"]:
        query = LiteratureQuery(
            query_id=planned["query_id"],
            text=planned["text"],
            rationale=planned["rationale"],
            source_names=tuple(planned["source_names"]),
        )
        for source_name in planned["source_names"]:
            doi = f"10.1000/{planned['query_id']}-{source_name}"
            first_paper_id = first_paper_id or doi
            store.record_search(
                search_run_id="literature-run",
                query=query,
                source=source_name,
                papers=[
                    PaperRecord(
                        source=source_name,
                        source_id=doi,
                        doi=doi,
                        title=f"Evidence for {planned['query_id']}",
                        abstract="Discovery evidence only.",
                        work_type="journal-article",
                    )
                ],
            )
    store.record_dataset_direction_candidates(
        search_run_id="literature-run",
        candidates=[
            DatasetDirectionCandidate(
                candidate_id="cross-session-adaptation",
                method_family="domain_adaptation",
                pipeline_stages=("session_adaptation", "models"),
                claim="Repeated sessions make adaptation a plausible hypothesis.",
                applicability=("multi-session EEG",),
                limitations=("Metadata and abstract evidence do not prove efficacy.",),
                supporting_papers=(first_paper_id,),
                novelty_level="frontier_for_local_registry",
                future_protocol_requirements=("Define leakage-safe target access later.",),
                proposed_validation="A downstream research-design Agent must test it.",
                dataset_binding={
                    "dataset_id": "unseen-dataset",
                    "dataset_profile_sha256": sha256_path(profile_path),
                    "supporting_profile_fields": ["sessions.sessions_per_subject"],
                },
            )
        ],
    )
    review = build_combined_search_space_review(
        canonical_search_space_path=canonical_path,
        evidence_db_path=evidence_path,
        literature_run_id="literature-run",
    )
    _write(review_path, review)
    return profile_path, evidence_path, review_path, review


def test_frontier_revision_hash_ignores_ids_but_detects_scientific_change() -> None:
    first = {
        "frontier_space": {
            "directions": [
                {
                    "candidate_id": "first-id",
                    "claim": "A claim",
                    "supporting_papers": ["paper-b", "paper-a"],
                    "status": "proposed_requires_review",
                }
            ]
        }
    }
    renamed = json.loads(json.dumps(first))
    renamed["frontier_space"]["directions"][0]["candidate_id"] = "renamed-id"
    renamed["frontier_space"]["directions"][0]["supporting_papers"].reverse()
    revised = json.loads(json.dumps(first))
    revised["frontier_space"]["directions"][0]["claim"] = "A narrower claim"

    assert _frontier_semantic_hash(first) == _frontier_semantic_hash(renamed)
    assert _frontier_semantic_hash(first) != _frontier_semantic_hash(revised)


def test_recovery_refuses_live_or_already_completed_source_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "source-run"
    draft_path = run_dir / "cycles" / "0001" / "dataset_level_draft.json"
    draft_path.parent.mkdir(parents=True)
    _write(draft_path, {"dataset_id": "fixture"})
    write_process_state(run_dir, run_id="source-run", status="running")

    with pytest.raises(AgentRecoveryError, match="still alive"):
        assert_source_run_recoverable(draft_path)

    _write(
        run_dir / "dataset_level_run.json",
        {"run_id": "source-run", "status": "completed"},
    )
    with pytest.raises(AgentRecoveryError, match="already terminal"):
        assert_source_run_recoverable(draft_path)


def test_independent_dataset_critic_can_freeze_exact_validated_contract(
    tmp_path: Path,
) -> None:
    _, _, review_path, review = _complete_review(tmp_path)
    review_hash = sha256_path(review_path)
    critique = {
        "schema_version": "1.0",
        "review_id": "dataset-critic-fixture",
        "dataset_id": "unseen-dataset",
        "reviewed_draft_sha256": review_hash,
        "verdict": "pass",
        "findings": [
            {
                "code": "stage-isolation-confirmed",
                "dimension": "stage_boundary",
                "severity": "note",
                "owner": "none",
                "message": "The dataset draft activates no downstream research choices.",
                "evidence_refs": ["draft.stage_boundary"],
            }
        ],
        "required_revisions": [],
        "rationale": "Deterministic integrity and dataset-level scope checks passed.",
    }
    tools, context = create_dataset_intelligence_critic_tools(
        dataset_level_draft_path=review_path
    )
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="read",
                        name="read_dataset_critic_context",
                        arguments={},
                    ),
                )
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="record",
                        name="record_dataset_critique",
                        arguments={"critique": critique},
                    ),
                )
            ),
            ModelResponse(content="Dataset Critic completed."),
        ]
    )
    result = DatasetIntelligenceCriticAgent(
        runtime=AgentRuntime(provider=provider, tools=tools, run_id="critic-fixture"),
        context=context,
    ).run()
    recorded = result.latest_tool_result("record_dataset_critique")
    assert recorded["critic_independence"]["authored_draft"] is False
    critique_path = tmp_path / "dataset_critique.json"
    _write(critique_path, recorded)

    frozen_path = tmp_path / "dataset_level_contract.json"
    frozen = freeze_dataset_level_contract(
        dataset_level_draft_path=review_path,
        dataset_critique_path=critique_path,
        output_path=frozen_path,
    )

    assert frozen["status"] == "frozen_dataset_level_contract"
    assert frozen["canonical_space"] == review["canonical_space"]
    assert frozen["freeze_record"]["human_itemized_approval_used"] is False
    assert frozen["freeze_record"]["session_protocol_created"] is False


def test_dataset_critic_cannot_route_immutable_blocker_to_literature_revision(
    tmp_path: Path,
) -> None:
    _, _, review_path, review = _complete_review(tmp_path)
    critique = {
        "schema_version": "1.0",
        "review_id": "misrouted-revision",
        "dataset_id": "unseen-dataset",
        "reviewed_draft_sha256": sha256_path(review_path),
        "verdict": "revise",
        "findings": [
            {
                "code": "registry-gap",
                "dimension": "canonical_coverage",
                "severity": "major",
                "owner": "canonical_registry",
                "message": "The immutable registry would require a code change.",
                "evidence_refs": ["canonical_space"],
            }
        ],
        "required_revisions": ["Add the missing registry family."],
        "rationale": "This cannot be repaired by Literature Scout.",
    }

    with pytest.raises(DatasetIntelligenceCriticError, match="cannot modify"):
        validate_dataset_critique(
            critique,
            draft=review,
            draft_sha256=sha256_path(review_path),
            deterministic_validation_passed=True,
        )


def test_interrupted_run_can_resume_at_critic_without_repeating_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, review_path, _ = _complete_review(tmp_path)
    critique = {
        "schema_version": "1.0",
        "review_id": "resumed-critic",
        "dataset_id": "unseen-dataset",
        "reviewed_draft_sha256": sha256_path(review_path),
        "verdict": "pass",
        "findings": [],
        "required_revisions": [],
        "rationale": "The immutable draft passed after an interrupted parent run.",
    }
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(ToolCall("read", "read_dataset_critic_context", {}),)
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "record",
                        "record_dataset_critique",
                        {"critique": critique},
                    ),
                )
            ),
            ModelResponse(content="Critic resume completed."),
        ]
    )
    monkeypatch.setattr(dataset_critic_cli, "_provider", lambda _args: provider)
    run_dir = tmp_path / "critic-resume"

    exit_code = dataset_critic_cli.main(
        [
            "--dataset-level-draft",
            str(review_path),
            "--provider",
            "kimi",
            "--run-id",
            "critic-resume-fixture",
            "--run-dir",
            str(run_dir),
        ]
    )

    assert exit_code == 0
    assert (run_dir / "dataset_critique.json").is_file()
    frozen = json.loads(
        (run_dir / "dataset_level_contract.json").read_text(encoding="utf-8")
    )
    assert frozen["status"] == "frozen_dataset_level_contract"


def test_literature_revision_resume_reuses_evidence_and_can_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, source_draft_path, source_draft = _complete_review(tmp_path)
    original = source_draft["frontier_space"]["directions"][0]
    revised = json.loads(json.dumps(original))
    revised.pop("evidence_scope", None)
    revised.pop("status", None)
    revised["claim"] = (
        "Repeated sessions may make adaptation worth testing; metadata and abstract "
        "discovery do not establish efficacy or activate execution."
    )
    critique_path = tmp_path / "revision_request.json"
    _write(
        critique_path,
        {
            "schema_version": "1.0",
            "review_id": "revision-request",
            "dataset_id": "unseen-dataset",
            "reviewed_draft_sha256": sha256_path(source_draft_path),
            "verdict": "revise",
            "findings": [
                {
                    "code": "overclaim",
                    "dimension": "frontier_claims",
                    "severity": "major",
                    "owner": "literature_scout",
                    "message": "Narrow the claim.",
                    "evidence_refs": ["frontier_space.directions"],
                }
            ],
            "required_revisions": ["Narrow the frontier claim."],
            "rationale": "Literature wording requires repair.",
            "source_draft": {
                "path": str(source_draft_path.resolve()),
                "sha256": sha256_path(source_draft_path),
            },
        },
    )
    literature_provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall("status-before", "inspect_frontier_discovery_status", {}),
                )
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "record-revision",
                        "record_frontier_directions",
                        {"candidates": [revised]},
                    ),
                )
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall("status-after", "inspect_frontier_discovery_status", {}),
                )
            ),
            ModelResponse(content="Literature revision completed."),
        ]
    )
    run_dir = tmp_path / "revision-resume"
    provider_calls = 0

    def provider_factory(_args):
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            return literature_provider
        revised_draft_path = run_dir / "dataset_level_draft.json"
        pass_critique = {
            "schema_version": "1.0",
            "review_id": "revised-pass",
            "dataset_id": "unseen-dataset",
            "reviewed_draft_sha256": sha256_path(revised_draft_path),
            "verdict": "pass",
            "findings": [],
            "required_revisions": [],
            "rationale": "The narrowed evidence-bound direction passed.",
        }
        return ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(ToolCall("read", "read_dataset_critic_context", {}),)
                ),
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            "record-pass",
                            "record_dataset_critique",
                            {"critique": pass_critique},
                        ),
                    )
                ),
                ModelResponse(content="Revised draft passed."),
            ]
        )

    monkeypatch.setattr(dataset_revision_cli, "_provider", provider_factory)
    exit_code = dataset_revision_cli.main(
        [
            "--source-draft",
            str(source_draft_path),
            "--revision-critique",
            str(critique_path),
            "--provider",
            "kimi",
            "--run-id",
            "revision-resume-fixture",
            "--run-dir",
            str(run_dir),
        ]
    )

    assert exit_code == 0
    assert provider_calls == 2
    status = json.loads(
        (run_dir / "frontier_discovery.json").read_text(encoding="utf-8")
    )
    assert status["attempted_search_count"] == status["planned_search_count"]
    assert (run_dir / "dataset_level_contract.json").is_file()
    audit_events = [
        json.loads(line)
        for line in (run_dir / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    reuse = next(
        item for item in audit_events if item["event_type"] == "literature_evidence_reused"
    )
    assert reuse["payload"]["network_access_allowed"] is False

    recovered_run_dir = tmp_path / "revision-recovered-after-final-text-failure"

    def recovered_critic_provider(_args):
        recovered_draft_path = recovered_run_dir / "dataset_level_draft.json"
        pass_critique = {
            "schema_version": "1.0",
            "review_id": "recovered-pass",
            "dataset_id": "unseen-dataset",
            "reviewed_draft_sha256": sha256_path(recovered_draft_path),
            "verdict": "pass",
            "findings": [],
            "required_revisions": [],
            "rationale": "Recovered completed tool state passed.",
        }
        return ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(ToolCall("read", "read_dataset_critic_context", {}),)
                ),
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            "record",
                            "record_dataset_critique",
                            {"critique": pass_critique},
                        ),
                    )
                ),
            ]
        )

    monkeypatch.setattr(
        dataset_revision_cli, "_provider", recovered_critic_provider
    )
    recovered_exit = dataset_revision_cli.main(
        [
            "--source-draft",
            str(source_draft_path),
            "--revision-critique",
            str(critique_path),
            "--completed-revision-evidence",
            str(run_dir / "evidence.sqlite"),
            "--completed-literature-run-id",
            "revision-resume-fixture:literature-revision",
            "--provider",
            "kimi",
            "--run-id",
            "revision-recovery-fixture",
            "--run-dir",
            str(recovered_run_dir),
        ]
    )

    assert recovered_exit == 0
    assert (recovered_run_dir / "dataset_level_contract.json").is_file()
    assert not (recovered_run_dir / "literature_agent_result.json").exists()


def test_dataset_critic_repairs_failed_verdict_record_before_completion(
    tmp_path: Path,
) -> None:
    _, _, review_path, _ = _complete_review(tmp_path)
    valid_critique = {
        "schema_version": "1.0",
        "review_id": "critic-self-repair",
        "dataset_id": "unseen-dataset",
        "reviewed_draft_sha256": sha256_path(review_path),
        "verdict": "pass",
        "findings": [],
        "required_revisions": [],
        "rationale": "The corrected verdict passes all deterministic checks.",
    }
    invalid_critique = dict(valid_critique)
    invalid_critique["reviewed_draft_sha256"] = "0" * 64
    tools, context = create_dataset_intelligence_critic_tools(
        dataset_level_draft_path=review_path
    )
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="read",
                        name="read_dataset_critic_context",
                        arguments={},
                    ),
                )
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="invalid-record",
                        name="record_dataset_critique",
                        arguments={"critique": invalid_critique},
                    ),
                )
            ),
            ModelResponse(content="Premature completion."),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="valid-record",
                        name="record_dataset_critique",
                        arguments={"critique": valid_critique},
                    ),
                )
            ),
            ModelResponse(content="Dataset Critic completed after repair."),
        ]
    )

    result = DatasetIntelligenceCriticAgent(
        runtime=AgentRuntime(provider=provider, tools=tools, run_id="critic-self-repair"),
        context=context,
    ).run()

    assert result.status == "completed"
    assert result.latest_tool_result("record_dataset_critique")["review_id"] == (
        "critic-self-repair"
    )
    assert tools.execute("inspect_dataset_critic_status", {})["complete"] is True


def test_dataset_contract_freeze_rejects_changed_source_profile(tmp_path: Path) -> None:
    profile_path, _, review_path, _ = _complete_review(tmp_path)
    critique_path = tmp_path / "critique.json"
    _write(
        critique_path,
        {
            "schema_version": "1.0",
            "review_id": "cannot-pass-tamper",
            "dataset_id": "unseen-dataset",
            "reviewed_draft_sha256": sha256_path(review_path),
            "verdict": "pass",
            "findings": [],
            "required_revisions": [],
            "rationale": "This verdict must not bypass provenance checks.",
            "source_draft": {
                "path": str(review_path.resolve()),
                "sha256": sha256_path(review_path),
            },
        },
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["volume"]["trials"] += 1
    _write(profile_path, profile)

    with pytest.raises(DatasetIntelligenceWorkflowError, match="integrity"):
        freeze_dataset_level_contract(
            dataset_level_draft_path=review_path,
            dataset_critique_path=critique_path,
            output_path=tmp_path / "must-not-exist.json",
        )
    assert not (tmp_path / "must-not-exist.json").exists()


def test_deterministic_dataset_review_rejects_evidence_ledger_tampering(
    tmp_path: Path,
) -> None:
    _, evidence_path, review_path, review = _complete_review(tmp_path)
    evidence_path.write_bytes(evidence_path.read_bytes() + b"tamper")

    with pytest.raises(DatasetIntelligenceCriticError, match="integrity"):
        validate_dataset_level_sources(review, draft_path=review_path)


def test_deterministic_dataset_review_rejects_invented_profile_field_alias(
    tmp_path: Path,
) -> None:
    _, evidence_path, review_path, _ = _complete_review(tmp_path)
    with sqlite3.connect(evidence_path) as connection:
        row = connection.execute(
            """
            SELECT dataset_binding_json
            FROM dataset_direction_candidates
            WHERE search_run_id = ? AND candidate_id = ?
            """,
            ("literature-run", "cross-session-adaptation"),
        ).fetchone()
        assert row is not None
        binding = json.loads(row[0])
        binding["supporting_profile_fields"] = [
            "session_structure_repeated_sessions"
        ]
        connection.execute(
            """
            UPDATE dataset_direction_candidates
            SET dataset_binding_json = ?
            WHERE search_run_id = ? AND candidate_id = ?
            """,
            (
                json.dumps(binding, ensure_ascii=False, sort_keys=True),
                "literature-run",
                "cross-session-adaptation",
            ),
        )

    canonical_path = tmp_path / "canonical.json"
    rebuilt_review = build_combined_search_space_review(
        canonical_search_space_path=canonical_path,
        evidence_db_path=evidence_path,
        literature_run_id="literature-run",
    )
    _write(review_path, rebuilt_review)

    with pytest.raises(
        DatasetIntelligenceCriticError,
        match="invalid DatasetProfile fields",
    ):
        validate_dataset_level_sources(rebuilt_review, draft_path=review_path)


def test_revision_loop_fails_closed_and_audits_an_unchanged_redraft(
    tmp_path: Path,
) -> None:
    _, _, _, review = _complete_review(tmp_path)

    def drafter(_cycle: int, _previous_draft: Path | None, _previous_critique: Path | None):
        return review

    def critic(draft_path: Path, cycle: int):
        assert cycle == 1
        return {
            "schema_version": "1.0",
            "review_id": "needs-revision",
            "dataset_id": "unseen-dataset",
            "reviewed_draft_sha256": sha256_path(draft_path),
            "verdict": "revise",
            "findings": [
                {
                    "code": "claim-needs-revision",
                    "dimension": "frontier_claims",
                    "severity": "major",
                    "owner": "literature_scout",
                    "message": "Revise the frontier claim.",
                    "evidence_refs": ["frontier_space.directions"],
                }
            ],
            "required_revisions": ["Revise the frontier claim."],
            "rationale": "A concrete repair is required.",
        }

    loop_dir = tmp_path / "loop"
    result = DatasetIntelligenceLoop(run_dir=loop_dir).run(
        drafter=drafter,
        critic=critic,
    )

    assert result.status == "failed"
    assert "did not change" in (result.error or "")
    assert (loop_dir / "dataset_intelligence_state.json").is_file()


class _NamedSource:
    def __init__(self, name: str) -> None:
        self.name = name

    def search(self, query: LiteratureQuery) -> list[PaperRecord]:
        doi = f"10.5000/{self.name}-{query.query_id}"
        return [
            PaperRecord(
                source=self.name,
                source_id=doi,
                doi=doi,
                title=f"{self.name} evidence for {query.query_id}",
                abstract="Metadata-level discovery evidence.",
                work_type="journal-article" if self.name == "crossref" else "article",
            )
        ]


def test_top_level_dataset_agent_runs_all_subagents_and_freezes_contract(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "input_profile.json"
    _write(profile_path, _profile_with_hashed_evidence(tmp_path))
    run_dir = tmp_path / "dataset-agent-run"

    def literature_provider(_cycle: int) -> ScriptedProvider:
        canonical = json.loads(
            (run_dir / "canonical_search_space.json").read_text(encoding="utf-8")
        )
        profile_hash = sha256_path(run_dir / "dataset_profile.json")
        calls = tuple(
            ToolCall(
                id=f"search-{query['query_id']}-{source_name}",
                name="search_scholarly_metadata",
                arguments={
                    "query_id": query["query_id"],
                    "source_name": source_name,
                },
            )
            for query in canonical["frontier_discovery"]["query_plan"]
            for source_name in query["source_names"]
        )
        candidate = {
            "candidate_id": "multi-session-adaptation",
            "method_family": "domain_adaptation",
            "pipeline_stages": ["session_adaptation", "models"],
            "claim": "Repeated sessions make adaptation a plausible research direction.",
            "applicability": ["multi-session EEG"],
            "limitations": ["Metadata evidence does not establish efficacy."],
            "supporting_papers": ["10.5000/crossref-paradigm_method_landscape"],
            "novelty_level": "frontier_for_local_registry",
            "future_protocol_requirements": ["Define target-data access downstream."],
            "proposed_validation": "A later Agent must run leakage-safe experiments.",
            "dataset_binding": {
                "dataset_id": "unseen-dataset",
                "dataset_profile_sha256": profile_hash,
                "supporting_profile_fields": ["sessions.sessions_per_subject"],
            },
        }
        return ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            id="frontier-status-before",
                            name="inspect_frontier_discovery_status",
                            arguments={},
                        ),
                    )
                ),
                ModelResponse(tool_calls=calls),
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            id="record-direction",
                            name="record_frontier_directions",
                            arguments={"candidates": [candidate]},
                        ),
                    )
                ),
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            id="frontier-status-after",
                            name="inspect_frontier_discovery_status",
                            arguments={},
                        ),
                    )
                ),
                ModelResponse(content="Literature discovery completed."),
            ]
        )

    def critic_provider(cycle: int) -> ScriptedProvider:
        review_path = run_dir / "cycles" / f"{cycle:04d}" / "dataset_level_draft.json"
        critique = {
            "schema_version": "1.0",
            "review_id": f"dataset-review-{cycle}",
            "dataset_id": "unseen-dataset",
            "reviewed_draft_sha256": sha256_path(review_path),
            "verdict": "pass",
            "findings": [],
            "required_revisions": [],
            "rationale": "The independent dataset-level review passed.",
        }
        return ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            id="read-draft",
                            name="read_dataset_critic_context",
                            arguments={},
                        ),
                    )
                ),
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            id="record-critique",
                            name="record_dataset_critique",
                            arguments={"critique": critique},
                        ),
                    )
                ),
                ModelResponse(content="Dataset Critic completed."),
            ]
        )

    agent = DatasetLevelAgent(
        run_id="dataset-agent-fixture",
        run_dir=run_dir,
        component_registry_path=Path("configs/component_registry.v0.json"),
        literature_provider_factory=literature_provider,
        critic_provider_factory=critic_provider,
        scholarly_source_factory=lambda _cycle: (
            _NamedSource("crossref"),
            _NamedSource("openalex"),
        ),
    )

    result = agent.run_from_profile(dataset_profile_path=profile_path)

    assert result.status == "completed"
    assert result.cycles == 1
    contract_path = Path(result.artifacts["dataset_level_contract"]["path"])
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["status"] == "frozen_dataset_level_contract"
    assert contract["stage_boundary"]["subject_data_accessed"] is False
    assert (run_dir / "dataset_level_run.json").is_file()


def test_top_level_dataset_agent_rejects_legacy_constraint_semantics(
    tmp_path: Path,
) -> None:
    profile = _profile_with_hashed_evidence(tmp_path)
    profile["constraints"] = {
        "allowed": [],
        "forbidden": [],
        "requires_human_decision": ["legacy mixed decision"],
    }
    profile_path = tmp_path / "legacy_profile.json"
    _write(profile_path, profile)
    agent = DatasetLevelAgent(
        run_id="legacy-profile-rejection",
        run_dir=tmp_path / "legacy-run",
        component_registry_path=Path("configs/component_registry.v0.json"),
        literature_provider_factory=lambda _cycle: (_ for _ in ()).throw(
            AssertionError("Literature provider must not be created")
        ),
        critic_provider_factory=lambda _cycle: (_ for _ in ()).throw(
            AssertionError("Critic provider must not be created")
        ),
    )

    result = agent.run_from_profile(dataset_profile_path=profile_path)

    assert result.status == "failed"
    assert "requires_human_decision" in (result.error or "")
    assert "canonical_search_space" not in result.artifacts
