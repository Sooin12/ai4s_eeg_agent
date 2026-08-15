from __future__ import annotations

import json
from pathlib import Path

import pytest

from bci_autodiscovery.agents.contracts import ModelResponse, ToolCall
from bci_autodiscovery.agents.audit import MemoryAuditSink
from bci_autodiscovery.agents.research_design_agent import ResearchDesignAgent
from bci_autodiscovery.agents.protocol_critic import (
    ProtocolCriticAgent,
    ProtocolCriticError,
    create_protocol_critic_tools,
)
from bci_autodiscovery.agents.providers import ScriptedProvider
from bci_autodiscovery.agents.research_protocol import (
    ResearchProtocolError,
    ResearchProtocolPlannerAgent,
    ResearchProtocolRevisionAgent,
    create_research_protocol_planner_tools,
    create_research_protocol_revision_tools,
    decision_requirements_from_profile,
    validate_research_protocol_proposal,
)
from bci_autodiscovery.agents.runtime import AgentRuntime
from bci_autodiscovery.agents.tools import ToolExecutionError
from bci_autodiscovery.workflow.autonomous_protocol import (
    AutonomousProtocolFreezeError,
    freeze_autonomous_protocol,
)
from bci_autodiscovery.workflow.protocol_loop import AutonomousProtocolLoop
from bci_autodiscovery.workflow.autonomy import (
    AutonomyEnvelopeError,
    sha256_path,
    validate_autonomy_envelope,
)
from bci_autodiscovery.workflow.budget import BudgetLedger, limits_from_envelope
from bci_autodiscovery.workflow.dataset_contract import load_dataset_level_contract
from bci_autodiscovery.workflow.research_contracts import (
    build_authoritative_unit_catalog,
    build_candidate_universe_contract,
    external_authority_requirements,
)
from tests.fixtures.contracts import (
    autonomy_envelope,
    build_frozen_dataset_contract,
    minimal_dataset_profile,
)


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _profile() -> dict:
    return {
        "schema_version": "1.0",
        "dataset": {"id": "autonomous-fixture"},
        "paradigm": {
            "family": "motor_imagery",
            "actions": [{"label": "left"}, {"label": "right"}],
        },
        "resting_state": {"present": False},
        "signal": {
            "channel_count": 2,
            "sampling_frequency_hz": 100,
            "modalities": ["EEG"],
        },
        "equipment": {},
        "events": {"common_analysis_window_s": [0, 1]},
        "sessions": {
            "session_indices": [1, 2, 3],
            "sessions_per_subject": 3,
        },
        "volume": {"trials": 60},
        "quality": {},
        "constraints": {
            "allowed": [],
            "forbidden": [],
            "requires_research_design_decision": [
                "policy for incomplete trials",
                "search versus confirmation role split",
            ],
            "external_authority_blockers": [],
        },
        "evidence": [{"source": "fixture-manifest"}],
    }


def _envelope(contract_path: Path) -> dict:
    return autonomy_envelope(contract_path, dataset_id="autonomous-fixture")


def _proposal(profile: dict, contract_path: Path) -> dict:
    contract = load_dataset_level_contract(contract_path)
    unit_catalog = build_authoritative_unit_catalog(profile)
    universe = build_candidate_universe_contract(
        contract,
        dataset_contract_sha256=sha256_path(contract_path),
    )
    decisions = [
        {
            "decision_id": item["decision_id"],
            "question": item["question"],
            "decision": "Apply a conservative deterministic policy before search.",
            "rationale": "The decision is outcome-blind and preserves confirmation isolation.",
            "evidence_refs": [item["source"], "fixture-manifest"],
            "confidence": 0.8,
        }
        for item in decision_requirements_from_profile(profile)
    ]
    units = unit_catalog["catalogs"]["session"]["unit_ids"]
    return {
        "schema_version": "3.0",
        "protocol_id": f"{profile['dataset']['id']}-protocol-v1",
        "dataset_id": profile["dataset"]["id"],
        "status": "proposed_for_autonomous_review",
        "split_unit": "session",
        "unit_catalog_sha256": unit_catalog["catalog_sha256"],
        "data_roles": {
            "profiling_and_calibration": [units[0]],
            "pipeline_search_and_lock": units[1:-1],
            "frozen_confirmation": [units[-1]],
        },
        "leakage_rules": {
            "confirmation_inaccessible_before_lock": True,
            "confirmation_cannot_select_pipeline": True,
            "confirmation_cannot_set_thresholds": True,
            "all_fitting_training_partition_only": True,
            "confirmation_access_once": True,
            "confirmation_cannot_reopen_search": True,
        },
        "evaluation": {
            "primary_metric": "balanced_accuracy",
            "secondary_metrics": ["accuracy", "cohen_kappa"],
            "aggregation": {
                "unit": "subject",
                "reducer": "macro_mean",
                "minimum_evaluable_units": 2,
                "missing_value_policy": "fail_closed",
            },
            "statistical_analysis": [
                {
                    "analysis_id": "paired-primary-v1",
                    "test_id": "paired_permutation",
                    "comparison": "selected_pipeline_vs_frozen_baseline",
                    "alternative": "greater",
                    "alpha": 0.05,
                    "permutations": 999,
                    "random_seed": 1701,
                    "multiple_comparison": {
                        "method": "none",
                        "family_id": "primary-confirmation-v1",
                    },
                }
            ],
            "confidence_interval": {
                "method": "bootstrap_percentile",
                "level": 0.95,
                "resamples": 999,
                "random_seed": 1702,
            },
            "decision_policy": {
                "policy_version": "2.0",
                "chance_level": 0.5,
                "minimum_confirmation_score": 0.6,
                "maximum_search_to_confirmation_drop": 0.15,
                "minimum_distinct_search_candidates": 2,
                "minimum_evaluable_units": 2,
                "success_requires_all_thresholds": True,
                "below_chance_outcome": "refuse",
                "otherwise_outcome": "inconclusive",
                "confirmation_failure_outcome": "refuse",
            },
        },
        "individual_oracle": {
            "kind": "finite_individual_oracle",
            "candidate_universe": {
                key: universe[key]
                for key in (
                    "schema_version",
                    "source",
                    "source_contract_sha256",
                    "selector_rule_id",
                    "candidate_universe_sha256",
                    "frontier_semantics",
                    "materialization_gate",
                )
            },
            "selection_data_role": "pipeline_search_and_lock",
            "confirmation_use_forbidden": True,
        },
        "resource_budget": {
            "max_research_cycles": 8,
            "max_candidate_executions": 16,
            "max_compute_seconds": 1800,
            "max_api_tokens": 50000,
            "max_paid_cost": 5.0,
            "paid_cost_currency": "USD",
        },
        "stopping_policy": {
            "policy_version": "1.0",
            "stop_on_budget_exhaustion": True,
            "stop_on_candidate_universe_exhaustion": True,
            "plateau": {
                "enabled": True,
                "minimum_candidates": 2,
                "patience": 2,
                "minimum_improvement": 0.005,
            },
        },
        "autonomous_decisions": decisions,
        "quality_anomaly_policy": {
            "policy_version": "1.0",
            "default_action": "retain_and_flag",
            "unknown_metadata_action": "block_dependent_operation",
            "exclusions_require_audit_record": True,
            "rules": [
                {
                    "rule_id": "missing-trial-fixture-v1",
                    "applies_to": "run",
                    "predicate": {
                        "field": "quality.missing_trials",
                        "operator": "gt",
                        "value": 0,
                    },
                    "action": "retain_and_flag",
                    "reason_code": "incomplete_trials_observed",
                    "evidence_refs": ["DatasetProfile.quality"],
                }
            ],
        },
        "execution_preconditions": {
            "external_authority_blockers": external_authority_requirements(contract),
        },
        "rationale": ["the split is longitudinal and outcome-blind"],
        "alternatives_considered": ["random role assignment rejected for temporal ambiguity"],
        "risks": ["cross-session drift may reduce confirmation performance"],
        "unresolved_blockers": [],
    }


def _contracts(tmp_path: Path) -> tuple[Path, Path, Path, dict, dict]:
    fixture = build_frozen_dataset_contract(
        tmp_path / "dataset-level",
        profile=_profile(),
    )
    profile = json.loads(fixture.profile_path.read_text(encoding="utf-8"))
    envelope = _envelope(fixture.contract_path)
    envelope_path = tmp_path / "autonomy.json"
    _write(envelope_path, envelope)
    return (
        fixture.contract_path,
        fixture.profile_path,
        envelope_path,
        profile,
        envelope,
    )


def _bind_proposal(
    proposal: dict,
    *,
    contract_path: Path,
    envelope_path: Path,
    envelope: dict,
) -> dict:
    contract = load_dataset_level_contract(contract_path)
    proposal["activation_state"] = {
        "protocol_frozen": False,
        "session_role_contract_activated": False,
        "raw_data_accessed": False,
        "confirmation_accessed": False,
        "pipeline_execution_started": False,
    }
    proposal["dataset_level_contract"] = {
        "path": str(contract_path.resolve()),
        "sha256": sha256_path(contract_path),
        "contract_id": contract["contract_id"],
    }
    proposal["autonomy_envelope"] = {
        "path": str(envelope_path.resolve()),
        "sha256": sha256_path(envelope_path),
        "envelope_id": envelope["envelope_id"],
    }
    return proposal


def test_autonomy_envelope_and_protocol_budget_fail_closed(tmp_path: Path) -> None:
    contract_path, _, _, profile, envelope = _contracts(tmp_path)
    validate_autonomy_envelope(envelope)
    proposal = _proposal(profile, contract_path)
    proposal["resource_budget"]["max_research_cycles"] = 13
    with pytest.raises(ResearchProtocolError, match="exceeds authorized limit"):
        validate_research_protocol_proposal(
            proposal,
            dataset_contract=load_dataset_level_contract(contract_path),
            profile=profile,
            envelope=envelope,
        )
    envelope["confirmation_policy"]["max_access_count"] = 2
    with pytest.raises(AutonomyEnvelopeError, match="must equal one"):
        validate_autonomy_envelope(envelope)


def test_planner_resolves_every_profile_decision_without_human_approval(
    tmp_path: Path,
) -> None:
    contract_path, _, envelope_path, profile, _ = _contracts(tmp_path)
    proposal = _proposal(profile, contract_path)
    tools, context = create_research_protocol_planner_tools(
        dataset_level_contract_path=contract_path,
        autonomy_envelope_path=envelope_path,
    )
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall("read", "read_autonomous_research_context", {}),
                )
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "record",
                        "record_research_protocol_proposal",
                        {"proposal": proposal},
                    ),
                )
            ),
            ModelResponse(content="Protocol recorded for autonomous critic review."),
        ]
    )
    result = ResearchProtocolPlannerAgent(
        runtime=AgentRuntime(provider=provider, tools=tools, run_id="planner-loop"),
        context=context,
    ).run()
    recorded = result.latest_tool_result("record_research_protocol_proposal")
    assert result.status == "completed"
    assert recorded["status"] == "proposed_for_autonomous_review"
    assert recorded["activation_state"]["protocol_frozen"] is False
    assert len(recorded["autonomous_decisions"]) == 2


def test_critic_pass_and_deterministic_freeze_complete_first_loop(tmp_path: Path) -> None:
    contract_path, _, envelope_path, profile, envelope = _contracts(tmp_path)
    proposal = _bind_proposal(
        _proposal(profile, contract_path),
        contract_path=contract_path,
        envelope_path=envelope_path,
        envelope=envelope,
    )
    proposal_path = tmp_path / "proposal.json"
    _write(proposal_path, proposal)
    critique = {
        "schema_version": "2.0",
        "review_id": "critic-review-v1",
        "dataset_id": proposal["dataset_id"],
        "protocol_id": proposal["protocol_id"],
        "reviewed_protocol_sha256": sha256_path(proposal_path),
        "verdict": "pass",
        "findings": [
            {
                "code": "outcome_blind_design",
                "severity": "note",
                "owner": "none",
                "message": "The design is outcome-blind and confirmation-isolated.",
                "evidence_refs": ["proposal.leakage_rules"],
            }
        ],
        "required_revisions": [],
        "rationale": "All deterministic and semantic checks passed.",
    }
    tools, context = create_protocol_critic_tools(
        dataset_level_contract_path=contract_path,
        autonomy_envelope_path=envelope_path,
        proposal_path=proposal_path,
    )
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(ToolCall("read", "read_protocol_critic_context", {}),)
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "critic",
                        "record_protocol_critique",
                        {"critique": critique},
                    ),
                )
            ),
            ModelResponse(content="Independent protocol critique passed."),
        ]
    )
    result = ProtocolCriticAgent(
        runtime=AgentRuntime(provider=provider, tools=tools, run_id="critic-loop"),
        context=context,
    ).run()
    recorded = result.latest_tool_result("record_protocol_critique")
    assert result.status == "completed"
    assert recorded["verdict"] == "pass"
    critique_path = tmp_path / "critique.json"
    _write(critique_path, recorded)
    frozen_path = tmp_path / "frozen_protocol.json"
    frozen = freeze_autonomous_protocol(
        proposal_path=proposal_path,
        critique_path=critique_path,
        dataset_level_contract_path=contract_path,
        autonomy_envelope_path=envelope_path,
        expected_dataset_level_contract_sha256=sha256_path(contract_path),
        expected_autonomy_envelope_sha256=sha256_path(envelope_path),
        expected_proposal_sha256=sha256_path(proposal_path),
        expected_critique_sha256=sha256_path(critique_path),
        output_path=frozen_path,
    )
    assert frozen["status"] == "frozen_autonomous"
    assert frozen["activation_state"] == {
        "protocol_frozen": True,
        "session_role_contract_activated": True,
        "raw_data_accessed": False,
        "confirmation_accessed": False,
        "pipeline_execution_started": False,
    }
    assert frozen["autonomous_freeze"]["human_itemized_approval_used"] is False
    assert frozen["autonomous_freeze"]["dataset_level_contract"]["sha256"] == (
        sha256_path(contract_path)
    )
    assert frozen["autonomous_freeze"]["autonomy_envelope"]["sha256"] == (
        sha256_path(envelope_path)
    )
    assert frozen["autonomous_freeze"]["proposal"]["sha256"] == sha256_path(
        proposal_path
    )
    assert frozen["autonomous_freeze"]["critique"]["sha256"] == sha256_path(
        critique_path
    )
    assert frozen_path.is_file()


def test_freeze_rejects_changed_contract_envelope_proposal_or_critique_sha(
    tmp_path: Path,
) -> None:
    contract_path, _, envelope_path, profile, envelope = _contracts(tmp_path)
    proposal = _bind_proposal(
        _proposal(profile, contract_path),
        contract_path=contract_path,
        envelope_path=envelope_path,
        envelope=envelope,
    )
    proposal_path = tmp_path / "proposal-for-tamper.json"
    _write(proposal_path, proposal)
    critique = {
        "schema_version": "2.0",
        "review_id": "tamper-review",
        "dataset_id": proposal["dataset_id"],
        "protocol_id": proposal["protocol_id"],
        "reviewed_protocol_sha256": sha256_path(proposal_path),
        "verdict": "pass",
        "findings": [],
        "required_revisions": [],
        "rationale": "Valid before the simulated post-stage mutation.",
        "source_proposal": {
            "path": str(proposal_path.resolve()),
            "sha256": sha256_path(proposal_path),
        },
    }
    critique_path = tmp_path / "critique-for-tamper.json"
    _write(critique_path, critique)
    expected = {
        "expected_dataset_level_contract_sha256": sha256_path(contract_path),
        "expected_autonomy_envelope_sha256": sha256_path(envelope_path),
        "expected_proposal_sha256": sha256_path(proposal_path),
        "expected_critique_sha256": sha256_path(critique_path),
    }
    cases = {
        "expected_dataset_level_contract_sha256": "DatasetLevelContract",
        "expected_autonomy_envelope_sha256": "AutonomyEnvelope",
        "expected_proposal_sha256": "proposal",
        "expected_critique_sha256": "critique",
    }
    for field, label in cases.items():
        hashes = dict(expected)
        hashes[field] = "0" * 64
        with pytest.raises(AutonomousProtocolFreezeError, match=label):
            freeze_autonomous_protocol(
                proposal_path=proposal_path,
                critique_path=critique_path,
                dataset_level_contract_path=contract_path,
                autonomy_envelope_path=envelope_path,
                output_path=tmp_path / f"must-not-freeze-{field}.json",
                **hashes,
            )


def test_missing_decision_and_false_critic_pass_are_blocked(tmp_path: Path) -> None:
    contract_path, _, envelope_path, profile, envelope = _contracts(tmp_path)
    proposal = _proposal(profile, contract_path)
    proposal["autonomous_decisions"] = proposal["autonomous_decisions"][:-1]
    with pytest.raises(ResearchProtocolError, match="missing="):
        validate_research_protocol_proposal(
            proposal,
            dataset_contract=load_dataset_level_contract(contract_path),
            profile=profile,
            envelope=envelope,
        )
    _bind_proposal(
        proposal,
        contract_path=contract_path,
        envelope_path=envelope_path,
        envelope=envelope,
    )
    proposal_path = tmp_path / "invalid_proposal.json"
    _write(proposal_path, proposal)
    tools, _ = create_protocol_critic_tools(
        dataset_level_contract_path=contract_path,
        autonomy_envelope_path=envelope_path,
        proposal_path=proposal_path,
    )
    tools.execute("read_protocol_critic_context", {})
    false_pass = {
        "schema_version": "2.0",
        "review_id": "false-pass",
        "dataset_id": proposal["dataset_id"],
        "protocol_id": proposal["protocol_id"],
        "reviewed_protocol_sha256": sha256_path(proposal_path),
        "verdict": "pass",
        "findings": [],
        "required_revisions": [],
        "rationale": "incorrect pass",
    }
    with pytest.raises(ToolExecutionError, match="deterministically invalid"):
        tools.execute("record_protocol_critique", {"critique": false_pass})
    with pytest.raises(AutonomousProtocolFreezeError, match="Refusing to overwrite"):
        existing = tmp_path / "existing.json"
        existing.write_text("{}", encoding="utf-8")
        freeze_autonomous_protocol(
            proposal_path=proposal_path,
            critique_path=existing,
            dataset_level_contract_path=contract_path,
            autonomy_envelope_path=envelope_path,
            expected_dataset_level_contract_sha256=sha256_path(contract_path),
            expected_autonomy_envelope_sha256=sha256_path(envelope_path),
            expected_proposal_sha256=sha256_path(proposal_path),
            expected_critique_sha256=sha256_path(existing),
            output_path=existing,
        )


def test_revise_verdict_returns_to_planner_without_user_decision(tmp_path: Path) -> None:
    contract_path, _, envelope_path, profile, envelope = _contracts(tmp_path)
    proposal = _bind_proposal(
        _proposal(profile, contract_path),
        contract_path=contract_path,
        envelope_path=envelope_path,
        envelope=envelope,
    )
    proposal_path = tmp_path / "proposal-v1.json"
    _write(proposal_path, proposal)
    critique = {
        "schema_version": "2.0",
        "review_id": "critic-revise-v1",
        "dataset_id": proposal["dataset_id"],
        "protocol_id": proposal["protocol_id"],
        "reviewed_protocol_sha256": sha256_path(proposal_path),
        "verdict": "revise",
        "findings": [
            {
                "code": "weak_refusal_rule",
                "severity": "major",
                "owner": "research_protocol_planner",
                "message": "Make the refusal rule operational.",
                "evidence_refs": ["proposal.evaluation.decision_policy"],
            }
        ],
        "required_revisions": [
            {
                "finding_code": "weak_refusal_rule",
                "instruction": "Add an operational no-strong-conclusion rule.",
            }
        ],
        "rationale": "The design is repairable without human input.",
        "activation_state": {
            "protocol_frozen": False,
            "session_role_contract_activated": False,
            "raw_data_accessed": False,
            "confirmation_accessed": False,
            "pipeline_execution_started": False,
        },
        "source_proposal": {
            "path": str(proposal_path.resolve()),
            "sha256": sha256_path(proposal_path),
        },
    }
    critique_path = tmp_path / "critique-v1.json"
    _write(critique_path, critique)
    revised = _proposal(profile, contract_path)
    revised["evaluation"]["decision_policy"]["minimum_confirmation_score"] = 0.61
    tools, context = create_research_protocol_revision_tools(
        dataset_level_contract_path=contract_path,
        autonomy_envelope_path=envelope_path,
        proposal_path=proposal_path,
        critique_path=critique_path,
    )
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(ToolCall("read", "read_protocol_revision_context", {}),)
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "revise",
                        "record_revised_research_protocol",
                        {
                            "revised_proposal": revised,
                            "revision_summary": ["Made refusal criterion operational."],
                            "critic_resolution": ["Resolved weak_refusal_rule."],
                        },
                    ),
                )
            ),
            ModelResponse(content="Revised protocol recorded for a fresh Critic turn."),
        ]
    )
    result = ResearchProtocolRevisionAgent(
        runtime=AgentRuntime(provider=provider, tools=tools, run_id="revision-loop"),
        context=context,
    ).run()
    recorded = result.latest_tool_result("record_revised_research_protocol")
    assert result.status == "completed"
    assert recorded["activation_state"]["protocol_frozen"] is False
    assert recorded["autonomous_revision"]["parent_sha256"] == sha256_path(proposal_path)
    assert recorded["evaluation"]["decision_policy"]["minimum_confirmation_score"] == 0.61


def test_orchestrator_revises_then_freezes_without_human_pause(tmp_path: Path) -> None:
    contract_path, _, envelope_path, profile, envelope = _contracts(tmp_path)

    def bound_proposal() -> dict:
        value = _proposal(profile, contract_path)
        return _bind_proposal(
            value,
            contract_path=contract_path,
            envelope_path=envelope_path,
            envelope=envelope,
        )

    def planner() -> dict:
        return bound_proposal()

    def critic(proposal_path: Path, cycle: int) -> dict:
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        verdict = "revise" if cycle == 1 else "pass"
        return {
            "schema_version": "2.0",
            "review_id": f"critic-{cycle}",
            "dataset_id": proposal["dataset_id"],
            "protocol_id": proposal["protocol_id"],
            "reviewed_protocol_sha256": sha256_path(proposal_path),
            "verdict": verdict,
            "findings": (
                [
                    {
                        "code": "make_rule_operational",
                        "severity": "major",
                        "owner": "research_protocol_planner",
                        "message": "Make refusal operational.",
                        "evidence_refs": ["evaluation.decision_policy"],
                    }
                ]
                if verdict == "revise"
                else []
            ),
            "required_revisions": (
                [
                    {
                        "finding_code": "make_rule_operational",
                        "instruction": "Make refusal operational.",
                    }
                ]
                if verdict == "revise"
                else []
            ),
            "rationale": "Revise once, then pass the corrected outcome-blind design.",
            "activation_state": {
                "protocol_frozen": False,
                "session_role_contract_activated": False,
                "raw_data_accessed": False,
                "confirmation_accessed": False,
                "pipeline_execution_started": False,
            },
            "source_proposal": {
                "path": str(proposal_path.resolve()),
                "sha256": sha256_path(proposal_path),
            },
        }

    def reviser(proposal_path: Path, critique_path: Path, cycle: int) -> dict:
        del critique_path, cycle
        value = json.loads(proposal_path.read_text(encoding="utf-8"))
        value["evaluation"]["decision_policy"]["minimum_confirmation_score"] = 0.61
        return value

    loop = AutonomousProtocolLoop(
        run_dir=tmp_path / "protocol-loop",
        dataset_level_contract_path=contract_path,
        autonomy_envelope_path=envelope_path,
        max_revision_cycles=2,
    )
    result = loop.run(planner=planner, critic=critic, reviser=reviser)
    assert result.status == "completed"
    assert result.cycles == 2
    assert result.frozen_protocol_path is not None
    assert Path(result.frozen_protocol_path).is_file()
    assert (tmp_path / "protocol-loop" / "protocol_loop_state.json").is_file()


def test_production_research_design_agent_accounts_and_freezes(tmp_path: Path) -> None:
    contract_path, _, envelope_path, profile, envelope = _contracts(tmp_path)
    proposal = _proposal(profile, contract_path)
    run_dir = tmp_path / "production-agent"
    ledger = BudgetLedger(
        run_dir / "budget_ledger.jsonl",
        run_id="production-agent",
        limits=limits_from_envelope(envelope),
        authority_sha256=sha256_path(envelope_path),
        create=True,
    )

    def provider_factory(stage: str, cycle: int):
        assert cycle == 1
        if stage == "planner":
            return ScriptedProvider(
                [
                    ModelResponse(
                        tool_calls=(
                            ToolCall("read", "read_autonomous_research_context", {}),
                        )
                    ),
                    ModelResponse(
                        tool_calls=(
                            ToolCall(
                                "record",
                                "record_research_protocol_proposal",
                                {"proposal": proposal},
                            ),
                        )
                    ),
                    ModelResponse(content="planner complete"),
                ]
            )
        proposal_path = run_dir / "proposal-0001.json"
        recorded = json.loads(proposal_path.read_text(encoding="utf-8"))
        critique = {
            "schema_version": "2.0",
            "review_id": "production-pass",
            "dataset_id": recorded["dataset_id"],
            "protocol_id": recorded["protocol_id"],
            "reviewed_protocol_sha256": sha256_path(proposal_path),
            "verdict": "pass",
            "findings": [],
            "required_revisions": [],
            "rationale": "All deterministic and independent semantic checks passed.",
        }
        return ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(ToolCall("read", "read_protocol_critic_context", {}),)
                ),
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            "record",
                            "record_protocol_critique",
                            {"critique": critique},
                        ),
                    )
                ),
                ModelResponse(content="critic complete"),
            ]
        )

    result = ResearchDesignAgent(
        run_id="production-agent",
        run_dir=run_dir,
        dataset_level_contract_path=contract_path,
        autonomy_envelope_path=envelope_path,
        provider_factory=provider_factory,
        budget_ledger=ledger,
        pricing=None,
        audit=MemoryAuditSink(),
        max_revision_cycles=1,
    ).run()
    assert result.status == "completed"
    assert Path(result.artifacts["frozen_protocol"]["path"]).is_file()
    state = json.loads(
        (run_dir / "research_design_state.json").read_text(encoding="utf-8")
    )
    assert state["status"] == "completed"
    assert state["budget"]["event_count"] >= 8
    assert (run_dir / "research_design_run.json").is_file()


def test_research_design_keeps_valid_terminal_tool_artifact_when_epilogue_fails(
    tmp_path: Path,
) -> None:
    contract_path, _, envelope_path, profile, envelope = _contracts(tmp_path)
    proposal = _proposal(profile, contract_path)
    run_dir = tmp_path / "post-tool-provider-failure"
    ledger = BudgetLedger(
        run_dir / "budget_ledger.jsonl",
        run_id="post-tool-provider-failure",
        limits=limits_from_envelope(envelope),
        authority_sha256=sha256_path(envelope_path),
        create=True,
    )
    audit = MemoryAuditSink()

    def provider_factory(stage: str, cycle: int):
        assert cycle == 1
        if stage == "planner":
            # No third response: the optional natural-language epilogue fails after
            # the validated terminal tool has already accepted the protocol.
            return ScriptedProvider(
                [
                    ModelResponse(
                        tool_calls=(
                            ToolCall("read", "read_autonomous_research_context", {}),
                        )
                    ),
                    ModelResponse(
                        tool_calls=(
                            ToolCall(
                                "record",
                                "record_research_protocol_proposal",
                                {"proposal": proposal},
                            ),
                        )
                    ),
                ]
            )
        proposal_path = run_dir / "proposal-0001.json"
        recorded = json.loads(proposal_path.read_text(encoding="utf-8"))
        return ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(ToolCall("read", "read_protocol_critic_context", {}),)
                ),
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            "record",
                            "record_protocol_critique",
                            {
                                "critique": {
                                    "schema_version": "2.0",
                                    "review_id": "post-tool-failure-pass",
                                    "dataset_id": recorded["dataset_id"],
                                    "protocol_id": recorded["protocol_id"],
                                    "reviewed_protocol_sha256": sha256_path(proposal_path),
                                    "verdict": "pass",
                                    "findings": [],
                                    "required_revisions": [],
                                    "rationale": "The valid terminal proposal remains reviewable.",
                                }
                            },
                        ),
                    )
                ),
                ModelResponse(content="critic complete"),
            ]
        )

    result = ResearchDesignAgent(
        run_id="post-tool-provider-failure",
        run_dir=run_dir,
        dataset_level_contract_path=contract_path,
        autonomy_envelope_path=envelope_path,
        provider_factory=provider_factory,
        budget_ledger=ledger,
        pricing=None,
        audit=audit,
        max_revision_cycles=1,
    ).run()
    assert result.status == "completed"
    assert (run_dir / "proposal-0001.json").is_file()
    assert any(
        event["event_type"] == "terminal_tool_artifact_recovered"
        for event in audit.events
    )


def test_research_design_agent_recovers_without_overwriting_checkpoints(
    tmp_path: Path,
) -> None:
    contract_path, _, envelope_path, profile, envelope = _contracts(tmp_path)
    proposal = _proposal(profile, contract_path)
    run_dir = tmp_path / "recoverable-agent"
    ledger_path = run_dir / "budget_ledger.jsonl"
    ledger = BudgetLedger(
        ledger_path,
        run_id="recoverable-agent",
        limits=limits_from_envelope(envelope),
        authority_sha256=sha256_path(envelope_path),
        create=True,
    )
    failed = ResearchDesignAgent(
        run_id="recoverable-agent",
        run_dir=run_dir,
        dataset_level_contract_path=contract_path,
        autonomy_envelope_path=envelope_path,
        provider_factory=lambda _stage, _cycle: ScriptedProvider([]),
        budget_ledger=ledger,
        pricing=None,
        audit=MemoryAuditSink(),
    ).run()
    assert failed.status == "failed_recoverable"
    assert not (run_dir / "proposal-0001.json").exists()

    resumed_ledger = BudgetLedger(
        ledger_path,
        run_id="recoverable-agent",
        limits=limits_from_envelope(envelope),
        authority_sha256=sha256_path(envelope_path),
        create=False,
    )

    def recovered_provider(stage: str, cycle: int):
        assert cycle == 1
        if stage == "planner":
            return ScriptedProvider(
                [
                    ModelResponse(
                        tool_calls=(
                            ToolCall("read", "read_autonomous_research_context", {}),
                        )
                    ),
                    ModelResponse(
                        tool_calls=(
                            ToolCall(
                                "record",
                                "record_research_protocol_proposal",
                                {"proposal": proposal},
                            ),
                        )
                    ),
                    ModelResponse(content="recovered planner complete"),
                ]
            )
        proposal_path = run_dir / "proposal-0001.json"
        recorded = json.loads(proposal_path.read_text(encoding="utf-8"))
        return ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(ToolCall("read", "read_protocol_critic_context", {}),)
                ),
                ModelResponse(
                    tool_calls=(
                        ToolCall(
                            "record",
                            "record_protocol_critique",
                            {
                                "critique": {
                                    "schema_version": "2.0",
                                    "review_id": "recovery-pass",
                                    "dataset_id": recorded["dataset_id"],
                                    "protocol_id": recorded["protocol_id"],
                                    "reviewed_protocol_sha256": sha256_path(
                                        proposal_path
                                    ),
                                    "verdict": "pass",
                                    "findings": [],
                                    "required_revisions": [],
                                    "rationale": "Recovered run passed independent review.",
                                }
                            },
                        ),
                    )
                ),
                ModelResponse(content="recovered critic complete"),
            ]
        )

    recovered = ResearchDesignAgent(
        run_id="recoverable-agent",
        run_dir=run_dir,
        dataset_level_contract_path=contract_path,
        autonomy_envelope_path=envelope_path,
        provider_factory=recovered_provider,
        budget_ledger=resumed_ledger,
        pricing=None,
        audit=MemoryAuditSink(),
    ).run(resume=True)
    assert recovered.status == "completed"
    assert resumed_ledger.totals["recovery_attempts"] == 1


@pytest.mark.parametrize(
    ("dataset_id", "sessions"),
    [
        ("unseen-three-session-dataset", (10, 20, 30)),
        ("unseen-six-session-dataset", (1, 2, 3, 4, 5, 6)),
    ],
)
def test_protocol_contract_is_dataset_neutral_across_authoritative_session_catalogs(
    tmp_path: Path,
    dataset_id: str,
    sessions: tuple[int, ...],
) -> None:
    fixture = build_frozen_dataset_contract(
        tmp_path / dataset_id,
        profile=minimal_dataset_profile(dataset_id, session_indices=sessions),
    )
    profile = json.loads(fixture.profile_path.read_text(encoding="utf-8"))
    envelope = autonomy_envelope(fixture.contract_path, dataset_id=dataset_id)
    proposal = _proposal(profile, fixture.contract_path)
    validate_research_protocol_proposal(
        proposal,
        dataset_contract=load_dataset_level_contract(fixture.contract_path),
        profile=profile,
        envelope=envelope,
    )
    assigned = {
        unit
        for values in proposal["data_roles"].values()
        for unit in values
    }
    assert assigned == {str(value) for value in sessions}
