from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bci_autodiscovery.agents.contracts import ModelResponse, ToolCall
from bci_autodiscovery.agents.evidence_reporter import (
    EvidenceReporterAgent,
    create_evidence_reporter_tools,
)
from bci_autodiscovery.agents.pipeline_lock_critic import (
    PipelineLockCriticAgent,
    create_pipeline_lock_critic_tools,
)
from bci_autodiscovery.agents.pipeline_search import create_pipeline_search_tools
from bci_autodiscovery.agents.providers import ScriptedProvider
from bci_autodiscovery.agents.runtime import AgentRuntime
from bci_autodiscovery.agents.scientific_critic import (
    ScientificCriticAgent,
    create_scientific_critic_tools,
)
from bci_autodiscovery.evaluation import (
    ConfirmationAccessError,
    OneShotConfirmationController,
)
from bci_autodiscovery.pipelines import DeterministicPipelineExecutor
from bci_autodiscovery.profiling.subject_measurements import EpochSession
from bci_autodiscovery.reporting import finalize_internal_evidence_report
from bci_autodiscovery.workflow.autonomy import sha256_path
from bci_autodiscovery.workflow.budget import BudgetLedger
from tests.fixtures.contracts import (
    autonomy_envelope,
    build_frozen_dataset_contract,
    minimal_dataset_profile,
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _session(session_id: str, seed: int) -> EpochSession:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.asarray([1, 2]), 24)
    sfreq = 128.0
    samples = 256
    time = np.arange(samples) / sfreq
    data = rng.normal(scale=0.8, size=(48, 6, samples))
    rhythm = np.sin(2 * np.pi * 11.0 * time)
    data[labels == 1, 1, :] += 1.25 * rhythm
    data[labels == 2, 4, :] += 1.25 * rhythm
    return EpochSession(
        subject_id="sub-confirmation",
        session_id=session_id,
        data=data,
        labels=labels,
        sampling_frequency_hz=sfreq,
        channel_names=("F3", "C3", "Cz", "C4", "P4", "Oz"),
        provenance={"path": f"synthetic-{session_id}", "sha256": f"sha-{session_id}"},
    )


def _pipeline(family: str, pipeline_id: str) -> dict:
    return {
        "pipeline_id": pipeline_id,
        "family": family,
        "bandpass_hz": [8.0, 30.0],
        "spatial_filter": "csp" if family == "csp_lda" else "none",
        "csp_components": 4 if family == "csp_lda" else 0,
        "feature": "csp_log_variance" if family == "csp_lda" else "log_bandpower",
        "model": "shrinkage_lda",
        "lda_shrinkage": 0.1,
        "cv_folds": 3,
        "random_seed": 11,
    }


def _contracts(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    fixture = build_frozen_dataset_contract(
        tmp_path / "dataset-level",
        profile=minimal_dataset_profile("confirm-fixture"),
    )
    dataset_path = fixture.profile_path
    envelope = autonomy_envelope(
        fixture.contract_path,
        dataset_id="confirm-fixture",
    )
    envelope["envelope_id"] = "confirm-envelope-v2"
    envelope["resource_budget"] = {
        "max_research_cycles": 4,
        "max_candidate_executions": 4,
        "max_compute_seconds": 1000,
        "max_api_tokens": 10000,
        "max_paid_cost": 0.0,
        "paid_cost_currency": "USD",
    }
    envelope["permissions"]["allow_network_literature"] = False
    envelope_path = tmp_path / "autonomy.json"
    _write(envelope_path, envelope)
    protocol_path = tmp_path / "protocol.json"
    _write(
        protocol_path,
        {
            "schema_version": "2.0",
            "protocol_id": "confirm-protocol-v1",
            "dataset_id": "confirm-fixture",
            "status": "frozen_autonomous",
            "data_roles": {
                "profiling_and_calibration": ["1"],
                "pipeline_search_and_lock": ["2"],
                "frozen_confirmation": ["3"],
            },
            "evaluation": {
                "primary_metric": "balanced_accuracy",
                "secondary_metrics": ["accuracy", "kappa"],
                "decision_policy": {
                    "chance_level": 0.5,
                    "minimum_confirmation_score": 0.6,
                    "maximum_search_to_confirmation_drop": 0.15,
                    "minimum_distinct_search_candidates": 2,
                    "success_requires_all_thresholds": True,
                    "below_chance_outcome": "refuse",
                    "otherwise_outcome": "inconclusive",
                    "confirmation_failure_outcome": "refuse",
                },
            },
            "individual_oracle": {"kind": "finite_individual_oracle"},
            "resource_budget": {
                "max_research_cycles": 4,
                "max_candidate_executions": 4,
                "max_compute_seconds": 1000,
                "max_api_tokens": 10000,
                "max_paid_cost": 0.0,
                "paid_cost_currency": "USD",
            },
            "stopping_conditions": ["evidence sufficient"],
        },
    )
    subject_path = tmp_path / "subject.json"
    _write(
        subject_path,
        {
            "schema_version": "1.0",
            "subject_id": "sub-confirmation",
            "status": "complete",
            "profile_complete": True,
            "search_implications": [],
        },
    )
    return dataset_path, envelope_path, protocol_path, subject_path


def _lock_and_pass_critique(
    tmp_path: Path,
) -> tuple[DeterministicPipelineExecutor, Path, Path, Path, Path]:
    _, envelope_path, protocol_path, subject_path = _contracts(tmp_path)
    executor = DeterministicPipelineExecutor(sessions=[_session("2", 30)])
    search_tools, _ = create_pipeline_search_tools(
        executor=executor,
        subject_profile_path=subject_path,
        frozen_protocol_path=protocol_path,
        autonomy_envelope_path=envelope_path,
        capability_registry_path=Path("configs/executable_pipeline_capabilities.v0.json"),
    )
    search_tools.execute("read_pipeline_search_context", {})
    first = search_tools.execute(
        "evaluate_pipeline_candidate",
        {"pipeline": _pipeline("bandpower_lda", "baseline")},
    )
    second = search_tools.execute(
        "evaluate_pipeline_candidate",
        {"pipeline": _pipeline("csp_lda", "profile-guided")},
    )
    selected = max(
        (first, second), key=lambda value: value["metrics"]["balanced_accuracy"]
    )
    lock = search_tools.execute(
        "lock_pipeline",
        {
            "selected_experiment_id": selected["experiment_id"],
            "evidence_experiment_ids": [first["experiment_id"], second["experiment_id"]],
            "selection_rationale": ["Two complete executable families were compared."],
            "rejected_alternatives": ["The alternative had weaker search evidence."],
            "uncertainty": ["Frozen confirmation performance is still unknown."],
            "stop_reason": "The planned informative comparison is complete.",
        },
    )
    lock_path = tmp_path / "pipeline_lock.json"
    _write(lock_path, lock)

    critic_tools, context = create_pipeline_lock_critic_tools(
        pipeline_lock_path=lock_path,
        subject_profile_path=subject_path,
        frozen_protocol_path=protocol_path,
        autonomy_envelope_path=envelope_path,
    )
    critique = {
        "schema_version": "1.0",
        "review_id": "lock-review-v1",
        "dataset_id": lock["dataset_id"],
        "subject_id": lock["subject_id"],
        "lock_id": lock["lock_id"],
        "reviewed_lock_sha256": sha256_path(lock_path),
        "verdict": "pass",
        "findings": [],
        "required_revisions": [],
        "rationale": "Lock is internally consistent, outcome blind, and within budget.",
    }
    provider = ScriptedProvider(
        [
            ModelResponse(tool_calls=(ToolCall("read", "read_pipeline_lock_critic_context", {}),)),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "record",
                        "record_pipeline_lock_critique",
                        {"critique": critique},
                    ),
                )
            ),
            ModelResponse(content="Lock passed independent outcome-blind review."),
        ]
    )
    result = PipelineLockCriticAgent(
        runtime=AgentRuntime(provider=provider, tools=critic_tools, run_id="lock-critic"),
        context=context,
    ).run()
    assert result.status == "completed"
    recorded = result.latest_tool_result("record_pipeline_lock_critique")
    critique_path = tmp_path / "lock_critique.json"
    _write(critique_path, recorded)
    return executor, lock_path, critique_path, protocol_path, envelope_path


def _controller(
    tmp_path: Path,
    *,
    loader,
    budget_ledger: BudgetLedger | None = None,
) -> OneShotConfirmationController:
    executor, lock_path, critique_path, protocol_path, envelope_path = (
        _lock_and_pass_critique(tmp_path)
    )
    return OneShotConfirmationController(
        search_executor=executor,
        confirmation_loader=loader,
        pipeline_lock_path=lock_path,
        lock_critique_path=critique_path,
        frozen_protocol_path=protocol_path,
        autonomy_envelope_path=envelope_path,
        access_record_path=tmp_path / "confirmation_access.json",
        confirmation_result_path=tmp_path / "confirmation_result.json",
        budget_ledger=budget_ledger,
    )


def test_one_shot_confirmation_is_bound_audited_and_never_refits(tmp_path: Path) -> None:
    calls = 0

    def load_confirmation() -> list[EpochSession]:
        nonlocal calls
        calls += 1
        return [_session("3", 31)]

    controller = _controller(tmp_path, loader=load_confirmation)
    result = controller.confirm()
    assert calls == 1
    assert result["status"] == "completed_one_shot"
    assert result["evaluation"]["fitting_performed_on_evaluation_data"] is False
    assert result["selection_or_refit_after_confirmation"] is False
    assert json.loads((tmp_path / "confirmation_access.json").read_text())["access_count"] == 1
    with pytest.raises(ConfirmationAccessError, match="already been accessed"):
        controller.confirm()
    assert calls == 1


def test_confirmation_access_and_compute_are_recorded_in_shared_budget(
    tmp_path: Path,
) -> None:
    ledger = BudgetLedger(
        tmp_path / "budget.jsonl",
        run_id="confirmation-budget",
        limits={
            "research_cycles": 4,
            "candidate_executions": 4,
            "compute_seconds": 100,
            "api_total_tokens": 1000,
            "paid_cost": 0,
            "provider_retries": 2,
            "recovery_attempts": 2,
            "confirmation_accesses": 1,
        },
        authority_sha256="fixture-authority",
        create=True,
    )
    controller = _controller(
        tmp_path,
        loader=lambda: [_session("3", 31)],
        budget_ledger=ledger,
    )
    controller.confirm()
    totals = ledger.totals
    assert totals["confirmation_accesses"] == 1
    assert totals["compute_seconds"] > 0


def test_nonpassing_or_unbound_critique_never_opens_confirmation(tmp_path: Path) -> None:
    calls = 0

    def load_confirmation() -> list[EpochSession]:
        nonlocal calls
        calls += 1
        return [_session("3", 31)]

    controller = _controller(tmp_path, loader=load_confirmation)
    critique = json.loads(controller.lock_critique_path.read_text(encoding="utf-8"))
    critique["reviewed_lock_sha256"] = "not-the-lock"
    _write(controller.lock_critique_path, critique)
    with pytest.raises(ConfirmationAccessError, match="Pipeline lock gate failed"):
        controller.confirm()
    assert calls == 0
    assert not controller.access_record_path.exists()


def test_failure_after_opening_consumes_access_and_forbids_retry(tmp_path: Path) -> None:
    calls = 0

    def failing_loader() -> list[EpochSession]:
        nonlocal calls
        calls += 1
        raise RuntimeError("fixture load failure")

    controller = _controller(tmp_path, loader=failing_loader)
    with pytest.raises(RuntimeError, match="fixture load failure"):
        controller.confirm()
    assert calls == 1
    failure = json.loads(controller.confirmation_result_path.read_text(encoding="utf-8"))
    assert failure["status"] == "confirmation_failed_after_access_consumed"
    assert failure["retry_allowed"] is False
    with pytest.raises(ConfirmationAccessError, match="already been accessed"):
        controller.confirm()
    assert calls == 1


def test_confirmed_evidence_is_reported_criticized_and_finalized(tmp_path: Path) -> None:
    controller = _controller(tmp_path, loader=lambda: [_session("3", 31)])
    confirmation = controller.confirm()
    lock = json.loads(controller.pipeline_lock_path.read_text(encoding="utf-8"))
    subject_path = Path(lock["source_contracts"]["subject_profile"]["path"])
    reporter_tools, reporter_context = create_evidence_reporter_tools(
        subject_profile_path=subject_path,
        pipeline_lock_path=controller.pipeline_lock_path,
        lock_critique_path=controller.lock_critique_path,
        confirmation_result_path=controller.confirmation_result_path,
        frozen_protocol_path=controller.frozen_protocol_path,
        autonomy_envelope_path=controller.autonomy_envelope_path,
    )
    decision = reporter_tools.execute("read_evidence_report_context", {})[
        "deterministic_decision"
    ]
    report = {
        "schema_version": "1.0",
        "report_id": "internal-report-v1",
        "status": "draft_for_scientific_critic",
        "dataset_id": lock["dataset_id"],
        "subject_id": lock["subject_id"],
        "protocol_id": lock["protocol_id"],
        "lock_id": lock["lock_id"],
        "conclusion": decision["outcome"],
        "headline": "The frozen decision policy was applied without post-confirmation tuning.",
        "claims": [
            {
                "claim_id": "confirmation-outcome",
                "statement": "The one-shot confirmation outcome matches the frozen decision rule.",
                "status": "supported",
                "evidence_refs": [
                    "confirmation_result#confirmation_score",
                    "deterministic_decision#outcome",
                    "frozen_protocol#evaluation.decision_policy",
                ],
            }
        ],
        "negative_results": lock["rejected_alternatives"],
        "uncertainties": ["This synthetic fixture is an engineering test, not a scientific result."],
        "outcome_summary": {
            "primary_metric": confirmation["primary_metric"],
            "search_score": confirmation["search_score"],
            "confirmation_score": confirmation["confirmation_score"],
            "confirmation_minus_search": confirmation["confirmation_minus_search"],
        },
        "research_cycle_efficiency": {
            "research_cycles": 2,
            "candidate_executions": 2,
            "authorized_maximum": 4,
            "unused_authorized_cycles": 2,
            "interpretation": "Two informative complete candidates were sufficient for this fixture.",
        },
        "reproducibility_notes": [
            "Every source artifact is bound by absolute path and SHA-256."
        ],
        "scope": "internal_evidence_report",
        "external_claim_authorized": False,
    }
    # Recreate the tools because each Reporter run requires its own context-first sequence.
    reporter_tools, reporter_context = create_evidence_reporter_tools(
        subject_profile_path=subject_path,
        pipeline_lock_path=controller.pipeline_lock_path,
        lock_critique_path=controller.lock_critique_path,
        confirmation_result_path=controller.confirmation_result_path,
        frozen_protocol_path=controller.frozen_protocol_path,
        autonomy_envelope_path=controller.autonomy_envelope_path,
    )
    reporter_provider = ScriptedProvider(
        [
            ModelResponse(tool_calls=(ToolCall("read-report", "read_evidence_report_context", {}),)),
            ModelResponse(
                tool_calls=(
                    ToolCall("write-report", "record_evidence_report", {"report": report}),
                )
            ),
            ModelResponse(content="Internal evidence report drafted."),
        ]
    )
    reporter_result = EvidenceReporterAgent(
        runtime=AgentRuntime(
            provider=reporter_provider,
            tools=reporter_tools,
            run_id="evidence-reporter",
        ),
        context=reporter_context,
    ).run()
    assert reporter_result.status == "completed"
    recorded_report = reporter_result.latest_tool_result("record_evidence_report")
    report_path = tmp_path / "evidence_report.json"
    _write(report_path, recorded_report)

    critic_tools, critic_context = create_scientific_critic_tools(
        evidence_report_path=report_path
    )
    critique = {
        "schema_version": "1.0",
        "review_id": "scientific-review-v1",
        "dataset_id": report["dataset_id"],
        "subject_id": report["subject_id"],
        "report_id": report["report_id"],
        "reviewed_report_sha256": sha256_path(report_path),
        "verdict": "pass",
        "findings": [],
        "required_revisions": [],
        "rationale": "Evidence, scope, negative results, uncertainty, and cycle counts are auditable.",
    }
    critic_provider = ScriptedProvider(
        [
            ModelResponse(tool_calls=(ToolCall("read-critic", "read_scientific_critic_context", {}),)),
            ModelResponse(
                tool_calls=(
                    ToolCall("write-critic", "record_scientific_critique", {"critique": critique}),
                )
            ),
            ModelResponse(content="Internal report passed scientific review."),
        ]
    )
    critic_result = ScientificCriticAgent(
        runtime=AgentRuntime(
            provider=critic_provider,
            tools=critic_tools,
            run_id="scientific-critic",
        ),
        context=critic_context,
    ).run()
    assert critic_result.status == "completed"
    recorded_critique = critic_result.latest_tool_result("record_scientific_critique")
    critique_path = tmp_path / "scientific_critique.json"
    _write(critique_path, recorded_critique)
    final_path = tmp_path / "final_internal_report.json"
    final = finalize_internal_evidence_report(
        evidence_report_path=report_path,
        scientific_critique_path=critique_path,
        final_report_path=final_path,
    )
    assert final["status"] == "finalized_internal_evidence_report"
    assert final["conclusion"] == decision["outcome"]
    assert final["external_claim_authorized"] is False
    assert final_path.is_file()
