from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bci_autodiscovery.agents.contracts import ModelResponse, ToolCall
from bci_autodiscovery.agents.pipeline_search import (
    PipelineSearchAgent,
    create_pipeline_search_tools,
)
from bci_autodiscovery.agents.providers import ScriptedProvider
from bci_autodiscovery.agents.runtime import AgentRuntime
from bci_autodiscovery.agents.tools import ToolExecutionError
from bci_autodiscovery.pipelines import DeterministicPipelineExecutor
from bci_autodiscovery.profiling.subject_measurements import EpochSession
from bci_autodiscovery.workflow.autonomy import sha256_path
from tests.fixtures.contracts import (
    autonomy_envelope,
    build_frozen_dataset_contract,
    minimal_dataset_profile,
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _search_session() -> EpochSession:
    rng = np.random.default_rng(66)
    labels = np.repeat(np.asarray([1, 2]), 30)
    sfreq = 128.0
    samples = 256
    time = np.arange(samples) / sfreq
    data = rng.normal(scale=0.8, size=(60, 6, samples))
    rhythm = np.sin(2 * np.pi * 11.0 * time)
    data[labels == 1, 1, :] += 1.3 * rhythm
    data[labels == 2, 4, :] += 1.3 * rhythm
    return EpochSession(
        subject_id="sub-001",
        session_id="2",
        data=data,
        labels=labels,
        sampling_frequency_hz=sfreq,
        channel_names=("F3", "C3", "Cz", "C4", "P4", "Oz"),
        provenance={"path": "synthetic-search", "sha256": "search-session-2"},
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
        "random_seed": 5,
    }


def _contracts(tmp_path: Path) -> tuple[Path, Path, Path]:
    profile = minimal_dataset_profile("search-fixture")
    profile["signal"] = {
        "channel_count": 6,
        "sampling_frequency_hz": 128,
        "modalities": ["EEG"],
    }
    profile["events"] = {"common_analysis_window_s": [0, 2]}
    profile["volume"] = {"trials": 180}
    fixture = build_frozen_dataset_contract(
        tmp_path / "dataset-level",
        profile=profile,
    )
    envelope = autonomy_envelope(
        fixture.contract_path,
        dataset_id="search-fixture",
    )
    envelope["envelope_id"] = "search-envelope-v2"
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
    protocol_path = tmp_path / "frozen_protocol.json"
    _write(
        protocol_path,
        {
            "schema_version": "2.0",
            "protocol_id": "search-protocol-v1",
            "dataset_id": "search-fixture",
            "status": "frozen_autonomous",
            "split_unit": "session",
            "data_roles": {
                "profiling_and_calibration": ["1"],
                "pipeline_search_and_lock": ["2"],
                "frozen_confirmation": ["3"],
            },
            "evaluation": {
                "primary_metric": "balanced_accuracy",
                "secondary_metrics": ["accuracy", "kappa"],
                "statistical_tests": ["subject-level permutation"],
                "aggregation": "subject",
                "success_criteria": ["frozen criterion"],
                "refusal_criteria": ["frozen refusal"],
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
            "individual_oracle": {
                "kind": "finite_individual_oracle",
                "candidate_universe": "frozen executable set",
                "selection_data_role": "pipeline_search_and_lock",
                "confirmation_use_forbidden": True,
            },
            "resource_budget": {
                "max_research_cycles": 4,
                "max_candidate_executions": 4,
                "max_compute_seconds": 1000,
                "max_api_tokens": 10000,
                "max_paid_cost": 0.0,
                "paid_cost_currency": "USD",
            },
            "stopping_conditions": ["evidence sufficient", "budget exhausted"],
        },
    )
    subject_path = tmp_path / "subject_profile.json"
    _write(
        subject_path,
        {
            "schema_version": "1.0",
            "subject_id": "sub-001",
            "status": "complete",
            "profile_complete": True,
            "search_implications": [
                {
                    "target_stage": "features",
                    "recommendation": "Compare CSP with bandpower.",
                    "evidence_measurement_ids": ["measurement-1"],
                    "status": "proposed_for_budgeted_search",
                }
            ],
        },
    )
    return subject_path, protocol_path, envelope_path


def test_agent_evaluates_distinct_candidates_and_locks_evidence(tmp_path: Path) -> None:
    subject_path, protocol_path, envelope_path = _contracts(tmp_path)
    capability_path = Path("configs/executable_pipeline_capabilities.v0.json").resolve()
    bandpower = _pipeline("bandpower_lda", "bandpower-baseline")
    csp = _pipeline("csp_lda", "profile-guided-csp")
    expected_executor = DeterministicPipelineExecutor(sessions=[_search_session()])
    bandpower_result = expected_executor.evaluate(bandpower)
    csp_result = expected_executor.evaluate(csp)
    selected = max(
        [bandpower_result, csp_result],
        key=lambda item: item["metrics"]["balanced_accuracy"],
    )
    tools, context = create_pipeline_search_tools(
        executor=DeterministicPipelineExecutor(sessions=[_search_session()]),
        subject_profile_path=subject_path,
        frozen_protocol_path=protocol_path,
        autonomy_envelope_path=envelope_path,
        capability_registry_path=capability_path,
    )
    provider = ScriptedProvider(
        [
            ModelResponse(tool_calls=(ToolCall("read", "read_pipeline_search_context", {}),)),
            ModelResponse(
                tool_calls=(
                    ToolCall("baseline", "evaluate_pipeline_candidate", {"pipeline": bandpower}),
                )
            ),
            ModelResponse(
                tool_calls=(ToolCall("guided", "evaluate_pipeline_candidate", {"pipeline": csp}),)
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "lock",
                        "lock_pipeline",
                        {
                            "selected_experiment_id": selected["experiment_id"],
                            "evidence_experiment_ids": [
                                bandpower_result["experiment_id"],
                                csp_result["experiment_id"],
                            ],
                            "selection_rationale": ["Selected from two complete deterministic candidates."],
                            "rejected_alternatives": ["The other family had weaker search evidence."],
                            "uncertainty": ["Confirmation performance remains unknown."],
                            "stop_reason": "Minimum comparison completed and evidence is sufficient.",
                        },
                    ),
                )
            ),
            ModelResponse(content="Pipeline locked without human itemized approval."),
        ]
    )
    result = PipelineSearchAgent(
        runtime=AgentRuntime(provider=provider, tools=tools, run_id="pipeline-search-loop"),
        context=context,
    ).run()
    locked = result.latest_tool_result("lock_pipeline")
    assert result.status == "completed"
    assert locked["status"] == "locked_awaiting_confirmation"
    assert locked["selected_experiment_id"] == selected["experiment_id"]
    assert locked["budget_usage"]["research_cycles"] == 2
    assert locked["confirmation_accessed"] is False


def test_equivalent_candidate_and_early_lock_fail_closed(tmp_path: Path) -> None:
    subject_path, protocol_path, envelope_path = _contracts(tmp_path)
    tools, _ = create_pipeline_search_tools(
        executor=DeterministicPipelineExecutor(sessions=[_search_session()]),
        subject_profile_path=subject_path,
        frozen_protocol_path=protocol_path,
        autonomy_envelope_path=envelope_path,
        capability_registry_path=Path("configs/executable_pipeline_capabilities.v0.json"),
    )
    tools.execute("read_pipeline_search_context", {})
    first = tools.execute(
        "evaluate_pipeline_candidate",
        {"pipeline": _pipeline("bandpower_lda", "first-id")},
    )
    duplicate = _pipeline("bandpower_lda", "renamed-same-config")
    with pytest.raises(ToolExecutionError, match="Equivalent pipeline"):
        tools.execute("evaluate_pipeline_candidate", {"pipeline": duplicate})
    with pytest.raises(ToolExecutionError, match="At least 2 distinct candidates"):
        tools.execute(
            "lock_pipeline",
            {
                "selected_experiment_id": first["experiment_id"],
                "evidence_experiment_ids": [first["experiment_id"]],
                "selection_rationale": ["too early"],
                "rejected_alternatives": [],
                "uncertainty": [],
                "stop_reason": "too early",
            },
        )
