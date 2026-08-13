"""Autonomous budgeted pipeline search and evidence-bound lock agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci_autodiscovery.pipelines import (
    DeterministicPipelineExecutor,
    PipelineSpec,
    pipeline_configuration_hash,
)
from bci_autodiscovery.pipelines.executor import pipeline_spec_schema
from bci_autodiscovery.workflow.autonomy import (
    load_autonomy_envelope,
    load_json_object,
    sha256_path,
)

from .contracts import AgentRunResult
from .language import DEFAULT_LANGUAGE_INSTRUCTION
from .runtime import AgentRuntime
from .tools import ToolDefinition, ToolRegistry


PIPELINE_SEARCH_SYSTEM_PROMPT = f"""{DEFAULT_LANGUAGE_INSTRUCTION}
You are the autonomous individualized Pipeline Search Agent. First call
read_pipeline_search_context. Use the frozen protocol, SubjectProfile, executable capability
registry, and remaining budget. You cannot access frozen-confirmation data.

Propose one complete declarative pipeline at a time, call evaluate_pipeline_candidate, inspect
the deterministic cross-validation evidence, diagnose the result, then choose the next most
informative candidate. Do not enumerate the full Cartesian product. Vary a component only
when it tests a profile-linked hypothesis, a plausible alternative, or a necessary baseline.

When a frozen stopping condition is met or the budget should stop, call lock_pipeline with
the selected experiment, all evidence used, alternatives, uncertainty, and stop reason. The
selected candidate need not be the numerically highest score only when the evidence-backed
rationale justifies the tradeoff. Do not ask a human to select the pipeline."""


class PipelineSearchError(ValueError):
    pass


def _load_capabilities(path: Path) -> dict[str, Any]:
    value = load_json_object(path)
    if value.get("schema_version") != "1.0":
        raise PipelineSearchError("Unsupported executable capability schema_version")
    if value.get("status") != "executable_capabilities_not_search_activation":
        raise PipelineSearchError("Capability registry has an unsafe activation status")
    required_arrays = (
        "families",
        "bandpasses_hz",
        "csp_components",
        "lda_shrinkage",
        "cv_folds",
    )
    if any(not isinstance(value.get(field), list) or not value[field] for field in required_arrays):
        raise PipelineSearchError("Executable capability registry is incomplete")
    return value


def _validate_capability(spec: PipelineSpec, capabilities: dict[str, Any]) -> None:
    if spec.family not in capabilities["families"]:
        raise PipelineSearchError(f"Pipeline family is not executable: {spec.family}")
    bands = {tuple(float(item) for item in band) for band in capabilities["bandpasses_hz"]}
    if tuple(spec.bandpass_hz) not in bands:
        raise PipelineSearchError(f"Bandpass is outside executable capabilities: {spec.bandpass_hz}")
    if spec.family == "csp_lda" and spec.csp_components not in capabilities["csp_components"]:
        raise PipelineSearchError(
            f"CSP component count is outside executable capabilities: {spec.csp_components}"
        )
    if spec.lda_shrinkage not in {
        float(item) for item in capabilities["lda_shrinkage"]
    }:
        raise PipelineSearchError("LDA shrinkage is outside executable capabilities")
    if spec.cv_folds not in capabilities["cv_folds"]:
        raise PipelineSearchError("CV fold count is outside executable capabilities")


def create_pipeline_search_tools(
    *,
    executor: DeterministicPipelineExecutor,
    subject_profile_path: Path,
    frozen_protocol_path: Path,
    autonomy_envelope_path: Path,
    capability_registry_path: Path,
) -> tuple[ToolRegistry, dict[str, Any]]:
    subject_path = Path(subject_profile_path).expanduser().resolve()
    protocol_path = Path(frozen_protocol_path).expanduser().resolve()
    envelope_path = Path(autonomy_envelope_path).expanduser().resolve()
    capability_path = Path(capability_registry_path).expanduser().resolve()
    subject_profile = load_json_object(subject_path)
    protocol = load_json_object(protocol_path)
    capabilities = _load_capabilities(capability_path)
    if subject_profile.get("profile_complete") is not True:
        raise PipelineSearchError("Pipeline search requires a completed SubjectProfile")
    if subject_profile.get("subject_id") != executor.subject_id:
        raise PipelineSearchError("SubjectProfile and executor refer to different subjects")
    if protocol.get("status") != "frozen_autonomous":
        raise PipelineSearchError("Pipeline search requires a frozen autonomous protocol")
    dataset_id = str(protocol.get("dataset_id") or "")
    if not dataset_id:
        raise PipelineSearchError("Frozen protocol lacks dataset_id")
    envelope = load_autonomy_envelope(envelope_path, expected_dataset_id=dataset_id)
    search_sessions = tuple(
        str(item) for item in protocol["data_roles"]["pipeline_search_and_lock"]
    )
    executor_sessions = tuple(session.session_id for session in executor.sessions)
    if set(search_sessions) != set(executor_sessions):
        raise PipelineSearchError(
            "Executor sessions do not match the frozen pipeline_search_and_lock role"
        )

    protocol_budget = protocol.get("resource_budget") or {}
    max_executions = int(protocol_budget.get("max_candidate_executions", 0))
    max_cycles = int(protocol_budget.get("max_research_cycles", 0))
    if max_executions < 1 or max_cycles < 1:
        raise PipelineSearchError("Frozen protocol lacks a positive search budget")
    maximum = min(max_executions, max_cycles)
    minimum_before_lock = int(capabilities["minimum_distinct_candidates_before_lock"])
    primary_metric = str((protocol.get("evaluation") or {}).get("primary_metric") or "")
    if not primary_metric:
        raise PipelineSearchError("Frozen protocol lacks a primary evaluation metric")

    registry = ToolRegistry()
    context_read = False
    locked = False
    experiments: dict[str, dict[str, Any]] = {}
    configuration_hashes: set[str] = set()

    def require_context() -> None:
        if not context_read:
            raise PipelineSearchError("read_pipeline_search_context must be called first")
        if locked:
            raise PipelineSearchError("Pipeline has already been locked")

    def state() -> dict[str, Any]:
        summaries = [
            {
                "experiment_id": item["experiment_id"],
                "pipeline_id": item["pipeline"]["pipeline_id"],
                "family": item["pipeline"]["family"],
                "metrics": item["metrics"],
                "elapsed_seconds": item["elapsed_seconds"],
                "research_cycle_index": item["research_cycle_index"],
            }
            for item in experiments.values()
        ]
        return {
            "subject_id": executor.subject_id,
            "primary_metric": primary_metric,
            "executions_used": len(experiments),
            "executions_remaining": maximum - len(experiments),
            "maximum_research_cycles": maximum,
            "minimum_distinct_candidates_before_lock": minimum_before_lock,
            "experiments": summaries,
            "confirmation_data_accessed": False,
        }

    def read_context() -> dict[str, Any]:
        nonlocal context_read
        context_read = True
        return {
            "dataset_id": dataset_id,
            "subject_id": executor.subject_id,
            "subject_profile": subject_profile,
            "frozen_research_design": {
                "protocol_id": protocol["protocol_id"],
                "evaluation": protocol["evaluation"],
                "individual_oracle": protocol["individual_oracle"],
                "stopping_conditions": protocol["stopping_conditions"],
                "resource_budget": protocol["resource_budget"],
                "search_sessions": list(search_sessions),
            },
            "executable_capabilities": capabilities,
            "search_state": state(),
            "provenance": {
                "subject_profile": {"path": str(subject_path), "sha256": sha256_path(subject_path)},
                "frozen_protocol": {"path": str(protocol_path), "sha256": sha256_path(protocol_path)},
                "autonomy_envelope": {"path": str(envelope_path), "sha256": sha256_path(envelope_path)},
                "capability_registry": {"path": str(capability_path), "sha256": sha256_path(capability_path)},
            },
        }

    def evaluate_candidate(pipeline: dict[str, Any]) -> dict[str, Any]:
        require_context()
        if len(experiments) >= maximum:
            raise PipelineSearchError("Autonomous search budget is exhausted")
        spec = PipelineSpec.from_dict(pipeline)
        _validate_capability(spec, capabilities)
        configuration_hash = pipeline_configuration_hash(spec)
        if configuration_hash in configuration_hashes:
            raise PipelineSearchError("Equivalent pipeline configuration was already evaluated")
        result = executor.evaluate(spec)
        if primary_metric not in result["metrics"]:
            raise PipelineSearchError(
                f"Executor result does not contain primary metric {primary_metric!r}"
            )
        value = float(result["metrics"][primary_metric])
        if not 0 <= value <= 1:
            raise PipelineSearchError("Primary metric is outside [0, 1]")
        result["research_cycle_index"] = len(experiments) + 1
        result["configuration_sha256"] = configuration_hash
        experiments[result["experiment_id"]] = result
        configuration_hashes.add(configuration_hash)
        result["budget_after_execution"] = state()
        return result

    def inspect_state() -> dict[str, Any]:
        require_context()
        return state()

    def lock_pipeline(
        selected_experiment_id: str,
        evidence_experiment_ids: list[str],
        selection_rationale: list[str],
        rejected_alternatives: list[str],
        uncertainty: list[str],
        stop_reason: str,
    ) -> dict[str, Any]:
        nonlocal locked
        require_context()
        if len(configuration_hashes) < minimum_before_lock:
            raise PipelineSearchError(
                f"At least {minimum_before_lock} distinct candidates are required before lock"
            )
        if selected_experiment_id not in experiments:
            raise PipelineSearchError("Selected experiment is not part of this search run")
        cited = set(evidence_experiment_ids)
        if selected_experiment_id not in cited or not cited.issubset(experiments):
            raise PipelineSearchError("Lock evidence must cite available experiments including selection")
        if not selection_rationale or not stop_reason.strip():
            raise PipelineSearchError("Pipeline lock requires rationale and stop reason")
        selected = experiments[selected_experiment_id]
        observed_best = max(
            float(item["metrics"][primary_metric]) for item in experiments.values()
        )
        selected_score = float(selected["metrics"][primary_metric])
        tolerance = float(capabilities["observed_best_selection_tolerance"])
        locked = True
        return {
            "schema_version": "1.0",
            "lock_id": f"lock-{selected['pipeline_sha256'][:16]}",
            "status": "locked_awaiting_confirmation",
            "dataset_id": dataset_id,
            "subject_id": executor.subject_id,
            "protocol_id": protocol["protocol_id"],
            "selected_experiment_id": selected_experiment_id,
            "selected_pipeline": selected["pipeline"],
            "pipeline_sha256": selected["pipeline_sha256"],
            "primary_metric": primary_metric,
            "selected_search_score": selected_score,
            "observed_best_search_score": observed_best,
            "gap_to_observed_best": observed_best - selected_score,
            "within_capability_selection_tolerance": (
                observed_best - selected_score <= tolerance
            ),
            "selection_rationale": selection_rationale,
            "rejected_alternatives": rejected_alternatives,
            "uncertainty": uncertainty,
            "stop_reason": stop_reason,
            "evidence_experiment_ids": evidence_experiment_ids,
            "search_trace": list(experiments.values()),
            "budget_usage": {
                "research_cycles": len(experiments),
                "candidate_executions": len(experiments),
                "authorized_maximum": maximum,
            },
            "source_contracts": {
                "subject_profile": {"path": str(subject_path), "sha256": sha256_path(subject_path)},
                "frozen_protocol": {"path": str(protocol_path), "sha256": sha256_path(protocol_path)},
                "autonomy_envelope": {"path": str(envelope_path), "sha256": sha256_path(envelope_path), "envelope_id": envelope["envelope_id"]},
                "capability_registry": {"path": str(capability_path), "sha256": sha256_path(capability_path)},
            },
            "confirmation_accessed": False,
            "search_reopen_allowed_after_confirmation": False,
        }

    registry.register(
        ToolDefinition(
            name="read_pipeline_search_context",
            description="Read subject evidence, frozen design, executable capabilities, and budget.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            approval="never",
            decision_kind="read_only_pipeline_search_contract",
            tags=("read-only", "search", "no-confirmation"),
        ),
        read_context,
    )
    registry.register(
        ToolDefinition(
            name="evaluate_pipeline_candidate",
            description="Execute one distinct complete pipeline on authorized search data.",
            input_schema={
                "type": "object",
                "properties": {"pipeline": pipeline_spec_schema()},
                "required": ["pipeline"],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="autonomous_budgeted_experiment",
            tags=("numeric", "budgeted", "search-data-only"),
        ),
        evaluate_candidate,
    )
    registry.register(
        ToolDefinition(
            name="inspect_pipeline_search_state",
            description="Inspect completed experiments and remaining budget without new computation.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            approval="never",
            decision_kind="read_only_search_state",
            tags=("read-only", "budget"),
        ),
        inspect_state,
    )
    registry.register(
        ToolDefinition(
            name="lock_pipeline",
            description="Lock one evidence-backed pipeline; no human itemized approval is used.",
            input_schema={
                "type": "object",
                "properties": {
                    "selected_experiment_id": {"type": "string"},
                    "evidence_experiment_ids": {"type": "array", "items": {"type": "string"}},
                    "selection_rationale": {"type": "array", "items": {"type": "string"}},
                    "rejected_alternatives": {"type": "array", "items": {"type": "string"}},
                    "uncertainty": {"type": "array", "items": {"type": "string"}},
                    "stop_reason": {"type": "string"},
                },
                "required": [
                    "selected_experiment_id",
                    "evidence_experiment_ids",
                    "selection_rationale",
                    "rejected_alternatives",
                    "uncertainty",
                    "stop_reason",
                ],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="autonomous_pipeline_lock",
            tags=("local-write", "pipeline-lock", "confirmation-prerequisite"),
        ),
        lock_pipeline,
    )
    return registry, {
        "dataset_id": dataset_id,
        "subject_id": executor.subject_id,
        "task": "budgeted_individualized_pipeline_search_and_lock",
        "maximum_research_cycles": maximum,
        "primary_metric": primary_metric,
    }


@dataclass
class PipelineSearchAgent:
    runtime: AgentRuntime
    context: dict[str, Any]

    def run(self) -> AgentRunResult:
        return self.runtime.run(
            system_prompt=PIPELINE_SEARCH_SYSTEM_PROMPT,
            user_prompt=json.dumps(self.context, ensure_ascii=False, sort_keys=True),
        )
