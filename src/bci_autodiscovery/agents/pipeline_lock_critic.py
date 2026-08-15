"""Outcome-blind independent review of an autonomous pipeline lock."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci_autodiscovery.pipelines import PipelineSpec
from bci_autodiscovery.search.dataset_incumbent import (
    DatasetIncumbentError,
    expected_selective_route,
    validate_dataset_incumbent,
)
from bci_autodiscovery.workflow.autonomy import (
    load_autonomy_envelope,
    load_json_object,
    sha256_path,
)

from .contracts import AgentRunResult
from .language import DEFAULT_LANGUAGE_INSTRUCTION
from .runtime import AgentRuntime
from .tools import ToolDefinition, ToolRegistry


PIPELINE_LOCK_CRITIC_SYSTEM_PROMPT = f"""{DEFAULT_LANGUAGE_INSTRUCTION}
You are the independent Pipeline Lock Critic. First call read_pipeline_lock_critic_context.
You cannot access frozen-confirmation data. Audit whether the selected experiment exists in
the search trace, every cited experiment was actually executed inside budget, the selected
pipeline and hashes match, the rationale is supported by the SubjectProfile and search
evidence, and any departure from the observed best score has a defensible pre-confirmation
tradeoff.

Call record_pipeline_lock_critique once with pass, revise, or reject. Pass only when the
deterministic validation succeeds and no critical or major finding remains. Do not ask a
human to select the pipeline and do not propose using confirmation data to resolve doubt."""


class PipelineLockCriticError(ValueError):
    pass


def validate_pipeline_lock(
    lock: dict[str, Any],
    *,
    protocol: dict[str, Any],
    envelope: dict[str, Any],
) -> None:
    if lock.get("schema_version") != "1.0":
        raise PipelineLockCriticError("Unsupported pipeline lock schema_version")
    if lock.get("status") != "locked_awaiting_confirmation":
        raise PipelineLockCriticError("Pipeline lock has an invalid status")
    if lock.get("dataset_id") != protocol.get("dataset_id"):
        raise PipelineLockCriticError("Pipeline lock belongs to another dataset")
    if lock.get("protocol_id") != protocol.get("protocol_id"):
        raise PipelineLockCriticError("Pipeline lock belongs to another protocol")
    if lock.get("confirmation_accessed") is not False:
        raise PipelineLockCriticError("Pipeline lock was created after confirmation access")
    if lock.get("search_reopen_allowed_after_confirmation") is not False:
        raise PipelineLockCriticError("Pipeline lock permits search reopening after confirmation")
    if not isinstance(lock.get("selection_rationale"), list) or not lock[
        "selection_rationale"
    ]:
        raise PipelineLockCriticError("Pipeline lock lacks selection rationale")
    if not isinstance(lock.get("search_trace"), list) or not lock["search_trace"]:
        raise PipelineLockCriticError("Pipeline lock lacks a search trace")
    experiments = {
        item.get("experiment_id"): item
        for item in lock["search_trace"]
        if isinstance(item, dict) and item.get("experiment_id")
    }
    selected_id = lock.get("selected_experiment_id")
    if selected_id not in experiments:
        raise PipelineLockCriticError("Selected experiment is absent from search trace")
    selected = experiments[selected_id]
    PipelineSpec.from_dict(lock.get("selected_pipeline") or {})
    if lock.get("selected_pipeline") != selected.get("pipeline"):
        raise PipelineLockCriticError("Selected pipeline differs from selected experiment")
    if lock.get("pipeline_sha256") != selected.get("pipeline_sha256"):
        raise PipelineLockCriticError("Selected pipeline hash differs from search evidence")
    evidence_ids = set(lock.get("evidence_experiment_ids") or [])
    if selected_id not in evidence_ids or not evidence_ids.issubset(experiments):
        raise PipelineLockCriticError("Pipeline lock cites unavailable experiment evidence")
    if any(
        item.get("data_role") != "pipeline_search_and_lock"
        or item.get("confirmation_data_accessed") is not False
        for item in experiments.values()
    ):
        raise PipelineLockCriticError("Search trace contains non-search or confirmation evidence")
    literature_ids = set(lock.get("evidence_literature_paper_ids") or [])
    literature_search = lock.get("literature_search") or {}
    available_literature_ids = {
        str(paper_id)
        for query in literature_search.get("queries") or []
        for paper_id in (query or {}).get("paper_ids") or []
    }
    if not literature_ids.issubset(available_literature_ids):
        raise PipelineLockCriticError("Pipeline lock cites unavailable literature evidence")
    if literature_ids and literature_search.get("evidence_scope") != (
        "scholarly_metadata_or_abstract_discovery_only"
    ):
        raise PipelineLockCriticError("Pipeline lock overstates its literature evidence scope")
    usage = lock.get("budget_usage") or {}
    executions = int(usage.get("candidate_executions", -1))
    cycles = int(usage.get("research_cycles", -1))
    if executions != len(experiments) or cycles != len(experiments):
        raise PipelineLockCriticError("Pipeline lock budget usage differs from search trace")
    authorized = min(
        int(protocol["resource_budget"]["max_candidate_executions"]),
        int(protocol["resource_budget"]["max_research_cycles"]),
        int(envelope["resource_budget"]["max_candidate_executions"]),
        int(envelope["resource_budget"]["max_research_cycles"]),
    )
    if executions < 1 or executions > authorized:
        raise PipelineLockCriticError("Pipeline search exceeded its authorized budget")
    metric = str(protocol["evaluation"]["primary_metric"])
    if lock.get("primary_metric") != metric or metric not in selected.get("metrics", {}):
        raise PipelineLockCriticError("Pipeline lock primary metric is inconsistent")
    selected_score = float(selected["metrics"][metric])
    if abs(float(lock.get("selected_search_score")) - selected_score) > 1e-12:
        raise PipelineLockCriticError("Selected search score differs from experiment evidence")
    incumbent_binding = (lock.get("source_contracts") or {}).get(
        "dataset_pipeline_incumbent"
    )
    if incumbent_binding is None:
        if lock.get("route_decision") is not None:
            raise PipelineLockCriticError(
                "Pipeline lock has an unbound selective route decision"
            )
        return
    incumbent_path = Path(str(incumbent_binding.get("path") or "")).expanduser().resolve()
    if (
        not incumbent_path.is_file()
        or sha256_path(incumbent_path) != incumbent_binding.get("sha256")
    ):
        raise PipelineLockCriticError("Dataset incumbent binding failed integrity check")
    incumbent = load_json_object(incumbent_path)
    try:
        validate_dataset_incumbent(incumbent)
        expected = expected_selective_route(
            incumbent_configuration_sha256=str(
                incumbent["pipeline_configuration_sha256"]
            ),
            minimum_gain=float(
                incumbent["personalization_policy"][
                    "minimum_search_gain_over_incumbent"
                ]
            ),
            experiments=list(experiments.values()),
            primary_metric=metric,
        )
    except DatasetIncumbentError as exc:
        raise PipelineLockCriticError(
            f"Selective personalization evidence is invalid: {exc}"
        ) from exc
    if lock.get("route_decision") != expected:
        raise PipelineLockCriticError("Selective route decision differs from frozen policy")
    if selected_id != expected["required_selected_experiment_id"]:
        raise PipelineLockCriticError("Selected pipeline violates the fallback gate")


def pipeline_lock_critique_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "enum": ["1.0"]},
            "review_id": {"type": "string"},
            "dataset_id": {"type": "string"},
            "subject_id": {"type": "string"},
            "lock_id": {"type": "string"},
            "reviewed_lock_sha256": {"type": "string"},
            "verdict": {"type": "string", "enum": ["pass", "revise", "reject"]},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "severity": {"type": "string", "enum": ["critical", "major", "minor", "note"]},
                        "message": {"type": "string"},
                        "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["code", "severity", "message", "evidence_refs"],
                    "additionalProperties": False,
                },
            },
            "required_revisions": {"type": "array", "items": {"type": "string"}},
            "rationale": {"type": "string"},
        },
        "required": [
            "schema_version",
            "review_id",
            "dataset_id",
            "subject_id",
            "lock_id",
            "reviewed_lock_sha256",
            "verdict",
            "findings",
            "required_revisions",
            "rationale",
        ],
        "additionalProperties": False,
    }


def validate_pipeline_lock_critique(
    critique: dict[str, Any],
    *,
    lock: dict[str, Any],
    lock_sha256: str,
    deterministic_validation_passed: bool,
) -> None:
    if critique.get("schema_version") != "1.0":
        raise PipelineLockCriticError("Unsupported pipeline lock critique schema_version")
    for field in ("dataset_id", "subject_id", "lock_id"):
        if critique.get(field) != lock.get(field):
            raise PipelineLockCriticError(f"Pipeline lock critique has mismatched {field}")
    if critique.get("reviewed_lock_sha256") != lock_sha256:
        raise PipelineLockCriticError("Pipeline lock critique is not bound to exact lock SHA")
    if not isinstance(critique.get("rationale"), str) or not critique["rationale"].strip():
        raise PipelineLockCriticError("Pipeline lock critique lacks rationale")
    findings = critique.get("findings")
    revisions = critique.get("required_revisions")
    if not isinstance(findings, list) or not isinstance(revisions, list):
        raise PipelineLockCriticError("Critique findings and revisions must be arrays")
    blocking = []
    for finding in findings:
        if not isinstance(finding, dict) or finding.get("severity") not in {
            "critical",
            "major",
            "minor",
            "note",
        }:
            raise PipelineLockCriticError("Malformed pipeline lock finding")
        if finding.get("severity") in {"critical", "major"}:
            blocking.append(finding.get("code"))
    verdict = critique.get("verdict")
    if verdict == "pass":
        if not deterministic_validation_passed:
            raise PipelineLockCriticError("Critic cannot pass a deterministically invalid lock")
        if blocking or revisions:
            raise PipelineLockCriticError("Pass verdict retains blocking findings or revisions")
    elif verdict == "revise":
        if not revisions:
            raise PipelineLockCriticError("Revise verdict requires concrete revisions")
    elif verdict == "reject":
        if not findings:
            raise PipelineLockCriticError("Reject verdict requires findings")
    else:
        raise PipelineLockCriticError("Unknown pipeline lock critique verdict")


def create_pipeline_lock_critic_tools(
    *,
    pipeline_lock_path: Path,
    subject_profile_path: Path,
    frozen_protocol_path: Path,
    autonomy_envelope_path: Path,
) -> tuple[ToolRegistry, dict[str, Any]]:
    lock_path = Path(pipeline_lock_path).expanduser().resolve()
    subject_path = Path(subject_profile_path).expanduser().resolve()
    protocol_path = Path(frozen_protocol_path).expanduser().resolve()
    envelope_path = Path(autonomy_envelope_path).expanduser().resolve()
    lock = load_json_object(lock_path)
    subject_profile = load_json_object(subject_path)
    protocol = load_json_object(protocol_path)
    envelope = load_autonomy_envelope(
        envelope_path,
        expected_dataset_id=str(protocol["dataset_id"]),
    )
    deterministic_errors: list[str] = []
    try:
        validate_pipeline_lock(lock, protocol=protocol, envelope=envelope)
    except (PipelineLockCriticError, KeyError, TypeError, ValueError) as exc:
        deterministic_errors.append(str(exc))
    passed = not deterministic_errors
    lock_hash = sha256_path(lock_path)
    registry = ToolRegistry()
    context_read = False
    recorded = False

    def read_context() -> dict[str, Any]:
        nonlocal context_read
        context_read = True
        return {
            "pipeline_lock": lock,
            "subject_profile": subject_profile,
            "frozen_protocol": protocol,
            "deterministic_validation": {"passed": passed, "errors": deterministic_errors},
            "confirmation_results_available": False,
            "provenance": {
                "pipeline_lock": {"path": str(lock_path), "sha256": lock_hash},
                "subject_profile": {"path": str(subject_path), "sha256": sha256_path(subject_path)},
                "frozen_protocol": {"path": str(protocol_path), "sha256": sha256_path(protocol_path)},
                "autonomy_envelope": {"path": str(envelope_path), "sha256": sha256_path(envelope_path)},
            },
        }

    def record(critique: dict[str, Any]) -> dict[str, Any]:
        nonlocal recorded
        if not context_read:
            raise PipelineLockCriticError("read_pipeline_lock_critic_context must be called first")
        if recorded:
            raise PipelineLockCriticError("Only one pipeline lock critique may be recorded")
        validate_pipeline_lock_critique(
            critique,
            lock=lock,
            lock_sha256=lock_hash,
            deterministic_validation_passed=passed,
        )
        recorded = True
        result = json.loads(json.dumps(critique))
        result["source_lock"] = {"path": str(lock_path), "sha256": lock_hash}
        result["confirmation_results_available"] = False
        return result

    registry.register(
        ToolDefinition(
            name="read_pipeline_lock_critic_context",
            description="Read the pipeline lock and outcome-blind evidence for independent review.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            approval="never",
            decision_kind="independent_pipeline_lock_review",
            tags=("read-only", "critic", "no-confirmation"),
        ),
        read_context,
    )
    registry.register(
        ToolDefinition(
            name="record_pipeline_lock_critique",
            description="Record one outcome-blind pass, revise, or reject verdict for the lock.",
            input_schema={
                "type": "object",
                "properties": {"critique": pipeline_lock_critique_schema()},
                "required": ["critique"],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="autonomous_pipeline_lock_critique",
            tags=("local-write", "critic", "confirmation-gate"),
        ),
        record,
    )
    return registry, {
        "dataset_id": lock.get("dataset_id"),
        "subject_id": lock.get("subject_id"),
        "lock_id": lock.get("lock_id"),
        "task": "outcome_blind_pipeline_lock_review",
        "deterministic_validation_passed": passed,
    }


@dataclass
class PipelineLockCriticAgent:
    runtime: AgentRuntime
    context: dict[str, Any]

    def run(self) -> AgentRunResult:
        return self.runtime.run(
            system_prompt=PIPELINE_LOCK_CRITIC_SYSTEM_PROMPT,
            user_prompt=json.dumps(self.context, ensure_ascii=False, sort_keys=True),
        )
