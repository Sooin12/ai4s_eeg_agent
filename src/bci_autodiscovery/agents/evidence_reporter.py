"""Evidence-bound internal report Agent after one-shot confirmation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci_autodiscovery.agents.pipeline_lock_critic import (
    PipelineLockCriticError,
    validate_pipeline_lock,
    validate_pipeline_lock_critique,
)
from bci_autodiscovery.evaluation import evaluate_frozen_decision
from bci_autodiscovery.workflow.autonomy import (
    load_autonomy_envelope,
    load_json_object,
    sha256_path,
)

from .contracts import AgentRunResult
from .language import DEFAULT_LANGUAGE_INSTRUCTION
from .runtime import AgentRuntime
from .tools import ToolDefinition, ToolRegistry


EVIDENCE_REPORTER_SYSTEM_PROMPT = f"""{DEFAULT_LANGUAGE_INSTRUCTION}
You are the autonomous Evidence Reporter. First call read_evidence_report_context. The
deterministic frozen-decision result is authoritative: you may explain it but cannot change
its outcome. Build an internal evidence report whose claims cite exact artifact fields.
Every evidence_refs item MUST use the literal syntax artifact_name#field_path, for example
deterministic_decision#outcome, confirmation_result#confirmation_score, or
pipeline_lock#selected_pipeline. Dot notation, JSONPath, and bare artifact names are invalid.

Report the selected and rejected alternatives, confirmation change, negative results,
uncertainty, research cycles used, and reproducibility bindings. Never turn an engineering
smoke result into a scientific claim, never hide an inconclusive/refuse result, and never
claim external publication authority. Call record_evidence_report once."""


class EvidenceReportError(ValueError):
    pass


_EVIDENCE_PREFIXES = {
    "subject_profile",
    "pipeline_lock",
    "pipeline_lock_critique",
    "confirmation_result",
    "frozen_protocol",
    "autonomy_envelope",
    "deterministic_decision",
}


def evidence_report_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "enum": ["1.0"]},
            "report_id": {"type": "string"},
            "status": {"type": "string", "enum": ["draft_for_scientific_critic"]},
            "dataset_id": {"type": "string"},
            "subject_id": {"type": "string"},
            "protocol_id": {"type": "string"},
            "lock_id": {"type": "string"},
            "conclusion": {"type": "string", "enum": ["success", "inconclusive", "refuse"]},
            "headline": {"type": "string"},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "statement": {"type": "string"},
                        "status": {"type": "string", "enum": ["supported", "uncertain", "refuted"]},
                        "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["claim_id", "statement", "status", "evidence_refs"],
                    "additionalProperties": False,
                },
            },
            "negative_results": {"type": "array", "items": {"type": "string"}},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
            "outcome_summary": {
                "type": "object",
                "properties": {
                    "primary_metric": {"type": "string"},
                    "search_score": {"type": "number"},
                    "confirmation_score": {"type": "number"},
                    "confirmation_minus_search": {"type": "number"},
                },
                "required": [
                    "primary_metric",
                    "search_score",
                    "confirmation_score",
                    "confirmation_minus_search",
                ],
                "additionalProperties": False,
            },
            "research_cycle_efficiency": {
                "type": "object",
                "properties": {
                    "research_cycles": {"type": "integer"},
                    "candidate_executions": {"type": "integer"},
                    "authorized_maximum": {"type": "integer"},
                    "unused_authorized_cycles": {"type": "integer"},
                    "interpretation": {"type": "string"},
                },
                "required": [
                    "research_cycles",
                    "candidate_executions",
                    "authorized_maximum",
                    "unused_authorized_cycles",
                    "interpretation",
                ],
                "additionalProperties": False,
            },
            "reproducibility_notes": {"type": "array", "items": {"type": "string"}},
            "scope": {"type": "string", "enum": ["internal_evidence_report"]},
            "external_claim_authorized": {"type": "boolean", "enum": [False]},
        },
        "required": [
            "schema_version",
            "report_id",
            "status",
            "dataset_id",
            "subject_id",
            "protocol_id",
            "lock_id",
            "conclusion",
            "headline",
            "claims",
            "negative_results",
            "uncertainties",
            "outcome_summary",
            "research_cycle_efficiency",
            "reproducibility_notes",
            "scope",
            "external_claim_authorized",
        ],
        "additionalProperties": False,
    }


def validate_evidence_report(
    report: dict[str, Any],
    *,
    protocol: dict[str, Any],
    pipeline_lock: dict[str, Any],
    confirmation_result: dict[str, Any],
    deterministic_decision: dict[str, Any],
) -> None:
    if report.get("schema_version") != "1.0":
        raise EvidenceReportError("Unsupported evidence report schema_version")
    if report.get("status") != "draft_for_scientific_critic":
        raise EvidenceReportError("Evidence report must await scientific critic review")
    expected = {
        "dataset_id": protocol.get("dataset_id"),
        "subject_id": pipeline_lock.get("subject_id"),
        "protocol_id": protocol.get("protocol_id"),
        "lock_id": pipeline_lock.get("lock_id"),
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise EvidenceReportError(f"Evidence report has mismatched {field}")
    if report.get("conclusion") != deterministic_decision.get("outcome"):
        raise EvidenceReportError("Reporter cannot override the frozen deterministic outcome")
    if report.get("scope") != "internal_evidence_report" or report.get(
        "external_claim_authorized"
    ) is not False:
        raise EvidenceReportError("Evidence report exceeds the authorized conclusion scope")
    for field in ("headline", "report_id"):
        if not isinstance(report.get(field), str) or not report[field].strip():
            raise EvidenceReportError(f"Evidence report has empty {field}")
    claims = report.get("claims")
    if not isinstance(claims, list) or not claims:
        raise EvidenceReportError("Evidence report requires at least one evidence-bound claim")
    claim_ids: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            raise EvidenceReportError("Evidence report claim must be an object")
        claim_id = str(claim.get("claim_id") or "").strip()
        if not claim_id or claim_id in claim_ids:
            raise EvidenceReportError("Evidence report claim IDs must be non-empty and unique")
        claim_ids.add(claim_id)
        if not isinstance(claim.get("statement"), str) or not claim["statement"].strip():
            raise EvidenceReportError(f"Claim {claim_id} has no statement")
        refs = claim.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            raise EvidenceReportError(f"Claim {claim_id} lacks evidence references")
        for ref in refs:
            prefix, separator, field = str(ref).partition("#")
            if not separator or not field or prefix not in _EVIDENCE_PREFIXES:
                raise EvidenceReportError(f"Claim {claim_id} has invalid evidence ref {ref!r}")
    if not isinstance(report.get("negative_results"), list):
        raise EvidenceReportError("negative_results must be an array")
    if not isinstance(report.get("uncertainties"), list) or not report["uncertainties"]:
        raise EvidenceReportError("Evidence report must state at least one uncertainty")
    if not isinstance(report.get("reproducibility_notes"), list) or not report[
        "reproducibility_notes"
    ]:
        raise EvidenceReportError("Evidence report requires reproducibility notes")

    outcome = report.get("outcome_summary") or {}
    exact_outcome = {
        "primary_metric": confirmation_result.get("primary_metric"),
        "search_score": confirmation_result.get("search_score"),
        "confirmation_score": confirmation_result.get("confirmation_score"),
        "confirmation_minus_search": confirmation_result.get("confirmation_minus_search"),
    }
    if outcome != exact_outcome:
        raise EvidenceReportError("Outcome summary differs from confirmation evidence")
    usage = pipeline_lock.get("budget_usage") or {}
    efficiency = report.get("research_cycle_efficiency") or {}
    expected_efficiency = {
        "research_cycles": int(usage["research_cycles"]),
        "candidate_executions": int(usage["candidate_executions"]),
        "authorized_maximum": int(usage["authorized_maximum"]),
    }
    for field, value in expected_efficiency.items():
        if efficiency.get(field) != value:
            raise EvidenceReportError(f"Research-cycle field {field} differs from audit evidence")
    if efficiency.get("unused_authorized_cycles") != (
        expected_efficiency["authorized_maximum"] - expected_efficiency["research_cycles"]
    ):
        raise EvidenceReportError("unused_authorized_cycles is not auditable")
    if not isinstance(efficiency.get("interpretation"), str) or not efficiency[
        "interpretation"
    ].strip():
        raise EvidenceReportError("Research-cycle efficiency lacks interpretation")


def create_evidence_reporter_tools(
    *,
    subject_profile_path: Path,
    pipeline_lock_path: Path,
    lock_critique_path: Path,
    confirmation_result_path: Path,
    frozen_protocol_path: Path,
    autonomy_envelope_path: Path,
) -> tuple[ToolRegistry, dict[str, Any]]:
    paths = {
        "subject_profile": Path(subject_profile_path).expanduser().resolve(),
        "pipeline_lock": Path(pipeline_lock_path).expanduser().resolve(),
        "pipeline_lock_critique": Path(lock_critique_path).expanduser().resolve(),
        "confirmation_result": Path(confirmation_result_path).expanduser().resolve(),
        "frozen_protocol": Path(frozen_protocol_path).expanduser().resolve(),
        "autonomy_envelope": Path(autonomy_envelope_path).expanduser().resolve(),
    }
    subject = load_json_object(paths["subject_profile"])
    lock = load_json_object(paths["pipeline_lock"])
    critique = load_json_object(paths["pipeline_lock_critique"])
    confirmation = load_json_object(paths["confirmation_result"])
    protocol = load_json_object(paths["frozen_protocol"])
    envelope = load_autonomy_envelope(
        paths["autonomy_envelope"], expected_dataset_id=str(protocol["dataset_id"])
    )
    lock_hash = sha256_path(paths["pipeline_lock"])
    try:
        validate_pipeline_lock(lock, protocol=protocol, envelope=envelope)
        validate_pipeline_lock_critique(
            critique,
            lock=lock,
            lock_sha256=lock_hash,
            deterministic_validation_passed=True,
        )
    except (PipelineLockCriticError, KeyError, TypeError, ValueError) as exc:
        raise EvidenceReportError(f"Invalid report source lock: {exc}") from exc
    if critique.get("verdict") != "pass":
        raise EvidenceReportError("Evidence report requires a passed lock critique")
    access_ref = confirmation.get("access_record") or {}
    access_path = Path(str(access_ref.get("path"))).expanduser().resolve()
    if not access_path.is_file() or sha256_path(access_path) != access_ref.get("sha256"):
        raise EvidenceReportError("Confirmation access record failed integrity validation")
    access_record = load_json_object(access_path)
    if access_record.get("status") != "consumed" or access_record.get("access_count") != 1:
        raise EvidenceReportError("Confirmation was not produced under one-shot access")
    decision = evaluate_frozen_decision(
        protocol=protocol,
        pipeline_lock=lock,
        confirmation_result=confirmation,
    )
    sources = {
        name: {"path": str(path), "sha256": sha256_path(path)}
        for name, path in paths.items()
    }
    sources["confirmation_access_record"] = {
        "path": str(access_path),
        "sha256": sha256_path(access_path),
    }
    registry = ToolRegistry()
    context_read = False
    recorded = False

    def read_context() -> dict[str, Any]:
        nonlocal context_read
        context_read = True
        confirmation_summary = {
            key: value
            for key, value in confirmation.items()
            if key not in {"fitted_pipeline"}
        }
        return {
            "subject_profile": subject,
            "pipeline_lock": lock,
            "pipeline_lock_critique": critique,
            "confirmation_result": confirmation_summary,
            "frozen_protocol": protocol,
            "deterministic_decision": decision,
            "source_contracts": sources,
            "authorized_scope": "internal_evidence_report",
        }

    def record(report: dict[str, Any]) -> dict[str, Any]:
        nonlocal recorded
        if not context_read:
            raise EvidenceReportError("read_evidence_report_context must be called first")
        if recorded:
            raise EvidenceReportError("Only one evidence report may be recorded")
        validate_evidence_report(
            report,
            protocol=protocol,
            pipeline_lock=lock,
            confirmation_result=confirmation,
            deterministic_decision=decision,
        )
        recorded = True
        result = json.loads(json.dumps(report))
        result["deterministic_decision"] = decision
        result["source_contracts"] = sources
        return result

    registry.register(
        ToolDefinition(
            name="read_evidence_report_context",
            description="Read bounded evidence and the immutable deterministic conclusion.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            approval="never",
            decision_kind="read_only_confirmed_evidence",
            tags=("read-only", "post-confirmation", "internal-only"),
        ),
        read_context,
    )
    registry.register(
        ToolDefinition(
            name="record_evidence_report",
            description=(
                "Record one evidence-bound internal report for scientific review. Claim refs "
                "must use artifact_name#field_path, e.g. deterministic_decision#outcome."
            ),
            input_schema={
                "type": "object",
                "properties": {"report": evidence_report_schema()},
                "required": ["report"],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="autonomous_internal_evidence_reporting",
            tags=("local-write", "evidence-report", "critic-required"),
        ),
        record,
    )
    return registry, {
        "dataset_id": protocol["dataset_id"],
        "subject_id": lock["subject_id"],
        "task": "evidence_bound_internal_report",
        "deterministic_outcome": decision["outcome"],
    }


@dataclass
class EvidenceReporterAgent:
    runtime: AgentRuntime
    context: dict[str, Any]

    def run(self) -> AgentRunResult:
        return self.runtime.run(
            system_prompt=EVIDENCE_REPORTER_SYSTEM_PROMPT,
            user_prompt=json.dumps(self.context, ensure_ascii=False, sort_keys=True),
        )
