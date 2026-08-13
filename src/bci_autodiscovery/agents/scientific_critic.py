"""Independent critic for confirmed internal evidence reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci_autodiscovery.evaluation import evaluate_frozen_decision
from bci_autodiscovery.workflow.autonomy import load_json_object, sha256_path

from .contracts import AgentRunResult
from .evidence_reporter import EvidenceReportError, validate_evidence_report
from .language import DEFAULT_LANGUAGE_INSTRUCTION
from .runtime import AgentRuntime
from .tools import ToolDefinition, ToolRegistry


SCIENTIFIC_CRITIC_SYSTEM_PROMPT = f"""{DEFAULT_LANGUAGE_INSTRUCTION}
You are the independent Scientific Critic for an internal individualized-pipeline report.
First call read_scientific_critic_context. Audit exact evidence binding, deterministic
conclusion consistency, negative-result disclosure, uncertainty, research-cycle accounting,
reproducibility, and scope discipline. You cannot alter the frozen decision thresholds or
reinterpret an inconclusive/refuse result as success.

Call record_scientific_critique once with pass, revise, or reject. Pass only if deterministic
validation succeeds and no critical or major finding remains. This review authorizes only an
internal finalized report, never an external scientific claim."""


class ScientificCriticError(ValueError):
    pass


def scientific_critique_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "enum": ["1.0"]},
            "review_id": {"type": "string"},
            "dataset_id": {"type": "string"},
            "subject_id": {"type": "string"},
            "report_id": {"type": "string"},
            "reviewed_report_sha256": {"type": "string"},
            "verdict": {"type": "string", "enum": ["pass", "revise", "reject"]},
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "dimension": {
                            "type": "string",
                            "enum": [
                                "evidence_binding",
                                "conclusion_consistency",
                                "negative_results",
                                "uncertainty",
                                "cycle_accounting",
                                "reproducibility",
                                "scope",
                            ],
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["critical", "major", "minor", "note"],
                        },
                        "message": {"type": "string"},
                        "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "code",
                        "dimension",
                        "severity",
                        "message",
                        "evidence_refs",
                    ],
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
            "report_id",
            "reviewed_report_sha256",
            "verdict",
            "findings",
            "required_revisions",
            "rationale",
        ],
        "additionalProperties": False,
    }


def validate_scientific_critique(
    critique: dict[str, Any],
    *,
    report: dict[str, Any],
    report_sha256: str,
    deterministic_validation_passed: bool,
) -> None:
    if critique.get("schema_version") != "1.0":
        raise ScientificCriticError("Unsupported scientific critique schema_version")
    for field in ("dataset_id", "subject_id", "report_id"):
        if critique.get(field) != report.get(field):
            raise ScientificCriticError(f"Scientific critique has mismatched {field}")
    if critique.get("reviewed_report_sha256") != report_sha256:
        raise ScientificCriticError("Scientific critique is not bound to exact report SHA")
    if not isinstance(critique.get("rationale"), str) or not critique["rationale"].strip():
        raise ScientificCriticError("Scientific critique lacks rationale")
    findings = critique.get("findings")
    revisions = critique.get("required_revisions")
    if not isinstance(findings, list) or not isinstance(revisions, list):
        raise ScientificCriticError("Scientific critique findings/revisions must be arrays")
    blocking = []
    for finding in findings:
        if not isinstance(finding, dict) or finding.get("severity") not in {
            "critical",
            "major",
            "minor",
            "note",
        }:
            raise ScientificCriticError("Malformed scientific finding")
        if finding.get("severity") in {"critical", "major"}:
            blocking.append(finding.get("code"))
    verdict = critique.get("verdict")
    if verdict == "pass":
        if not deterministic_validation_passed:
            raise ScientificCriticError("Critic cannot pass an invalid evidence report")
        if blocking or revisions:
            raise ScientificCriticError("Pass verdict retains blocking findings or revisions")
    elif verdict == "revise":
        if not revisions:
            raise ScientificCriticError("Revise verdict requires concrete revisions")
    elif verdict == "reject":
        if not findings:
            raise ScientificCriticError("Reject verdict requires findings")
    else:
        raise ScientificCriticError("Unknown scientific critique verdict")


def _load_report_sources(report: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    source_refs = report.get("source_contracts")
    required = {
        "pipeline_lock",
        "confirmation_result",
        "frozen_protocol",
    }
    if not isinstance(source_refs, dict) or not required.issubset(source_refs):
        raise ScientificCriticError("Evidence report lacks required source contracts")
    loaded: dict[str, dict[str, Any]] = {}
    for name, ref in source_refs.items():
        if not isinstance(ref, dict):
            raise ScientificCriticError(f"Malformed source contract {name}")
        path = Path(str(ref.get("path"))).expanduser().resolve()
        if not path.is_file() or sha256_path(path) != ref.get("sha256"):
            raise ScientificCriticError(f"Source contract failed integrity check: {name}")
        loaded[name] = load_json_object(path)
    decision = evaluate_frozen_decision(
        protocol=loaded["frozen_protocol"],
        pipeline_lock=loaded["pipeline_lock"],
        confirmation_result=loaded["confirmation_result"],
    )
    return loaded, decision


def create_scientific_critic_tools(
    *, evidence_report_path: Path
) -> tuple[ToolRegistry, dict[str, Any]]:
    report_path = Path(evidence_report_path).expanduser().resolve()
    report = load_json_object(report_path)
    report_hash = sha256_path(report_path)
    errors: list[str] = []
    loaded: dict[str, dict[str, Any]] = {}
    decision: dict[str, Any] = {}
    try:
        loaded, decision = _load_report_sources(report)
        validate_evidence_report(
            report,
            protocol=loaded["frozen_protocol"],
            pipeline_lock=loaded["pipeline_lock"],
            confirmation_result=loaded["confirmation_result"],
            deterministic_decision=decision,
        )
        if report.get("deterministic_decision") != decision:
            raise EvidenceReportError("Embedded deterministic decision differs from recomputation")
    except (EvidenceReportError, ScientificCriticError, KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    passed = not errors
    registry = ToolRegistry()
    context_read = False
    recorded = False

    def read_context() -> dict[str, Any]:
        nonlocal context_read
        context_read = True
        return {
            "evidence_report": report,
            "recomputed_deterministic_decision": decision,
            "deterministic_validation": {"passed": passed, "errors": errors},
            "report_provenance": {"path": str(report_path), "sha256": report_hash},
            "external_claim_authorized": False,
        }

    def record(critique: dict[str, Any]) -> dict[str, Any]:
        nonlocal recorded
        if not context_read:
            raise ScientificCriticError("read_scientific_critic_context must be called first")
        if recorded:
            raise ScientificCriticError("Only one scientific critique may be recorded")
        validate_scientific_critique(
            critique,
            report=report,
            report_sha256=report_hash,
            deterministic_validation_passed=passed,
        )
        recorded = True
        result = json.loads(json.dumps(critique))
        result["source_report"] = {"path": str(report_path), "sha256": report_hash}
        result["external_claim_authorized"] = False
        return result

    registry.register(
        ToolDefinition(
            name="read_scientific_critic_context",
            description="Read the internal evidence report and recomputed deterministic checks.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            approval="never",
            decision_kind="independent_scientific_review",
            tags=("read-only", "critic", "post-confirmation"),
        ),
        read_context,
    )
    registry.register(
        ToolDefinition(
            name="record_scientific_critique",
            description="Record one pass, revise, or reject verdict for the internal report.",
            input_schema={
                "type": "object",
                "properties": {"critique": scientific_critique_schema()},
                "required": ["critique"],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="autonomous_scientific_critique",
            tags=("local-write", "critic", "finalization-gate"),
        ),
        record,
    )
    return registry, {
        "dataset_id": report.get("dataset_id"),
        "subject_id": report.get("subject_id"),
        "report_id": report.get("report_id"),
        "task": "independent_internal_scientific_review",
        "deterministic_validation_passed": passed,
    }


@dataclass
class ScientificCriticAgent:
    runtime: AgentRuntime
    context: dict[str, Any]

    def run(self) -> AgentRunResult:
        return self.runtime.run(
            system_prompt=SCIENTIFIC_CRITIC_SYSTEM_PROMPT,
            user_prompt=json.dumps(self.context, ensure_ascii=False, sort_keys=True),
        )
