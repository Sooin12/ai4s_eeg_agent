"""Independent critic for autonomous, outcome-blind research protocols."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci_autodiscovery.workflow.autonomy import (
    load_json_object,
    sha256_path,
)

from .contracts import AgentRunResult
from .language import DEFAULT_LANGUAGE_INSTRUCTION
from .research_protocol import (
    ResearchProtocolError,
    _load_authorized_research_context,
    validate_research_protocol_authority_bindings,
    validate_research_protocol_proposal,
)
from .runtime import AgentRuntime
from .tools import ToolArgumentError, ToolDefinition, ToolRegistry, validate_json_value


PROTOCOL_CRITIC_SYSTEM_PROMPT = f"""{DEFAULT_LANGUAGE_INSTRUCTION}
You are an independent Protocol Critic. First call read_protocol_critic_context. You did not
author the proposal and cannot access experiment outcomes or frozen-confirmation results.

Audit the proposal for dataset-contract conflicts, leakage, outcome peeking, incomplete
decision resolution, unsuitable metrics or statistical tests, ill-defined oracle, budget
overreach, weak stopping/refusal rules, and unsupported quality exclusions. Treat the
deterministic validation result as mandatory but also perform a semantic scientific review.

Call record_protocol_critique exactly once with pass, revise, or reject. Pass only when no
critical or major finding remains and no revision is required. Do not ask a human to choose
the scientific design. Assign every finding to its true owner. Use revise only when every
blocking finding is owned by research_protocol_planner; immutable dataset, authority,
validator, engineering, or omitted/mutated external-authority blockers must fail closed
with reject.
An external-authority issue faithfully preserved as an unresolved execution precondition
is not itself a Research Design defect: it may remain while the outcome-blind protocol
freezes, but pipeline execution and confirmation access must stay blocked.
"""


class ProtocolCriticError(ValueError):
    pass


def protocol_critique_schema() -> dict[str, Any]:
    owners = [
        "research_protocol_planner",
        "dataset_contract",
        "autonomy_authority",
        "method_engineering",
        "deterministic_validator",
        "external_authority",
        "none",
    ]
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "enum": ["2.0"]},
            "review_id": {"type": "string"},
            "dataset_id": {"type": "string"},
            "protocol_id": {"type": "string"},
            "reviewed_protocol_sha256": {"type": "string"},
            "verdict": {
                "type": "string",
                "enum": ["pass", "revise", "reject"],
            },
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "severity": {
                            "type": "string",
                            "enum": ["critical", "major", "minor", "note"],
                        },
                        "owner": {"type": "string", "enum": owners},
                        "message": {"type": "string"},
                        "evidence_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "code",
                        "severity",
                        "owner",
                        "message",
                        "evidence_refs",
                    ],
                    "additionalProperties": False,
                },
            },
            "required_revisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "finding_code": {"type": "string"},
                        "instruction": {"type": "string"},
                    },
                    "required": ["finding_code", "instruction"],
                    "additionalProperties": False,
                },
            },
            "rationale": {"type": "string"},
        },
        "required": [
            "schema_version",
            "review_id",
            "dataset_id",
            "protocol_id",
            "reviewed_protocol_sha256",
            "verdict",
            "findings",
            "required_revisions",
            "rationale",
        ],
        "additionalProperties": False,
    }


def validate_protocol_critique(
    critique: dict[str, Any],
    *,
    proposal: dict[str, Any],
    proposal_sha256: str,
    deterministic_validation_passed: bool,
) -> None:
    schema = protocol_critique_schema()
    schema_fields = set(schema["properties"])
    allowed_metadata = {"activation_state", "source_proposal", "critic_independence"}
    unknown = sorted(set(critique).difference(schema_fields | allowed_metadata))
    if unknown:
        raise ProtocolCriticError(f"Protocol critique has unknown fields: {unknown}")
    core = {field: critique[field] for field in schema_fields if field in critique}
    try:
        validate_json_value(core, schema, location="protocol_critique")
    except ToolArgumentError as exc:
        raise ProtocolCriticError(str(exc)) from exc
    if critique.get("schema_version") != "2.0":
        raise ProtocolCriticError("Unsupported protocol critique schema_version")
    if critique.get("dataset_id") != proposal.get("dataset_id"):
        raise ProtocolCriticError("Critique belongs to another dataset")
    if critique.get("protocol_id") != proposal.get("protocol_id"):
        raise ProtocolCriticError("Critique belongs to another protocol")
    if critique.get("reviewed_protocol_sha256") != proposal_sha256:
        raise ProtocolCriticError("Critique is not bound to the exact proposal SHA")
    if not isinstance(critique.get("review_id"), str) or not critique["review_id"].strip():
        raise ProtocolCriticError("review_id must be non-empty")
    if not isinstance(critique.get("rationale"), str) or not critique["rationale"].strip():
        raise ProtocolCriticError("Critique rationale must be non-empty")

    findings = critique.get("findings")
    revisions = critique.get("required_revisions")
    if not isinstance(findings, list) or not isinstance(revisions, list):
        raise ProtocolCriticError("findings and required_revisions must be arrays")
    revision_codes: set[str] = set()
    for revision in revisions:
        if not isinstance(revision, dict):
            raise ProtocolCriticError("Every required revision must be an object")
        code = str(revision.get("finding_code") or "").strip()
        instruction = str(revision.get("instruction") or "").strip()
        if not code or code in revision_codes or not instruction:
            raise ProtocolCriticError(
                "Required revisions need unique finding_code and non-empty instruction"
            )
        revision_codes.add(code)
    codes: set[str] = set()
    owners = {
        "research_protocol_planner",
        "dataset_contract",
        "autonomy_authority",
        "method_engineering",
        "deterministic_validator",
        "external_authority",
        "none",
    }
    for finding in findings:
        if not isinstance(finding, dict):
            raise ProtocolCriticError("Every critic finding must be an object")
        code = str(finding.get("code") or "").strip()
        if not code or code in codes:
            raise ProtocolCriticError("Critic finding codes must be non-empty and unique")
        codes.add(code)
        if finding.get("severity") not in {"critical", "major", "minor", "note"}:
            raise ProtocolCriticError(f"Finding {code} has invalid severity")
        if finding.get("owner") not in owners:
            raise ProtocolCriticError(f"Finding {code} has invalid owner")
        if not isinstance(finding.get("message"), str) or not finding["message"].strip():
            raise ProtocolCriticError(f"Finding {code} has an empty message")
        refs = finding.get("evidence_refs")
        if not isinstance(refs, list) or any(
            not isinstance(item, str) or not item.strip() for item in refs
        ):
            raise ProtocolCriticError(f"Finding {code} has invalid evidence_refs")

    verdict = critique.get("verdict")
    if not revision_codes.issubset(codes):
        raise ProtocolCriticError("A required revision references an unknown finding")
    blocking_findings = [
        item for item in findings if item.get("severity") in {"critical", "major"}
    ]
    if verdict == "pass":
        if not deterministic_validation_passed:
            raise ProtocolCriticError("Critic cannot pass a deterministically invalid protocol")
        blocking = [item["code"] for item in blocking_findings]
        if blocking or revisions:
            raise ProtocolCriticError(
                f"Pass verdict cannot retain blocking findings or revisions: {blocking}"
            )
    elif verdict == "revise":
        if not revisions:
            raise ProtocolCriticError("Revise verdict requires concrete revisions")
        non_planner = [
            item["code"]
            for item in blocking_findings
            if item.get("owner") != "research_protocol_planner"
        ]
        if non_planner:
            raise ProtocolCriticError(
                "Revise cannot route non-planner blocking findings to the planner: "
                f"{non_planner}"
            )
        planner_blocking = {
            item["code"]
            for item in blocking_findings
            if item.get("owner") == "research_protocol_planner"
        }
        if revision_codes != planner_blocking:
            raise ProtocolCriticError(
                "Required revisions must map exactly to planner-owned blocking findings"
            )
    elif verdict == "reject":
        if not blocking_findings:
            raise ProtocolCriticError("Reject verdict requires a blocking finding")
        if revisions:
            raise ProtocolCriticError("Reject verdict cannot request planner revisions")
    else:
        raise ProtocolCriticError("Unknown critic verdict")


def create_protocol_critic_tools(
    *,
    dataset_level_contract_path: Path,
    autonomy_envelope_path: Path,
    proposal_path: Path,
) -> tuple[ToolRegistry, dict[str, Any]]:
    (
        contract_path,
        contract,
        profile_path,
        profile,
        envelope_path,
        envelope,
    ) = _load_authorized_research_context(
        dataset_level_contract_path=dataset_level_contract_path,
        autonomy_envelope_path=autonomy_envelope_path,
    )
    proposal_path = Path(proposal_path).expanduser().resolve()
    dataset_id = str(contract["dataset_id"])
    proposal = load_json_object(proposal_path)
    deterministic_errors: list[str] = []
    try:
        validate_research_protocol_proposal(
            proposal,
            dataset_contract=contract,
            profile=profile,
            envelope=envelope,
        )
        validate_research_protocol_authority_bindings(
            proposal,
            dataset_contract_path=contract_path,
            autonomy_envelope_path=envelope_path,
        )
    except (ResearchProtocolError, KeyError, TypeError) as exc:
        deterministic_errors.append(str(exc))
    deterministic_passed = not deterministic_errors
    proposal_hash = sha256_path(proposal_path)
    registry = ToolRegistry()
    context_read = False
    recorded = False

    def read_context() -> dict[str, Any]:
        nonlocal context_read
        context_read = True
        return {
            "autonomy_envelope": envelope,
            "dataset_level_contract": contract,
            "dataset_profile": profile,
            "protocol_proposal": proposal,
            "deterministic_validation": {
                "passed": deterministic_passed,
                "errors": deterministic_errors,
            },
            "finding_owner_routing": {
                "revision_allowed_owner": "research_protocol_planner",
                "reject_blocking_owners": [
                    "dataset_contract",
                    "autonomy_authority",
                    "method_engineering",
                    "deterministic_validator",
                    "external_authority",
                ],
                "nonblocking_owner": "none",
            },
            "provenance": {
                "autonomy_envelope": {
                    "path": str(envelope_path),
                    "sha256": sha256_path(envelope_path),
                },
                "dataset_level_contract": {
                    "path": str(contract_path),
                    "sha256": sha256_path(contract_path),
                },
                "dataset_profile": {
                    "path": str(profile_path),
                    "sha256": sha256_path(profile_path),
                },
                "protocol_proposal": {
                    "path": str(proposal_path),
                    "sha256": proposal_hash,
                },
            },
        }

    def record_critique(critique: dict[str, Any]) -> dict[str, Any]:
        nonlocal recorded
        if not context_read:
            raise ProtocolCriticError("read_protocol_critic_context must be called first")
        if recorded:
            raise ProtocolCriticError("Only one protocol critique may be recorded per run")
        validate_protocol_critique(
            critique,
            proposal=proposal,
            proposal_sha256=proposal_hash,
            deterministic_validation_passed=deterministic_passed,
        )
        recorded = True
        result = json.loads(json.dumps(critique))
        result["activation_state"] = {
            "protocol_frozen": False,
            "session_role_contract_activated": False,
            "raw_data_accessed": False,
            "confirmation_accessed": False,
            "pipeline_execution_started": False,
        }
        result["source_proposal"] = {
            "path": str(proposal_path),
            "sha256": proposal_hash,
        }
        result["critic_independence"] = {
            "outcomes_available": False,
            "confirmation_available": False,
            "authored_proposal": False,
        }
        return result

    registry.register(
        ToolDefinition(
            name="read_protocol_critic_context",
            description=(
                "Read the outcome-blind protocol, authoritative contracts, and deterministic "
                "validation result."
            ),
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            approval="never",
            decision_kind="independent_scientific_review",
            tags=("read-only", "critic", "outcome-blind"),
        ),
        read_context,
    )
    registry.register(
        ToolDefinition(
            name="record_protocol_critique",
            description="Record one independent pass, revise, or reject verdict.",
            input_schema={
                "type": "object",
                "properties": {"critique": protocol_critique_schema()},
                "required": ["critique"],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="autonomous_protocol_critique",
            tags=("local-write", "critic", "freeze-gate"),
        ),
        record_critique,
    )
    return registry, {
        "dataset_id": dataset_id,
        "protocol_id": proposal.get("protocol_id"),
        "task": "independently_review_outcome_blind_protocol",
        "deterministic_validation_passed": deterministic_passed,
    }


@dataclass
class ProtocolCriticAgent:
    runtime: AgentRuntime
    context: dict[str, Any]

    def run(self) -> AgentRunResult:
        return self.runtime.run(
            system_prompt=PROTOCOL_CRITIC_SYSTEM_PROMPT,
            user_prompt=json.dumps(self.context, ensure_ascii=False, sort_keys=True),
        )
