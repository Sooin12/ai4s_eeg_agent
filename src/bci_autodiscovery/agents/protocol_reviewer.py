"""Dataset-neutral protocol review/revision agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import AgentRunResult
from .language import DEFAULT_LANGUAGE_INSTRUCTION
from .protocol_planner import (
    _hash,
    _load_profile,
    protocol_proposal_schema,
    validate_protocol_proposal,
)
from .runtime import AgentRuntime
from .tools import ToolDefinition, ToolRegistry
from bci_autodiscovery.workflow.protocol_artifacts import load_json


PROTOCOL_REVIEWER_SYSTEM_PROMPT = f"""{DEFAULT_LANGUAGE_INSTRUCTION}
You are the Protocol Review Agent in an auditable research system. First call
read_protocol_review_context to read the dataset profile, current proposal, and verbatim
user feedback. Respond only from those contracts. Do not access raw signals or pretend the
user granted approval that was not explicitly given.

If the user requests an explanation without a change, call record_protocol_explanation.
If the user requests changes, generate a complete revised proposal and call
record_protocol_revision; do not output only a patch. The revised proposal must remain
proposed_requires_human_approval, satisfy all leakage safeguards, and include a change
summary and itemized feedback resolution. You have no tool to approve or activate the
protocol. Record one result, then stop."""


def create_protocol_reviewer_tools(
    *, dataset_profile_path: Path, proposal_path: Path, user_feedback: str
) -> tuple[ToolRegistry, dict[str, Any]]:
    profile_path = Path(dataset_profile_path).expanduser().resolve()
    current_path = Path(proposal_path).expanduser().resolve()
    profile = _load_profile(profile_path)
    proposal = load_json(current_path)
    dataset_id = str(profile["dataset"]["id"])
    if proposal.get("dataset_id") != dataset_id:
        raise ValueError("Dataset profile and protocol proposal refer to different datasets")
    if proposal.get("status") != "proposed_requires_human_approval":
        raise ValueError("Protocol Reviewer only accepts a pending proposal")
    if not user_feedback.strip():
        raise ValueError("User feedback cannot be empty")
    registry = ToolRegistry()
    context_read = False
    result_recorded = False

    def read_context() -> dict[str, Any]:
        nonlocal context_read
        context_read = True
        return {
            "dataset_profile": profile,
            "current_proposal": proposal,
            "user_feedback_verbatim": user_feedback,
            "provenance": {
                "dataset_profile": {"path": str(profile_path), "sha256": _hash(profile_path)},
                "current_proposal": {"path": str(current_path), "sha256": _hash(current_path)},
            },
        }

    def require_ready() -> None:
        if not context_read:
            raise ValueError("read_protocol_review_context must be called first")
        if result_recorded:
            raise ValueError("Only one review result may be recorded per run")

    def record_explanation(response_to_user: str) -> dict[str, Any]:
        nonlocal result_recorded
        require_ready()
        result_recorded = True
        return {
            "schema_version": "1.0",
            "dataset_id": dataset_id,
            "review_type": "explanation_only",
            "user_feedback_verbatim": user_feedback,
            "response_to_user": response_to_user,
            "proposal_changed": False,
            "source_proposal": {"path": str(current_path), "sha256": _hash(current_path)},
        }

    def record_revision(
        revised_proposal: dict[str, Any],
        change_summary: list[str],
        feedback_resolution: list[str],
    ) -> dict[str, Any]:
        nonlocal result_recorded
        require_ready()
        validate_protocol_proposal(revised_proposal, profile=profile, dataset_id=dataset_id)
        result_recorded = True
        result = json.loads(json.dumps(revised_proposal))
        result["activation_performed"] = False
        result["source_profile"] = {"path": str(profile_path), "sha256": _hash(profile_path)}
        result["revision"] = {
            "parent_path": str(current_path),
            "parent_sha256": _hash(current_path),
            "user_feedback_verbatim": user_feedback,
            "change_summary": change_summary,
            "feedback_resolution": feedback_resolution,
        }
        return result

    registry.register(
        ToolDefinition(
            name="read_protocol_review_context",
            description="Read the authoritative profile, current proposal, and verbatim user feedback.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            approval="never",
            decision_kind="read_only_contract",
            tags=("read-only", "protocol-review", "authoritative"),
        ),
        read_context,
    )
    registry.register(
        ToolDefinition(
            name="record_protocol_explanation",
            description="Record an explanation that does not change or approve the protocol.",
            input_schema={
                "type": "object",
                "properties": {"response_to_user": {"type": "string"}},
                "required": ["response_to_user"],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="scientific_explanation_only",
            tags=("local-write", "review-record"),
        ),
        record_explanation,
    )
    registry.register(
        ToolDefinition(
            name="record_protocol_revision",
            description="Record a complete revised proposal; it remains pending human approval.",
            input_schema={
                "type": "object",
                "properties": {
                    "revised_proposal": protocol_proposal_schema(),
                    "change_summary": {"type": "array", "items": {"type": "string"}},
                    "feedback_resolution": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["revised_proposal", "change_summary", "feedback_resolution"],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="scientific_revision_only",
            tags=("local-write", "revision", "human-approval-required"),
        ),
        record_revision,
    )
    return registry, {
        "dataset_id": dataset_id,
        "task": "review_or_revise_protocol_without_approving_it",
        "user_feedback_verbatim": user_feedback,
    }


@dataclass
class ProtocolReviewerAgent:
    runtime: AgentRuntime
    context: dict[str, Any]

    def run(self) -> AgentRunResult:
        return self.runtime.run(
            system_prompt=PROTOCOL_REVIEWER_SYSTEM_PROMPT,
            user_prompt=json.dumps(self.context, ensure_ascii=False, sort_keys=True),
        )
