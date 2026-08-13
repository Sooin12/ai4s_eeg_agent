"""Independent review of a complete dataset-level understanding draft."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci_autodiscovery.literature import LiteratureStore
from bci_autodiscovery.profiling import (
    dataset_profile_field_catalog,
    validate_dataset_profile_provenance,
)
from bci_autodiscovery.search import (
    build_combined_search_space_review,
    build_search_space_draft,
    validate_dataset_level_review_draft,
)
from .contracts import AgentRunResult
from .language import DEFAULT_LANGUAGE_INSTRUCTION
from .runtime import AgentRuntime
from .tools import ToolDefinition, ToolRegistry


DATASET_INTELLIGENCE_CRITIC_PROMPT = f"""{DEFAULT_LANGUAGE_INSTRUCTION}
You are the independent Dataset Critic. First call read_dataset_critic_context. Audit only
the dataset-level task: local structure/semantics evidence, hard compatibility exclusions,
scholarly-network coverage, frontier citations, limitations, uncertainty, and downstream
stage isolation. Do not assign session roles, metrics, budgets, subject pipelines, or
confirmation access.

The deterministic validation is mandatory. Also check whether frontier claims overstate
metadata/abstract evidence, whether the proposed coarse range omits plausible compatible
families, and whether unsupported assumptions are presented as facts. Profile fields that
explicitly preserve unknown metadata and canonical components that are explicitly deferred
to downstream Research Design are not by themselves blocking defects; never demand that an
unknown value be guessed.

External-authority blockers must remain visible in the contract. They gate later dataset
activation or disclosure; they do not by themselves invalidate an otherwise traceable
dataset-level understanding. Metadata-only discovery is also not full-text verification:
do not require DOI resolution or full-text access at this stage, and judge every cited
identifier against the local evidence ledger while preserving claim limitations.

The active revision loop can modify only Literature Scout-authored frontier directions and
their evidence selection. Set finding owner=literature_scout only for those issues. Findings
owned by dataset_profiler_adapter, canonical_registry, deterministic_builder, or
external_authority cannot be repaired by a Literature revision: use reject for a truly
blocking defect, or a non-blocking minor/note finding with pass. Never issue revise while a
critical/major finding is owned by another layer. Call
record_dataset_critique exactly once successfully with pass, revise, or reject, then call
inspect_dataset_critic_status. A repairable issue must produce concrete structured
revisions; do not transfer scientific decisions to the user."""


class DatasetIntelligenceCriticError(ValueError):
    pass


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetIntelligenceCriticError(f"Cannot load JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetIntelligenceCriticError(f"JSON artifact must be an object: {path}")
    return value


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_dataset_level_sources(
    draft: dict[str, Any], *, draft_path: Path
) -> dict[str, Any]:
    validate_dataset_level_review_draft(draft)
    provenance = draft.get("provenance") or {}
    required = {
        "dataset_profile",
        "canonical_search_space",
        "component_registry",
        "evidence_db",
    }
    if not required.issubset(provenance):
        raise DatasetIntelligenceCriticError(
            f"Dataset draft lacks provenance: {sorted(required.difference(provenance))}"
        )
    loaded_paths: dict[str, Path] = {}
    for name in required:
        ref = provenance[name]
        if not isinstance(ref, dict):
            raise DatasetIntelligenceCriticError(f"Malformed provenance reference: {name}")
        path = Path(str(ref.get("path"))).expanduser().resolve()
        if not path.is_file() or _sha256_path(path) != ref.get("sha256"):
            raise DatasetIntelligenceCriticError(
                f"Dataset draft provenance failed integrity check: {name}"
            )
        loaded_paths[name] = path

    profile = _load_json_object(loaded_paths["dataset_profile"])
    validate_dataset_profile_provenance(
        profile,
        require_hashed_evidence=True,
        require_current_constraints=True,
    )
    if profile["dataset"]["id"] != draft.get("dataset_id"):
        raise DatasetIntelligenceCriticError("DatasetProfile and review use different datasets")
    profile_field_paths = set(dataset_profile_field_catalog(profile))
    canonical = _load_json_object(loaded_paths["canonical_search_space"])
    rebuilt = build_search_space_draft(
        dataset_profile_path=str(loaded_paths["dataset_profile"]),
        component_registry_path=str(loaded_paths["component_registry"]),
    )
    if canonical != rebuilt:
        raise DatasetIntelligenceCriticError(
            "Canonical coarse space differs from deterministic recomputation"
        )
    if draft.get("canonical_space") != canonical.get("canonical_space"):
        raise DatasetIntelligenceCriticError("Merged draft changed the canonical coarse space")
    if draft.get("excluded_components") != canonical.get("excluded_components"):
        raise DatasetIntelligenceCriticError("Merged draft changed hard exclusions")

    frontier = draft.get("frontier_space") or {}
    literature_run_id = str(frontier.get("literature_run_id") or "")
    if not literature_run_id:
        raise DatasetIntelligenceCriticError("Dataset draft lacks literature_run_id")
    rebuilt_review = build_combined_search_space_review(
        canonical_search_space_path=loaded_paths["canonical_search_space"],
        evidence_db_path=loaded_paths["evidence_db"],
        literature_run_id=literature_run_id,
    )
    if draft != rebuilt_review:
        raise DatasetIntelligenceCriticError(
            "Dataset-level draft differs from deterministic evidence-ledger recomputation"
        )
    store = LiteratureStore(loaded_paths["evidence_db"])
    known_ids = store.known_paper_ids(search_run_id=literature_run_id)
    for direction in frontier.get("directions") or []:
        for field in (
            "candidate_id",
            "method_family",
            "claim",
            "novelty_level",
            "proposed_validation",
        ):
            if not isinstance(direction.get(field), str) or not direction[field].strip():
                raise DatasetIntelligenceCriticError(
                    f"Frontier direction has empty {field}"
                )
        for field in (
            "pipeline_stages",
            "applicability",
            "limitations",
            "supporting_papers",
            "future_protocol_requirements",
        ):
            values = direction.get(field)
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(item, str) or not item.strip() for item in values)
            ):
                raise DatasetIntelligenceCriticError(
                    f"Frontier direction has invalid {field}"
                )
        missing = sorted(set(direction.get("supporting_papers") or []).difference(known_ids))
        if missing:
            raise DatasetIntelligenceCriticError(
                f"Frontier direction {direction.get('candidate_id')} cites unknown evidence: {missing}"
            )
        if direction.get("evidence_scope") != "scholarly_metadata_or_abstract_discovery_only":
            raise DatasetIntelligenceCriticError("Frontier direction overstates its evidence scope")
        binding = direction.get("dataset_binding")
        if not isinstance(binding, dict):
            raise DatasetIntelligenceCriticError(
                f"Frontier direction {direction.get('candidate_id')} lacks dataset binding"
            )
        supporting_fields = binding.get("supporting_profile_fields")
        if (
            not isinstance(supporting_fields, list)
            or not supporting_fields
            or any(
                not isinstance(item, str) or not item.strip()
                for item in supporting_fields
            )
        ):
            raise DatasetIntelligenceCriticError(
                f"Frontier direction {direction.get('candidate_id')} has invalid "
                "supporting_profile_fields"
            )
        invalid_fields = sorted(set(supporting_fields).difference(profile_field_paths))
        if invalid_fields:
            raise DatasetIntelligenceCriticError(
                f"Frontier direction {direction.get('candidate_id')} cites invalid "
                f"DatasetProfile fields: {invalid_fields}"
            )
    return {
        "dataset_profile": profile,
        "canonical_search_space": canonical,
        "loaded_paths": {name: str(path) for name, path in loaded_paths.items()},
        "review_path": str(Path(draft_path).expanduser().resolve()),
    }


def dataset_critique_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "enum": ["1.0"]},
            "review_id": {"type": "string"},
            "dataset_id": {"type": "string"},
            "reviewed_draft_sha256": {"type": "string"},
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
                                "dataset_semantics",
                                "evidence_integrity",
                                "canonical_coverage",
                                "hard_exclusions",
                                "network_coverage",
                                "frontier_claims",
                                "uncertainty",
                                "stage_boundary",
                            ],
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["critical", "major", "minor", "note"],
                        },
                        "owner": {
                            "type": "string",
                            "enum": [
                                "literature_scout",
                                "dataset_profiler_adapter",
                                "canonical_registry",
                                "deterministic_builder",
                                "external_authority",
                                "none",
                            ],
                        },
                        "message": {"type": "string"},
                        "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "code",
                        "dimension",
                        "severity",
                        "owner",
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
            "reviewed_draft_sha256",
            "verdict",
            "findings",
            "required_revisions",
            "rationale",
        ],
        "additionalProperties": False,
    }


def validate_dataset_critique(
    critique: dict[str, Any],
    *,
    draft: dict[str, Any],
    draft_sha256: str,
    deterministic_validation_passed: bool,
) -> None:
    if critique.get("schema_version") != "1.0":
        raise DatasetIntelligenceCriticError("Unsupported Dataset Critique schema_version")
    if critique.get("dataset_id") != draft.get("dataset_id"):
        raise DatasetIntelligenceCriticError("Dataset Critique belongs to another dataset")
    if critique.get("reviewed_draft_sha256") != draft_sha256:
        raise DatasetIntelligenceCriticError("Dataset Critique is not bound to exact draft SHA")
    for field in ("review_id", "rationale"):
        if not isinstance(critique.get(field), str) or not critique[field].strip():
            raise DatasetIntelligenceCriticError(f"Dataset Critique has empty {field}")
    findings = critique.get("findings")
    revisions = critique.get("required_revisions")
    if not isinstance(findings, list) or not isinstance(revisions, list):
        raise DatasetIntelligenceCriticError("Critique findings and revisions must be arrays")
    if any(not isinstance(item, str) or not item.strip() for item in revisions):
        raise DatasetIntelligenceCriticError("Dataset Critique revisions must be non-empty text")
    codes: set[str] = set()
    blocking: list[str] = []
    blocking_owners: dict[str, str] = {}
    allowed_dimensions = {
        "dataset_semantics",
        "evidence_integrity",
        "canonical_coverage",
        "hard_exclusions",
        "network_coverage",
        "frontier_claims",
        "uncertainty",
        "stage_boundary",
    }
    allowed_severities = {"critical", "major", "minor", "note"}
    allowed_owners = {
        "literature_scout",
        "dataset_profiler_adapter",
        "canonical_registry",
        "deterministic_builder",
        "external_authority",
        "none",
    }
    for finding in findings:
        if not isinstance(finding, dict):
            raise DatasetIntelligenceCriticError("Dataset Critique finding must be an object")
        code = str(finding.get("code") or "").strip()
        if not code or code in codes:
            raise DatasetIntelligenceCriticError("Dataset Critique finding codes must be unique")
        codes.add(code)
        if finding.get("dimension") not in allowed_dimensions:
            raise DatasetIntelligenceCriticError("Dataset Critique finding dimension is invalid")
        if finding.get("severity") not in allowed_severities:
            raise DatasetIntelligenceCriticError("Dataset Critique finding severity is invalid")
        owner = finding.get("owner")
        if owner not in allowed_owners:
            raise DatasetIntelligenceCriticError("Dataset Critique finding owner is invalid")
        if not isinstance(finding.get("message"), str) or not finding["message"].strip():
            raise DatasetIntelligenceCriticError("Dataset Critique finding message is empty")
        refs = finding.get("evidence_refs")
        if not isinstance(refs, list) or any(
            not isinstance(item, str) or not item.strip() for item in refs
        ):
            raise DatasetIntelligenceCriticError(
                "Dataset Critique evidence_refs must be a text array"
            )
        if finding.get("severity") in {"critical", "major"}:
            blocking.append(code)
            blocking_owners[code] = str(owner)
    verdict = critique.get("verdict")
    if verdict == "pass":
        if not deterministic_validation_passed:
            raise DatasetIntelligenceCriticError(
                "Dataset Critic cannot pass a deterministically invalid draft"
            )
        if blocking or revisions:
            raise DatasetIntelligenceCriticError(
                "Pass verdict retains blocking findings or required revisions"
            )
    elif verdict == "revise":
        if not revisions:
            raise DatasetIntelligenceCriticError("Revise verdict requires concrete revisions")
        misrouted = sorted(
            code
            for code, owner in blocking_owners.items()
            if owner != "literature_scout"
        )
        if misrouted:
            raise DatasetIntelligenceCriticError(
                "Revise verdict routes blocking findings to a layer that Literature "
                f"Scout cannot modify: {misrouted}"
            )
    elif verdict == "reject":
        if not findings:
            raise DatasetIntelligenceCriticError("Reject verdict requires findings")
    else:
        raise DatasetIntelligenceCriticError("Unknown Dataset Critique verdict")


def create_dataset_intelligence_critic_tools(
    *, dataset_level_draft_path: Path
) -> tuple[ToolRegistry, dict[str, Any]]:
    draft_path = Path(dataset_level_draft_path).expanduser().resolve()
    draft = _load_json_object(draft_path)
    draft_hash = _sha256_path(draft_path)
    errors: list[str] = []
    sources: dict[str, Any] = {}
    try:
        sources = validate_dataset_level_sources(draft, draft_path=draft_path)
    except (DatasetIntelligenceCriticError, KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    passed = not errors
    registry = ToolRegistry()
    context_read = False
    recorded = False

    def read_context() -> dict[str, Any]:
        nonlocal context_read
        context_read = True
        return {
            "dataset_level_draft": draft,
            "dataset_profile": sources.get("dataset_profile"),
            "deterministic_validation": {"passed": passed, "errors": errors},
            "draft_provenance": {"path": str(draft_path), "sha256": draft_hash},
            "forbidden_context": {
                "subject_measurements_available": False,
                "experiment_outcomes_available": False,
                "confirmation_available": False,
            },
            "revision_capability": {
                "modifiable_owner": "literature_scout",
                "modifiable_content": [
                    "frontier direction claims",
                    "frontier evidence citations",
                    "frontier applicability and limitations",
                    "frontier proposed validation",
                ],
                "immutable_in_this_loop": [
                    "DatasetProfile",
                    "component registry",
                    "canonical coarse space",
                    "deterministic builder rules",
                ],
            },
        }

    def record(critique: dict[str, Any]) -> dict[str, Any]:
        nonlocal recorded
        if not context_read:
            raise DatasetIntelligenceCriticError("read_dataset_critic_context must be called first")
        if recorded:
            raise DatasetIntelligenceCriticError("Only one Dataset Critique may be recorded")
        validate_dataset_critique(
            critique,
            draft=draft,
            draft_sha256=draft_hash,
            deterministic_validation_passed=passed,
        )
        recorded = True
        result = json.loads(json.dumps(critique))
        result["source_draft"] = {"path": str(draft_path), "sha256": draft_hash}
        result["critic_independence"] = {
            "authored_draft": False,
            "subject_data_available": False,
            "outcomes_available": False,
            "confirmation_available": False,
        }
        return result

    registry.register(
        ToolDefinition(
            name="read_dataset_critic_context",
            description="Read the complete dataset-level draft and deterministic checks.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            approval="never",
            decision_kind="independent_dataset_level_review",
            tags=("read-only", "dataset-critic", "no-subject-data"),
        ),
        read_context,
    )
    registry.register(
        ToolDefinition(
            name="record_dataset_critique",
            description="Record one independent pass, revise, or reject verdict.",
            input_schema={
                "type": "object",
                "properties": {"critique": dataset_critique_schema()},
                "required": ["critique"],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="autonomous_dataset_level_critique",
            tags=("local-write", "dataset-critic", "freeze-gate"),
        ),
        record,
    )

    def status() -> dict[str, Any]:
        return {
            "context_read": context_read,
            "critique_recorded": recorded,
            "complete": recorded,
        }

    registry.register(
        ToolDefinition(
            name="inspect_dataset_critic_status",
            description="Inspect whether the independent critique was recorded successfully.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="read_only_status",
            tags=("read-only", "completion-check", "dataset-critic"),
        ),
        status,
    )
    return registry, {
        "dataset_id": draft.get("dataset_id"),
        "task": "independent_dataset_level_review",
        "deterministic_validation_passed": passed,
    }


@dataclass
class DatasetIntelligenceCriticAgent:
    runtime: AgentRuntime
    context: dict[str, Any]

    def run(self) -> AgentRunResult:
        def completion_check() -> dict[str, Any]:
            return self.runtime.tools.execute("inspect_dataset_critic_status", {})

        return self.runtime.run(
            system_prompt=DATASET_INTELLIGENCE_CRITIC_PROMPT,
            user_prompt=json.dumps(self.context, ensure_ascii=False, sort_keys=True),
            completion_check=completion_check,
            complete_on_tool_state=True,
        )
