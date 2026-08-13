"""Typed, fail-closed ledger for progressing one dataset through the workflow."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_ORDER = (
    "dataset_inspection",
    "dataset_profile",
    "canonical_search_space",
    "frontier_discovery",
    "dataset_critic",
    "dataset_level_contract",
    "research_protocol_proposal",
    "protocol_critic",
    "research_protocol",
    "subject_profile",
    "pipeline_search",
    "pipeline_lock",
    "frozen_confirmation",
    "evidence_report",
)

STAGE_CONTRACTS = {
    "dataset_inspection": ("DatasetInspection", "validated"),
    "dataset_profile": ("DatasetProfile", "validated"),
    "canonical_search_space": ("CanonicalSearchSpace", "validated"),
    "frontier_discovery": ("FrontierDiscovery", "completed"),
    "dataset_critic": ("DatasetCritique", "pass"),
    "dataset_level_contract": (
        "DatasetLevelContract",
        "frozen_dataset_level_contract",
    ),
    "research_protocol_proposal": (
        "ResearchProtocolProposal",
        "proposed_for_autonomous_review",
    ),
    "protocol_critic": ("ProtocolCritique", "pass"),
    "research_protocol": ("ResearchProtocol", "frozen_autonomous"),
    "subject_profile": ("SubjectProfile", "validated"),
    "pipeline_search": ("PipelineSearchTrace", "completed"),
    "pipeline_lock": ("PipelineLock", "locked_awaiting_confirmation"),
    "frozen_confirmation": ("FrozenConfirmation", "completed_one_shot"),
    "evidence_report": ("EvidenceReport", "completed"),
}

_PREREQUISITES = {
    stage: list(STAGE_ORDER[:index]) for index, stage in enumerate(STAGE_ORDER)
}


class WorkflowTransitionError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_artifact(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowTransitionError(f"Stage artifact must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkflowTransitionError("Stage artifact must be a JSON object")
    return value


def _artifact_dataset_id(value: dict[str, Any]) -> str:
    direct = value.get("dataset_id")
    if direct:
        return str(direct)
    nested = value.get("dataset") or {}
    return str(nested.get("id") or nested.get("dataset_id") or "")


class WorkflowLedger:
    def __init__(self, *, path: Path, dataset_id: str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.dataset_id = dataset_id
        self._legacy_read_only = False
        if self.path.exists():
            self._state = json.loads(self.path.read_text(encoding="utf-8"))
            if self._state.get("dataset_id") != dataset_id:
                raise WorkflowTransitionError("Workflow ledger belongs to another dataset")
            self._legacy_read_only = self._state.get("schema_version") != "3.0"
        else:
            self._state = {
                "schema_version": "3.0",
                "dataset_id": dataset_id,
                "status": "in_progress",
                "stages": {},
            }

    def next_stage(self) -> str | None:
        completed = self._state["stages"]
        return next((stage for stage in STAGE_ORDER if stage not in completed), None)

    def record_stage(
        self,
        *,
        stage: str,
        artifact_path: Path,
        artifact_type: str,
        artifact_status: str,
        upstream_sha256: list[str],
    ) -> dict[str, Any]:
        if self._legacy_read_only:
            raise WorkflowTransitionError(
                "Legacy workflow ledgers are historical read-only artifacts"
            )
        if stage not in STAGE_ORDER:
            raise WorkflowTransitionError(f"Unknown workflow stage: {stage}")
        expected = self.next_stage()
        if stage != expected:
            raise WorkflowTransitionError(
                f"Expected next stage {expected!r}; cannot record {stage!r}"
            )
        missing = [
            item for item in _PREREQUISITES[stage] if item not in self._state["stages"]
        ]
        if missing:
            raise WorkflowTransitionError(f"Missing prerequisite stages: {missing}")
        expected_type, expected_status = STAGE_CONTRACTS[stage]
        if artifact_type != expected_type or artifact_status != expected_status:
            raise WorkflowTransitionError(
                f"Stage {stage} requires {expected_type}/{expected_status}, observed "
                f"{artifact_type}/{artifact_status}"
            )
        artifact = Path(artifact_path).expanduser().resolve()
        if not artifact.is_file():
            raise WorkflowTransitionError(f"Stage artifact does not exist: {artifact}")
        value = _load_artifact(artifact)
        if _artifact_dataset_id(value) != self.dataset_id:
            raise WorkflowTransitionError("Stage artifact belongs to another dataset")
        content_status = value.get("status", value.get("verdict"))
        if stage in {
            "dataset_critic",
            "dataset_level_contract",
            "research_protocol_proposal",
            "protocol_critic",
            "research_protocol",
            "pipeline_lock",
            "frozen_confirmation",
        } and content_status != expected_status:
            raise WorkflowTransitionError(
                f"Stage artifact content status is {content_status!r}, not {expected_status!r}"
            )
        if stage in {"dataset_critic", "protocol_critic"} and value.get("verdict") != "pass":
            raise WorkflowTransitionError("Critic stage requires an exact pass verdict")
        if stage == "dataset_level_contract":
            critic = value.get("dataset_critic") or {}
            if critic.get("verdict") != "pass":
                raise WorkflowTransitionError("DatasetLevelContract lacks a passing Critic")
        if stage == "research_protocol":
            critic = ((value.get("autonomous_freeze") or {}).get("critique") or {})
            if critic.get("verdict") != "pass":
                raise WorkflowTransitionError("ResearchProtocol lacks a passing Protocol Critic")

        prior_records = self._state["stages"]
        if prior_records:
            previous_stage = STAGE_ORDER[len(prior_records) - 1]
            expected_upstream = [prior_records[previous_stage]["sha256"]]
        else:
            expected_upstream = []
        if upstream_sha256 != expected_upstream:
            raise WorkflowTransitionError(
                "Stage upstream binding does not match the immediately preceding artifact"
            )
        record = {
            "artifact_type": artifact_type,
            "artifact_status": artifact_status,
            "artifact_path": str(artifact),
            "sha256": _sha256(artifact),
            "upstream_sha256": list(upstream_sha256),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "transition_authority": "deterministic_typed_stage_contract_v3",
            "human_itemized_approval_used": False,
        }
        self._state["stages"][stage] = record
        if self.next_stage() is None:
            self._state["status"] = "completed"
        self._write()
        return record

    def to_dict(self) -> dict[str, Any]:
        result = json.loads(json.dumps(self._state))
        result["next_stage"] = self.next_stage()
        result["legacy_read_only"] = self._legacy_read_only
        return result

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(self.path)
