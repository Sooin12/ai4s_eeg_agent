"""Persist, revise, criticize, and freeze one Dataset-Level Agent contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from bci_autodiscovery.agents.dataset_intelligence_critic import (
    DatasetIntelligenceCriticError,
    validate_dataset_critique,
    validate_dataset_level_sources,
)

from .autonomy import load_json_object, sha256_path
from .protocol_artifacts import atomic_json


class DatasetIntelligenceWorkflowError(ValueError):
    pass


def freeze_dataset_level_contract(
    *,
    dataset_level_draft_path: Path,
    dataset_critique_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    draft_path = Path(dataset_level_draft_path).expanduser().resolve()
    critique_path = Path(dataset_critique_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise DatasetIntelligenceWorkflowError("Frozen DatasetLevelContract already exists")
    draft = load_json_object(draft_path)
    critique = load_json_object(critique_path)
    draft_hash = sha256_path(draft_path)
    try:
        validate_dataset_level_sources(draft, draft_path=draft_path)
        validate_dataset_critique(
            critique,
            draft=draft,
            draft_sha256=draft_hash,
            deterministic_validation_passed=True,
        )
    except (DatasetIntelligenceCriticError, KeyError, TypeError, ValueError) as exc:
        raise DatasetIntelligenceWorkflowError(
            f"Dataset-level freeze gate failed: {exc}"
        ) from exc
    if critique.get("verdict") != "pass":
        raise DatasetIntelligenceWorkflowError("Dataset Critic did not pass the draft")
    source = critique.get("source_draft") or {}
    if (
        Path(str(source.get("path"))).expanduser().resolve() != draft_path
        or source.get("sha256") != draft_hash
    ):
        raise DatasetIntelligenceWorkflowError(
            "Dataset Critique is not bound to the exact draft file"
        )
    frozen = json.loads(json.dumps(draft))
    frozen["schema_version"] = "2.0"
    frozen["contract_id"] = (
        f"{draft['dataset_id']}-dataset-level-{draft_hash[:12]}"
    )
    frozen["status"] = "frozen_dataset_level_contract"
    frozen["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    frozen["dataset_critic"] = {
        "path": str(critique_path),
        "sha256": sha256_path(critique_path),
        "verdict": "pass",
    }
    frozen["freeze_record"] = {
        "source_draft": {"path": str(draft_path), "sha256": draft_hash},
        "human_itemized_approval_used": False,
        "network_discovery_completed": True,
        "executable_activation_performed": False,
        "session_protocol_created": False,
        "subject_or_confirmation_data_accessed": False,
    }
    atomic_json(output, frozen, refuse_overwrite=True)
    return frozen


DraftCallback = Callable[[int, Path | None, Path | None], dict[str, Any]]
CriticCallback = Callable[[Path, int], dict[str, Any]]


@dataclass
class DatasetIntelligenceLoopResult:
    status: str
    cycles: int = 0
    draft_paths: list[str] = field(default_factory=list)
    critique_paths: list[str] = field(default_factory=list)
    frozen_contract_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetIntelligenceLoop:
    """Dataset draft -> independent Critic -> supplemental discovery -> freeze."""

    def __init__(self, *, run_dir: Path, max_revision_cycles: int = 2) -> None:
        if max_revision_cycles < 0:
            raise ValueError("max_revision_cycles must be non-negative")
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.max_revision_cycles = max_revision_cycles

    def run(
        self,
        *,
        drafter: DraftCallback,
        critic: CriticCallback,
    ) -> DatasetIntelligenceLoopResult:
        state_path = self.run_dir / "dataset_intelligence_state.json"
        if state_path.exists():
            raise DatasetIntelligenceWorkflowError(
                f"Refusing to append to existing Dataset Intelligence loop: {self.run_dir}"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        result = DatasetIntelligenceLoopResult(status="in_progress")
        previous_draft: Path | None = None
        previous_critique: Path | None = None
        revision_count = 0
        try:
            while True:
                cycle = len(result.draft_paths) + 1
                result.cycles = cycle
                draft = drafter(cycle, previous_draft, previous_critique)
                draft_path = self.run_dir / f"dataset-draft-{cycle:04d}.json"
                atomic_json(draft_path, draft, refuse_overwrite=True)
                result.draft_paths.append(str(draft_path))
                if previous_draft is not None and sha256_path(draft_path) == sha256_path(
                    previous_draft
                ):
                    raise DatasetIntelligenceWorkflowError(
                        "Dataset revision did not change the criticized draft"
                    )
                deterministic_passed = True
                try:
                    validate_dataset_level_sources(draft, draft_path=draft_path)
                except (DatasetIntelligenceCriticError, KeyError, TypeError, ValueError):
                    deterministic_passed = False

                critique = critic(draft_path, cycle)
                critique_path = self.run_dir / f"dataset-critique-{cycle:04d}.json"
                atomic_json(critique_path, critique, refuse_overwrite=True)
                result.critique_paths.append(str(critique_path))
                validate_dataset_critique(
                    critique,
                    draft=draft,
                    draft_sha256=sha256_path(draft_path),
                    deterministic_validation_passed=deterministic_passed,
                )
                verdict = critique["verdict"]
                if verdict == "pass":
                    frozen_path = self.run_dir / "dataset_level_contract.json"
                    freeze_dataset_level_contract(
                        dataset_level_draft_path=draft_path,
                        dataset_critique_path=critique_path,
                        output_path=frozen_path,
                    )
                    result.status = "completed"
                    result.frozen_contract_path = str(frozen_path)
                    self._write_state(state_path, result)
                    return result
                if verdict == "reject":
                    result.status = "rejected"
                    result.error = (
                        critique.get("rationale") or "Dataset Critic rejected draft"
                    )
                    self._write_state(state_path, result)
                    return result
                revision_count += 1
                if revision_count > self.max_revision_cycles:
                    result.status = "revision_limit_reached"
                    result.error = "Maximum Dataset Intelligence revision cycles reached"
                    self._write_state(state_path, result)
                    return result
                previous_draft = draft_path
                previous_critique = critique_path
        except Exception as exc:
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
            self._write_state(state_path, result)
            return result

    @staticmethod
    def _write_state(path: Path, result: DatasetIntelligenceLoopResult) -> None:
        state = result.to_dict()
        state["artifacts"] = {
            "drafts": [
                {"path": item, "sha256": sha256_path(Path(item))}
                for item in result.draft_paths
            ],
            "critiques": [
                {"path": item, "sha256": sha256_path(Path(item))}
                for item in result.critique_paths
            ],
        }
        if result.frozen_contract_path:
            state["artifacts"]["frozen_contract"] = {
                "path": result.frozen_contract_path,
                "sha256": sha256_path(Path(result.frozen_contract_path)),
            }
        atomic_json(path, state, refuse_overwrite=True)
