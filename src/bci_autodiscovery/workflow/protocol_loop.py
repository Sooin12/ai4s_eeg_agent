"""Orchestrate Planner -> Critic -> revision -> deterministic protocol freeze."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from bci_autodiscovery.agents.protocol_critic import validate_protocol_critique
from bci_autodiscovery.agents.research_protocol import (
    ResearchProtocolError,
    _load_authorized_research_context,
    validate_research_protocol_authority_bindings,
    validate_research_protocol_proposal,
)

from .autonomous_protocol import freeze_autonomous_protocol
from .autonomy import sha256_path
from .protocol_artifacts import atomic_json


PlannerCallback = Callable[[], dict[str, Any]]
CriticCallback = Callable[[Path, int], dict[str, Any]]
RevisionCallback = Callable[[Path, Path, int], dict[str, Any]]


class AutonomousProtocolLoopError(ValueError):
    pass


@dataclass
class AutonomousProtocolLoopResult:
    status: str
    cycles: int = 0
    proposal_paths: list[str] = field(default_factory=list)
    critique_paths: list[str] = field(default_factory=list)
    frozen_protocol_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AutonomousProtocolLoop:
    """Persist every autonomous design turn and freeze only an exact pass verdict."""

    def __init__(
        self,
        *,
        run_dir: Path,
        dataset_level_contract_path: Path,
        autonomy_envelope_path: Path,
        max_revision_cycles: int = 3,
    ) -> None:
        if max_revision_cycles < 0:
            raise ValueError("max_revision_cycles must be non-negative")
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.contract_path = Path(dataset_level_contract_path).expanduser().resolve()
        self.envelope_path = Path(autonomy_envelope_path).expanduser().resolve()
        self.max_revision_cycles = max_revision_cycles

    def run(
        self,
        *,
        planner: PlannerCallback,
        critic: CriticCallback,
        reviser: RevisionCallback,
    ) -> AutonomousProtocolLoopResult:
        state_path = self.run_dir / "protocol_loop_state.json"
        if state_path.exists():
            raise AutonomousProtocolLoopError(
                f"Refusing to append to an existing protocol loop: {self.run_dir}"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (
            contract_path,
            contract,
            _profile_path,
            profile,
            envelope_path,
            envelope,
        ) = _load_authorized_research_context(
            dataset_level_contract_path=self.contract_path,
            autonomy_envelope_path=self.envelope_path,
        )
        result = AutonomousProtocolLoopResult(status="in_progress")
        proposal = planner()
        revision_count = 0

        while True:
            cycle = len(result.proposal_paths) + 1
            result.cycles = cycle
            proposal_path = self.run_dir / f"proposal-{cycle:04d}.json"
            atomic_json(proposal_path, proposal, refuse_overwrite=True)
            result.proposal_paths.append(str(proposal_path))
            deterministic_passed = True
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
            except (ResearchProtocolError, KeyError, TypeError):
                deterministic_passed = False

            critique = critic(proposal_path, cycle)
            critique_path = self.run_dir / f"critique-{cycle:04d}.json"
            atomic_json(critique_path, critique, refuse_overwrite=True)
            result.critique_paths.append(str(critique_path))
            validate_protocol_critique(
                critique,
                proposal=proposal,
                proposal_sha256=sha256_path(proposal_path),
                deterministic_validation_passed=deterministic_passed,
            )

            verdict = critique["verdict"]
            if verdict == "pass":
                frozen_path = self.run_dir / "frozen_protocol.json"
                freeze_autonomous_protocol(
                    proposal_path=proposal_path,
                    critique_path=critique_path,
                    dataset_level_contract_path=self.contract_path,
                    autonomy_envelope_path=self.envelope_path,
                    expected_dataset_level_contract_sha256=sha256_path(
                        self.contract_path
                    ),
                    expected_autonomy_envelope_sha256=sha256_path(
                        self.envelope_path
                    ),
                    expected_proposal_sha256=sha256_path(proposal_path),
                    expected_critique_sha256=sha256_path(critique_path),
                    output_path=frozen_path,
                )
                result.status = "completed"
                result.frozen_protocol_path = str(frozen_path)
                self._write_state(state_path, result)
                return result
            if verdict == "reject":
                result.status = "rejected"
                result.error = critique.get("rationale") or "Protocol critic rejected design"
                self._write_state(state_path, result)
                return result

            revision_count += 1
            if revision_count > self.max_revision_cycles:
                result.status = "revision_limit_reached"
                result.error = "Maximum autonomous protocol revision cycles reached"
                self._write_state(state_path, result)
                return result
            proposal = reviser(proposal_path, critique_path, cycle)

    @staticmethod
    def _write_state(path: Path, result: AutonomousProtocolLoopResult) -> None:
        state = result.to_dict()
        state["artifacts"] = {
            "proposals": [
                {"path": item, "sha256": sha256_path(Path(item))}
                for item in result.proposal_paths
            ],
            "critiques": [
                {"path": item, "sha256": sha256_path(Path(item))}
                for item in result.critique_paths
            ],
        }
        if result.frozen_protocol_path:
            state["artifacts"]["frozen_protocol"] = {
                "path": result.frozen_protocol_path,
                "sha256": sha256_path(Path(result.frozen_protocol_path)),
            }
        atomic_json(path, json.loads(json.dumps(state)), refuse_overwrite=True)
