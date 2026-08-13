"""Production orchestration for Planner -> independent Critic -> deterministic freeze."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from bci_autodiscovery.workflow.autonomous_protocol import (
    freeze_autonomous_protocol,
    validate_frozen_autonomous_protocol,
)
from bci_autodiscovery.workflow.autonomy import (
    load_autonomy_envelope,
    load_json_object,
    sha256_path,
)
from bci_autodiscovery.workflow.budget import BudgetLedger
from bci_autodiscovery.workflow.dataset_contract import load_dataset_level_contract
from bci_autodiscovery.workflow.protocol_artifacts import atomic_json

from .audit import AuditSink, NullAuditSink
from .budgeted_provider import BudgetedProvider, TokenPricing
from .protocol_critic import (
    ProtocolCriticAgent,
    create_protocol_critic_tools,
    validate_protocol_critique,
)
from .providers import ModelProvider
from .research_protocol import (
    ResearchProtocolPlannerAgent,
    ResearchProtocolRevisionAgent,
    create_research_protocol_planner_tools,
    create_research_protocol_revision_tools,
    _load_authorized_research_context,
    validate_research_protocol_authority_bindings,
    validate_research_protocol_proposal,
)
from .runtime import AgentRuntime, RuntimeLimits


ProviderFactory = Callable[[str, int], ModelProvider]


class ResearchDesignAgentError(RuntimeError):
    pass


@dataclass
class ResearchDesignAgentResult:
    run_id: str
    status: str
    cycles: int = 0
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    phase_results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    recoverable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResearchDesignAgent:
    """Outcome-blind Research Design stage with resumable immutable checkpoints."""

    def __init__(
        self,
        *,
        run_id: str,
        run_dir: Path,
        dataset_level_contract_path: Path,
        autonomy_envelope_path: Path,
        provider_factory: ProviderFactory,
        budget_ledger: BudgetLedger,
        pricing: TokenPricing | None,
        audit: AuditSink | None = None,
        audit_path: Path | None = None,
        max_revision_cycles: int = 2,
    ) -> None:
        if not 0 <= max_revision_cycles <= 8:
            raise ValueError("max_revision_cycles must be between 0 and 8")
        self.run_id = run_id
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.contract_path = Path(dataset_level_contract_path).expanduser().resolve()
        self.envelope_path = Path(autonomy_envelope_path).expanduser().resolve()
        self.provider_factory = provider_factory
        self.ledger = budget_ledger
        self.pricing = pricing
        self.audit = audit or NullAuditSink()
        self.audit_path = str(Path(audit_path).resolve()) if audit_path else None
        self.max_revision_cycles = max_revision_cycles
        self.state_path = self.run_dir / "research_design_state.json"
        self.manifest_path = self.run_dir / "research_design_run.json"
        self._provider_instances: list[ModelProvider] = []

    def run(self, *, resume: bool = False) -> ResearchDesignAgentResult:
        contract = load_dataset_level_contract(self.contract_path)
        envelope = load_autonomy_envelope(
            self.envelope_path,
            expected_dataset_id=str(contract["dataset_id"]),
            expected_dataset_contract_path=self.contract_path,
        )
        state = self._load_or_initialize_state(
            resume=resume,
            dataset_id=str(contract["dataset_id"]),
            envelope=envelope,
        )
        result = self._result_from_state(state)
        self.audit.record(
            "research_design_run_started" if not resume else "research_design_run_resumed",
            {
                "run_id": self.run_id,
                "dataset_id": contract["dataset_id"],
                "contract_sha256": sha256_path(self.contract_path),
                "autonomy_envelope_sha256": sha256_path(self.envelope_path),
                "budget": self.ledger.snapshot(),
            },
        )
        if resume:
            self.ledger.record_recovery(source_run_id=self.run_id)

        while True:
            cycles: list[dict[str, Any]] = state["cycles"]
            if not cycles or cycles[-1].get("critique") is not None:
                if cycles and cycles[-1]["critique"]["verdict"] != "revise":
                    raise ResearchDesignAgentError("Cannot extend a non-revise protocol cycle")
                if len(cycles) > self.max_revision_cycles:
                    return self._finish_terminal(
                        state,
                        result,
                        status="revision_limit_reached",
                        error="Maximum autonomous protocol revision cycles reached",
                    )
                cycle_number = len(cycles) + 1
                proposal = self._run_proposal_stage(
                    cycle=cycle_number,
                    previous=cycles[-1] if cycles else None,
                    result=result,
                    state=state,
                )
                if proposal is None:
                    return result
                cycles.append(
                    {
                        "cycle": cycle_number,
                        "proposal": self._artifact_record(
                            self.run_dir / f"proposal-{cycle_number:04d}.json"
                        ),
                        "critique": None,
                    }
                )
                result.cycles = cycle_number
                self._checkpoint(state, result, stage="proposal_recorded")

            current = cycles[-1]
            proposal_path = Path(current["proposal"]["path"])
            if current.get("critique") is None:
                critique = self._run_critic_stage(
                    proposal_path=proposal_path,
                    cycle=int(current["cycle"]),
                    result=result,
                    state=state,
                )
                if critique is None:
                    return result
                critique_path = self.run_dir / f"critique-{int(current['cycle']):04d}.json"
                current["critique"] = {
                    **self._artifact_record(critique_path),
                    "verdict": critique["verdict"],
                }
                self._checkpoint(state, result, stage="critique_recorded")

            verdict = current["critique"]["verdict"]
            if verdict == "reject":
                critique = load_json_object(Path(current["critique"]["path"]))
                return self._finish_terminal(
                    state,
                    result,
                    status="rejected",
                    error=str(critique.get("rationale") or "Protocol Critic rejected design"),
                )
            if verdict == "revise":
                continue
            if verdict != "pass":
                raise ResearchDesignAgentError(f"Unknown protocol verdict: {verdict}")

            frozen_path = self.run_dir / "frozen_protocol.json"
            if not frozen_path.exists():
                self.audit.record(
                    "research_design_stage_started",
                    {"stage": "deterministic_freeze", "cycle": current["cycle"]},
                )
                freeze_autonomous_protocol(
                    proposal_path=proposal_path,
                    critique_path=Path(current["critique"]["path"]),
                    dataset_level_contract_path=self.contract_path,
                    autonomy_envelope_path=self.envelope_path,
                    expected_dataset_level_contract_sha256=sha256_path(self.contract_path),
                    expected_autonomy_envelope_sha256=sha256_path(self.envelope_path),
                    expected_proposal_sha256=current["proposal"]["sha256"],
                    expected_critique_sha256=current["critique"]["sha256"],
                    output_path=frozen_path,
                )
            else:
                validate_frozen_autonomous_protocol(
                    frozen_protocol_path=frozen_path,
                    proposal_path=proposal_path,
                    critique_path=Path(current["critique"]["path"]),
                    dataset_level_contract_path=self.contract_path,
                    autonomy_envelope_path=self.envelope_path,
                )
            result.artifacts["frozen_protocol"] = self._artifact_record(frozen_path)
            return self._finish_terminal(state, result, status="completed", error=None)

    def _run_proposal_stage(
        self,
        *,
        cycle: int,
        previous: dict[str, Any] | None,
        result: ResearchDesignAgentResult,
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        stage = "planner" if previous is None else "reviser"
        self.audit.record(
            "research_design_stage_started", {"stage": stage, "cycle": cycle}
        )
        if previous is None:
            tools, context = create_research_protocol_planner_tools(
                dataset_level_contract_path=self.contract_path,
                autonomy_envelope_path=self.envelope_path,
            )
            runtime = self._runtime(stage=stage, cycle=cycle, tools=tools)
            agent_result = ResearchProtocolPlannerAgent(
                runtime=runtime,
                context=context,
            ).run()
            proposal = agent_result.latest_tool_result("record_research_protocol_proposal")
        else:
            tools, context = create_research_protocol_revision_tools(
                dataset_level_contract_path=self.contract_path,
                autonomy_envelope_path=self.envelope_path,
                proposal_path=Path(previous["proposal"]["path"]),
                critique_path=Path(previous["critique"]["path"]),
            )
            runtime = self._runtime(stage=stage, cycle=cycle, tools=tools)
            agent_result = ResearchProtocolRevisionAgent(
                runtime=runtime,
                context=context,
            ).run()
            proposal = agent_result.latest_tool_result("record_revised_research_protocol")
        self._record_phase(result, f"{stage}-{cycle:04d}", agent_result.to_dict())
        if agent_result.status != "completed" or proposal is None:
            self._fail_recoverable(
                state,
                result,
                stage=stage,
                error=agent_result.error or f"{stage} produced no valid protocol",
            )
            return None
        proposal_path = self.run_dir / f"proposal-{cycle:04d}.json"
        atomic_json(proposal_path, proposal, refuse_overwrite=True)
        result.artifacts[f"proposal_{cycle:04d}"] = self._artifact_record(proposal_path)
        self.audit.record(
            "artifact_recorded",
            {"name": f"proposal_{cycle:04d}", **result.artifacts[f"proposal_{cycle:04d}"]},
        )
        return proposal

    def _run_critic_stage(
        self,
        *,
        proposal_path: Path,
        cycle: int,
        result: ResearchDesignAgentResult,
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        stage = "protocol_critic"
        self.audit.record(
            "research_design_stage_started", {"stage": stage, "cycle": cycle}
        )
        tools, context = create_protocol_critic_tools(
            dataset_level_contract_path=self.contract_path,
            autonomy_envelope_path=self.envelope_path,
            proposal_path=proposal_path,
        )
        runtime = self._runtime(stage=stage, cycle=cycle, tools=tools)
        agent_result = ProtocolCriticAgent(runtime=runtime, context=context).run()
        critique = agent_result.latest_tool_result("record_protocol_critique")
        self._record_phase(result, f"{stage}-{cycle:04d}", agent_result.to_dict())
        if agent_result.status != "completed" or critique is None:
            self._fail_recoverable(
                state,
                result,
                stage=stage,
                error=agent_result.error or "Protocol Critic produced no valid verdict",
            )
            return None
        critique_path = self.run_dir / f"critique-{cycle:04d}.json"
        atomic_json(critique_path, critique, refuse_overwrite=True)
        result.artifacts[f"critique_{cycle:04d}"] = self._artifact_record(critique_path)
        self.audit.record(
            "artifact_recorded",
            {"name": f"critique_{cycle:04d}", **result.artifacts[f"critique_{cycle:04d}"]},
        )
        return critique

    def _runtime(self, *, stage: str, cycle: int, tools: Any) -> AgentRuntime:
        provider = self.provider_factory(stage, cycle)
        if any(provider is previous for previous in self._provider_instances):
            raise ResearchDesignAgentError(
                "Provider factory reused an instance across independent Agent turns"
            )
        self._provider_instances.append(provider)
        budgeted = BudgetedProvider(
            provider=provider,
            ledger=self.ledger,
            pricing=self.pricing,
            stage=f"{stage}:{cycle:04d}",
        )
        return AgentRuntime(
            provider=budgeted,
            tools=tools,
            audit=self.audit,
            limits=RuntimeLimits(max_iterations=8, max_tool_calls=6),
            run_id=f"{self.run_id}:{stage}:{cycle:04d}",
            audit_path=self.audit_path,
        )

    def _load_or_initialize_state(
        self,
        *,
        resume: bool,
        dataset_id: str,
        envelope: dict[str, Any],
    ) -> dict[str, Any]:
        if resume:
            if not self.state_path.is_file():
                raise ResearchDesignAgentError("Cannot resume without research_design_state.json")
            state = load_json_object(self.state_path)
            if state.get("run_id") != self.run_id:
                raise ResearchDesignAgentError("Research Design state belongs to another run")
            if state.get("status") not in {"in_progress", "failed_recoverable"}:
                raise ResearchDesignAgentError("Research Design state is not recoverable")
            self._validate_state_artifacts(state)
            expected_authorities = {
                "dataset_level_contract_sha256": sha256_path(self.contract_path),
                "autonomy_envelope_sha256": sha256_path(self.envelope_path),
            }
            if any(state.get(field) != value for field, value in expected_authorities.items()):
                raise ResearchDesignAgentError("Research Design authorities changed before recovery")
            self._reconcile_orphan_checkpoints(state)
            state["status"] = "in_progress"
            state["error"] = None
            return state

        allowed = {"audit.jsonl", "budget_ledger.jsonl", "run_process.json"}
        existing = list(self.run_dir.iterdir()) if self.run_dir.exists() else []
        unexpected = [item for item in existing if item.name not in allowed]
        if unexpected:
            raise ResearchDesignAgentError(
                f"Refusing to append to existing Research Design run: {self.run_dir}"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": "2.0",
            "run_id": self.run_id,
            "dataset_id": dataset_id,
            "status": "in_progress",
            "stage": "initialized",
            "error": None,
            "dataset_level_contract_sha256": sha256_path(self.contract_path),
            "autonomy_envelope_sha256": sha256_path(self.envelope_path),
            "envelope_id": envelope["envelope_id"],
            "cycles": [],
            "artifacts": {},
            "phase_results": [],
            "budget": self.ledger.snapshot(),
        }
        atomic_json(self.state_path, state, refuse_overwrite=True)
        return state

    def _validate_state_artifacts(self, state: dict[str, Any]) -> None:
        for cycle in state.get("cycles") or []:
            for field in ("proposal", "critique"):
                record = cycle.get(field)
                if not record:
                    continue
                path = Path(str(record.get("path") or "")).expanduser().resolve()
                if path.parent != self.run_dir:
                    raise ResearchDesignAgentError("Recovery artifact escaped the run directory")
                if not path.is_file() or sha256_path(path) != record.get("sha256"):
                    raise ResearchDesignAgentError(
                        f"Recovery artifact integrity check failed: {path}"
                    )

    def _reconcile_orphan_checkpoints(self, state: dict[str, Any]) -> None:
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
        changed = False
        while True:
            cycles: list[dict[str, Any]] = state["cycles"]
            if cycles and cycles[-1].get("critique") is None:
                cycle = int(cycles[-1]["cycle"])
                critique_path = self.run_dir / f"critique-{cycle:04d}.json"
                if not critique_path.exists():
                    break
                proposal_path = Path(cycles[-1]["proposal"]["path"])
                proposal = load_json_object(proposal_path)
                critique = load_json_object(critique_path)
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
                validate_protocol_critique(
                    critique,
                    proposal=proposal,
                    proposal_sha256=sha256_path(proposal_path),
                    deterministic_validation_passed=True,
                )
                cycles[-1]["critique"] = {
                    **self._artifact_record(critique_path),
                    "verdict": critique["verdict"],
                }
                changed = True
                continue
            if cycles and cycles[-1]["critique"]["verdict"] != "revise":
                break
            cycle = len(cycles) + 1
            proposal_path = self.run_dir / f"proposal-{cycle:04d}.json"
            if not proposal_path.exists():
                break
            proposal = load_json_object(proposal_path)
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
            cycles.append(
                {
                    "cycle": cycle,
                    "proposal": self._artifact_record(proposal_path),
                    "critique": None,
                }
            )
            changed = True
        if changed:
            state["stage"] = "recovered_orphan_checkpoints"
            atomic_json(self.state_path, state)

    def _result_from_state(self, state: dict[str, Any]) -> ResearchDesignAgentResult:
        return ResearchDesignAgentResult(
            run_id=self.run_id,
            status="in_progress",
            cycles=len(state.get("cycles") or []),
            artifacts=json.loads(json.dumps(state.get("artifacts") or {})),
            phase_results=json.loads(json.dumps(state.get("phase_results") or [])),
        )

    def _record_phase(
        self, result: ResearchDesignAgentResult, phase: str, phase_result: dict[str, Any]
    ) -> None:
        result.phase_results.append({"phase": phase, "result": phase_result})

    @staticmethod
    def _artifact_record(path: Path) -> dict[str, Any]:
        resolved = Path(path).expanduser().resolve()
        return {"path": str(resolved), "sha256": sha256_path(resolved)}

    def _checkpoint(
        self, state: dict[str, Any], result: ResearchDesignAgentResult, *, stage: str
    ) -> None:
        state.update(
            {
                "status": result.status,
                "stage": stage,
                "error": result.error,
                "artifacts": result.artifacts,
                "phase_results": result.phase_results,
                "budget": self.ledger.snapshot(),
            }
        )
        atomic_json(self.state_path, state)
        self.audit.record(
            "research_design_checkpoint",
            {"stage": stage, "cycles": len(state["cycles"]), "budget": state["budget"]},
        )

    def _fail_recoverable(
        self,
        state: dict[str, Any],
        result: ResearchDesignAgentResult,
        *,
        stage: str,
        error: str,
    ) -> None:
        result.status = "failed_recoverable"
        result.error = error
        result.recoverable = True
        self._checkpoint(state, result, stage=stage)
        self.audit.record(
            "research_design_run_failed",
            {"stage": stage, "error": error, "recoverable": True},
        )

    def _finish_terminal(
        self,
        state: dict[str, Any],
        result: ResearchDesignAgentResult,
        *,
        status: str,
        error: str | None,
    ) -> ResearchDesignAgentResult:
        result.status = status
        result.error = error
        result.recoverable = False
        result.cycles = len(state["cycles"])
        state["status"] = status
        state["error"] = error
        state["artifacts"] = result.artifacts
        state["phase_results"] = result.phase_results
        state["stage"] = "terminal"
        state["budget"] = self.ledger.close(status)
        atomic_json(self.state_path, state)
        atomic_json(self.manifest_path, result.to_dict(), refuse_overwrite=True)
        self.audit.record(
            "research_design_run_finished",
            {
                "status": status,
                "cycles": result.cycles,
                "error": error,
                "artifacts": result.artifacts,
                "budget": state["budget"],
            },
        )
        return result
