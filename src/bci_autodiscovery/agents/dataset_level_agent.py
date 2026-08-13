"""Top-level Dataset Agent coordinating profiling, discovery, critique, and freeze."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from bci_autodiscovery.literature import CrossrefSource, LiteratureStore, OpenAlexSource
from bci_autodiscovery.profiling import validate_dataset_profile_provenance
from bci_autodiscovery.search import (
    build_combined_search_space_review,
    build_search_space_draft,
)
from bci_autodiscovery.workflow.autonomy import load_json_object, sha256_path
from bci_autodiscovery.workflow.dataset_intelligence import (
    DatasetIntelligenceWorkflowError,
    freeze_dataset_level_contract,
)
from bci_autodiscovery.workflow.protocol_artifacts import atomic_json

from .audit import AuditSink, NullAuditSink
from .dataset_intelligence_critic import (
    DatasetIntelligenceCriticAgent,
    create_dataset_intelligence_critic_tools,
)
from .dataset_profiler import DatasetProfilerAgent, create_dataset_profiler_tools
from .literature_scout import LiteratureScoutAgent, create_literature_scout_tools
from .providers import ModelProvider
from .runtime import AgentRuntime, RuntimeLimits


class DatasetLevelAgentError(RuntimeError):
    pass


ProviderFactory = Callable[[int], ModelProvider]
SourceFactory = Callable[[int], tuple[Any, ...]]


def _normalized_frontier_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalized_frontier_value(item)
            for key, item in sorted(value.items())
            if key not in {"candidate_id", "status"}
        }
    if isinstance(value, list):
        normalized = [_normalized_frontier_value(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        )
    return value


def _frontier_semantic_hash(review: dict[str, Any]) -> str:
    directions = (review.get("frontier_space") or {}).get("directions") or []
    payload = _normalized_frontier_value(directions)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class DatasetLevelAgentResult:
    run_id: str
    status: str = "in_progress"
    cycles: int = 0
    artifacts: dict[str, Any] = field(default_factory=dict)
    phase_results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetLevelAgent:
    """One dataset-level Agent composed from bounded specialist runtimes.

    The public output is a frozen ``DatasetLevelContract``. This coordinator deliberately
    stops before research protocol design, subject measurements, pipeline execution, and
    confirmation access.
    """

    def __init__(
        self,
        *,
        run_id: str,
        run_dir: Path,
        component_registry_path: Path,
        literature_provider_factory: ProviderFactory,
        critic_provider_factory: ProviderFactory,
        profiler_provider: ModelProvider | None = None,
        scholarly_source_factory: SourceFactory | None = None,
        audit: AuditSink | None = None,
        max_revision_cycles: int = 1,
    ) -> None:
        if max_revision_cycles < 0:
            raise ValueError("max_revision_cycles must be non-negative")
        self.run_id = run_id
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.component_registry_path = (
            Path(component_registry_path).expanduser().resolve()
        )
        self.literature_provider_factory = literature_provider_factory
        self.critic_provider_factory = critic_provider_factory
        self.profiler_provider = profiler_provider
        self.scholarly_source_factory = scholarly_source_factory or (
            lambda _cycle: (CrossrefSource(), OpenAlexSource())
        )
        self.audit = audit or NullAuditSink()
        self.max_revision_cycles = max_revision_cycles

    def run(
        self,
        *,
        dataset_id: str,
        dataset_root: Path,
        validation_path: Path,
    ) -> DatasetLevelAgentResult:
        """Run local structure inspection before the remaining dataset-level loop."""

        self._prepare_run_dir()
        result = DatasetLevelAgentResult(run_id=self.run_id)
        if self.profiler_provider is None:
            return self._fail(result, "Dataset profiler provider is not configured")
        try:
            tools = create_dataset_profiler_tools()
            runtime = AgentRuntime(
                provider=self.profiler_provider,
                tools=tools,
                audit=self.audit,
                limits=RuntimeLimits(max_iterations=8, max_tool_calls=8),
                run_id=f"{self.run_id}:dataset-profiler",
            )
            phase = DatasetProfilerAgent(runtime).run(
                dataset_id=dataset_id,
                dataset_root=dataset_root,
                validation_path=validation_path,
            )
            self._record_phase(result, "dataset_profiler", phase.to_dict())
            inspection = phase.latest_tool_result("inspect_dataset")
            profile = phase.latest_tool_result("profile_dataset")
            if inspection is not None:
                inspection_path = self.run_dir / "dataset_inspection.json"
                atomic_json(inspection_path, inspection, refuse_overwrite=True)
                self._record_artifact(result, "dataset_inspection", inspection_path)
            if phase.status != "completed" or inspection is None:
                return self._fail(
                    result,
                    phase.error or "Dataset inspection/profiling did not complete",
                )
            if profile is None:
                return self._fail(
                    result,
                    "Dataset inspection stopped before profiling: "
                    f"{inspection.get('status', 'unknown_status')}",
                )
            profile_path = self.run_dir / "dataset_profile.json"
            atomic_json(profile_path, profile, refuse_overwrite=True)
            self._record_artifact(result, "dataset_profile", profile_path)
            return self._run_from_owned_profile(result, profile_path)
        except Exception as exc:
            return self._fail(result, f"{type(exc).__name__}: {exc}")

    def run_from_profile(self, *, dataset_profile_path: Path) -> DatasetLevelAgentResult:
        """Start from an existing standard profile while preserving an immutable local copy."""

        self._prepare_run_dir()
        result = DatasetLevelAgentResult(run_id=self.run_id)
        try:
            source_profile = load_json_object(
                Path(dataset_profile_path).expanduser().resolve()
            )
            validate_dataset_profile_provenance(
                source_profile,
                require_hashed_evidence=True,
                require_current_constraints=True,
            )
            profile_path = self.run_dir / "dataset_profile.json"
            atomic_json(profile_path, source_profile, refuse_overwrite=True)
            self._record_artifact(result, "dataset_profile", profile_path)
            return self._run_from_owned_profile(result, profile_path)
        except Exception as exc:
            return self._fail(result, f"{type(exc).__name__}: {exc}")

    def _run_from_owned_profile(
        self,
        result: DatasetLevelAgentResult,
        profile_path: Path,
    ) -> DatasetLevelAgentResult:
        self.audit.record(
            "deterministic_stage_started",
            {
                "stage": "canonical_search_space",
                "dataset_profile_path": str(profile_path),
                "component_registry_path": str(self.component_registry_path),
            },
        )
        canonical = build_search_space_draft(
            dataset_profile_path=str(profile_path),
            component_registry_path=str(self.component_registry_path),
        )
        canonical_path = self.run_dir / "canonical_search_space.json"
        atomic_json(canonical_path, canonical, refuse_overwrite=True)
        self._record_artifact(result, "canonical_search_space", canonical_path)
        self.audit.record(
            "deterministic_stage_completed",
            {
                "stage": "canonical_search_space",
                "artifact": result.artifacts["canonical_search_space"],
            },
        )

        previous_review_hash: str | None = None
        previous_frontier_hash: str | None = None
        previous_critique_path: Path | None = None
        previous_evidence_path: Path | None = None
        previous_literature_run_id: str | None = None
        for cycle in range(1, self.max_revision_cycles + 2):
            result.cycles = cycle
            cycle_dir = self.run_dir / "cycles" / f"{cycle:04d}"
            cycle_dir.mkdir(parents=True, exist_ok=False)
            evidence_path = cycle_dir / "evidence.sqlite"
            literature_run_id = f"{self.run_id}:literature:{cycle:04d}"
            if previous_evidence_path is not None:
                if previous_literature_run_id is None:
                    raise DatasetLevelAgentError(
                        "Revision evidence source lacks a literature run ID"
                    )
                shutil.copy2(previous_evidence_path, evidence_path)
                clone_result = LiteratureStore(evidence_path).clone_search_evidence(
                    source_search_run_id=previous_literature_run_id,
                    target_search_run_id=literature_run_id,
                )
                self.audit.record(
                    "literature_evidence_reused",
                    {
                        "cycle": cycle,
                        "source_evidence_path": str(previous_evidence_path),
                        "source_literature_run_id": previous_literature_run_id,
                        "target_evidence_path": str(evidence_path),
                        "target_literature_run_id": literature_run_id,
                        **clone_result,
                    },
                )
            literature_tools, literature_context = create_literature_scout_tools(
                search_space_path=canonical_path,
                evidence_db_path=evidence_path,
                search_run_id=literature_run_id,
                sources=self.scholarly_source_factory(cycle),
                revision_critique_path=previous_critique_path,
            )
            literature_runtime = AgentRuntime(
                provider=self.literature_provider_factory(cycle),
                tools=literature_tools,
                audit=self.audit,
                limits=RuntimeLimits(max_iterations=32, max_tool_calls=32),
                run_id=f"{self.run_id}:literature-scout:{cycle:04d}",
            )
            literature_result = LiteratureScoutAgent(
                runtime=literature_runtime,
                context=literature_context,
            ).run()
            self._record_phase(
                result,
                f"literature_scout_{cycle:04d}",
                literature_result.to_dict(),
            )
            literature_result_path = cycle_dir / "literature_agent_result.json"
            atomic_json(
                literature_result_path,
                literature_result.to_dict(),
                refuse_overwrite=True,
            )
            frontier_status = literature_tools.execute(
                "inspect_frontier_discovery_status", {}
            )
            frontier_status_path = cycle_dir / "frontier_discovery.json"
            atomic_json(frontier_status_path, frontier_status, refuse_overwrite=True)
            self._record_artifact(
                result, f"frontier_discovery_{cycle:04d}", frontier_status_path
            )
            self._record_artifact(
                result, f"evidence_db_{cycle:04d}", evidence_path
            )
            if literature_result.status != "completed" or not frontier_status["complete"]:
                return self._fail(
                    result,
                    literature_result.error
                    or "Literature Scout did not complete all planned searches and directions",
                )

            self.audit.record(
                "deterministic_stage_started",
                {
                    "stage": "dataset_level_merge",
                    "canonical_search_space_path": str(canonical_path),
                    "evidence_db_path": str(evidence_path),
                    "literature_run_id": literature_run_id,
                },
            )
            review = build_combined_search_space_review(
                canonical_search_space_path=canonical_path,
                evidence_db_path=evidence_path,
                literature_run_id=literature_run_id,
            )
            review_path = cycle_dir / "dataset_level_draft.json"
            atomic_json(review_path, review, refuse_overwrite=True)
            review_hash = sha256_path(review_path)
            frontier_hash = _frontier_semantic_hash(review)
            if previous_review_hash is not None and review_hash == previous_review_hash:
                return self._fail(
                    result,
                    "Dataset revision reproduced the criticized draft without change",
                )
            if (
                previous_frontier_hash is not None
                and frontier_hash == previous_frontier_hash
            ):
                return self._fail(
                    result,
                    "Literature revision did not change the criticized frontier semantics",
                )
            previous_review_hash = review_hash
            previous_frontier_hash = frontier_hash
            self._record_artifact(
                result, f"dataset_level_draft_{cycle:04d}", review_path
            )
            self.audit.record(
                "deterministic_stage_completed",
                {
                    "stage": "dataset_level_merge",
                    "artifact": result.artifacts[f"dataset_level_draft_{cycle:04d}"],
                },
            )

            critic_tools, critic_context = create_dataset_intelligence_critic_tools(
                dataset_level_draft_path=review_path
            )
            critic_runtime = AgentRuntime(
                provider=self.critic_provider_factory(cycle),
                tools=critic_tools,
                audit=self.audit,
                limits=RuntimeLimits(max_iterations=6, max_tool_calls=4),
                run_id=f"{self.run_id}:dataset-critic:{cycle:04d}",
            )
            critic_result = DatasetIntelligenceCriticAgent(
                runtime=critic_runtime,
                context=critic_context,
            ).run()
            self._record_phase(
                result,
                f"dataset_critic_{cycle:04d}",
                critic_result.to_dict(),
            )
            critique = critic_result.latest_tool_result("record_dataset_critique")
            if critic_result.status != "completed" or critique is None:
                return self._fail(
                    result,
                    critic_result.error or "Dataset Critic produced no verdict",
                )
            critique_path = cycle_dir / "dataset_critique.json"
            atomic_json(critique_path, critique, refuse_overwrite=True)
            self._record_artifact(
                result, f"dataset_critique_{cycle:04d}", critique_path
            )
            verdict = critique["verdict"]
            if verdict == "pass":
                frozen_path = self.run_dir / "dataset_level_contract.json"
                self.audit.record(
                    "deterministic_stage_started",
                    {
                        "stage": "dataset_level_freeze",
                        "dataset_level_draft_path": str(review_path),
                        "dataset_critique_path": str(critique_path),
                    },
                )
                freeze_dataset_level_contract(
                    dataset_level_draft_path=review_path,
                    dataset_critique_path=critique_path,
                    output_path=frozen_path,
                )
                self._record_artifact(result, "dataset_level_contract", frozen_path)
                self.audit.record(
                    "deterministic_stage_completed",
                    {
                        "stage": "dataset_level_freeze",
                        "artifact": result.artifacts["dataset_level_contract"],
                    },
                )
                result.status = "completed"
                self._finish(result)
                return result
            if verdict == "reject":
                result.status = "rejected"
                result.error = critique["rationale"]
                self._finish(result)
                return result
            previous_critique_path = critique_path
            previous_evidence_path = evidence_path
            previous_literature_run_id = literature_run_id

        result.status = "revision_limit_reached"
        result.error = "Maximum Dataset Agent revision cycles reached"
        self._finish(result)
        return result

    def _prepare_run_dir(self) -> None:
        existing = (
            [item for item in self.run_dir.iterdir() if item.name != "run_process.json"]
            if self.run_dir.exists()
            else []
        )
        if existing:
            raise DatasetLevelAgentError(
                f"Refusing to append to an existing Dataset Agent run: {self.run_dir}"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _record_phase(
        self,
        result: DatasetLevelAgentResult,
        phase: str,
        phase_result: dict[str, Any],
    ) -> None:
        result.phase_results.append({"phase": phase, "result": phase_result})

    def _record_artifact(
        self,
        result: DatasetLevelAgentResult,
        name: str,
        path: Path,
    ) -> None:
        resolved = Path(path).expanduser().resolve()
        result.artifacts[name] = {
            "path": str(resolved),
            "sha256": sha256_path(resolved),
        }
        self.audit.record(
            "artifact_recorded",
            {"name": name, **result.artifacts[name]},
        )

    def _fail(
        self,
        result: DatasetLevelAgentResult,
        error: str,
    ) -> DatasetLevelAgentResult:
        result.status = "failed"
        result.error = error
        self._finish(result)
        return result

    def _finish(self, result: DatasetLevelAgentResult) -> None:
        self.audit.record(
            "dataset_level_run_finished",
            {
                "status": result.status,
                "cycles": result.cycles,
                "error": result.error,
                "artifacts": result.artifacts,
            },
        )
        manifest = self.run_dir / "dataset_level_run.json"
        if manifest.exists():
            raise DatasetIntelligenceWorkflowError(
                "Dataset Agent run manifest already exists"
            )
        atomic_json(manifest, result.to_dict(), refuse_overwrite=True)
