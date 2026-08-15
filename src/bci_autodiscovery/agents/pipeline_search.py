"""Autonomous budgeted pipeline search and evidence-bound lock agent."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci_autodiscovery.literature import (
    CrossrefSource,
    LiteratureQuery,
    LiteratureStore,
    OpenAlexSource,
)
from bci_autodiscovery.pipelines import (
    DeterministicPipelineExecutor,
    PipelineSpec,
    pipeline_configuration_hash,
)
from bci_autodiscovery.pipelines.executor import pipeline_spec_schema
from bci_autodiscovery.search.dataset_incumbent import (
    DatasetIncumbentError,
    expected_selective_route,
    validate_dataset_incumbent,
)
from bci_autodiscovery.workflow.autonomy import (
    load_autonomy_envelope,
    load_json_object,
    sha256_path,
)
from bci_autodiscovery.workflow.budget import BudgetLedger

from .contracts import AgentRunResult
from .language import DEFAULT_LANGUAGE_INSTRUCTION
from .runtime import AgentRuntime
from .tools import ToolDefinition, ToolRegistry


PIPELINE_SEARCH_SYSTEM_PROMPT = f"""{DEFAULT_LANGUAGE_INSTRUCTION}
You are the autonomous individualized Pipeline Search Agent. First call
read_pipeline_search_context. Use the frozen protocol, SubjectProfile, executable capability
registry, and remaining budget. You cannot access frozen-confirmation data.

When a frozen dataset-wide incumbent is present, evaluate that exact configuration as the
mandatory first control. You may search profile-linked alternatives, but the frozen selective
personalization gate decides whether the final route is personalized or falls back to the
dataset incumbent. Do not override that gate.

When literature discovery is authorized, first formulate a phenotype-linked method query and
call search_subject_method_evidence. Treat returned metadata/abstracts as directional evidence,
not proof that a method will work for this subject. Cite only returned stable paper IDs.

Propose one complete declarative pipeline at a time, call evaluate_pipeline_candidate, inspect
the deterministic cross-validation evidence, diagnose the result, then choose the next most
informative candidate. Do not enumerate the full Cartesian product. Vary a component only
when it tests a profile-linked hypothesis, a plausible alternative, or a necessary baseline.

When a frozen stopping condition is met or the budget should stop, call lock_pipeline with
the selected experiment, all evidence used, alternatives, uncertainty, and stop reason. The
selected candidate need not be the numerically highest score only when the evidence-backed
rationale justifies the tradeoff. Do not ask a human to select the pipeline."""


class PipelineSearchError(ValueError):
    pass


_FAMILY_COMPONENT_REQUIREMENTS = {
    "bandpower_lda": frozenset(
        {"reference_original", "filter_bandpass", "feature_log_bandpower", "model_lda"}
    ),
    "csp_lda": frozenset(
        {"reference_original", "filter_bandpass", "feature_csp", "model_lda"}
    ),
}


def _constrain_capabilities_by_dataset_contract(
    *, protocol: dict[str, Any], capabilities: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Intersect executable families with the frozen DatasetLevelContract."""

    binding = protocol.get("dataset_level_contract")
    if not isinstance(binding, dict):
        return capabilities, None
    contract_path = Path(str(binding.get("path") or "")).expanduser().resolve()
    expected_sha256 = str(binding.get("sha256") or "")
    if not contract_path.is_file() or sha256_path(contract_path) != expected_sha256:
        raise PipelineSearchError("Frozen protocol DatasetLevelContract binding changed")
    contract = load_json_object(contract_path)
    if contract.get("status") != "frozen_dataset_level_contract":
        raise PipelineSearchError("Pipeline search requires a frozen DatasetLevelContract")
    if contract.get("dataset_id") != protocol.get("dataset_id"):
        raise PipelineSearchError("Frozen protocol and DatasetLevelContract disagree")

    canonical_ids = {
        str(item.get("component_id"))
        for items in (contract.get("canonical_space") or {}).get("dimensions", {}).values()
        for item in items
        if isinstance(item, dict) and item.get("component_id")
    }
    excluded_ids = {
        str(item.get("component_id"))
        for item in contract.get("excluded_components") or []
        if isinstance(item, dict) and item.get("component_id")
    }
    allowed_families: list[str] = []
    excluded_families: dict[str, list[str]] = {}
    for family in capabilities["families"]:
        requirements = _FAMILY_COMPONENT_REQUIREMENTS.get(str(family))
        if requirements is None:
            excluded_families[str(family)] = ["family_has_no_contract_component_mapping"]
            continue
        missing = sorted(requirements.difference(canonical_ids))
        explicitly_excluded = sorted(requirements.intersection(excluded_ids))
        reasons = [
            *[f"not_in_canonical_space:{item}" for item in missing],
            *[f"dataset_contract_excluded:{item}" for item in explicitly_excluded],
        ]
        if reasons:
            excluded_families[str(family)] = reasons
        else:
            allowed_families.append(str(family))
    if not allowed_families:
        raise PipelineSearchError(
            "No executable pipeline family survives the frozen DatasetLevelContract"
        )
    constrained = deepcopy(capabilities)
    constrained["families"] = allowed_families
    return constrained, {
        "path": str(contract_path),
        "sha256": expected_sha256,
        "contract_id": contract.get("contract_id"),
        "allowed_executable_families": allowed_families,
        "excluded_executable_families": excluded_families,
    }


def _load_capabilities(path: Path) -> dict[str, Any]:
    value = load_json_object(path)
    if value.get("schema_version") != "1.0":
        raise PipelineSearchError("Unsupported executable capability schema_version")
    if value.get("status") != "executable_capabilities_not_search_activation":
        raise PipelineSearchError("Capability registry has an unsafe activation status")
    required_arrays = (
        "families",
        "bandpasses_hz",
        "csp_components",
        "lda_shrinkage",
        "cv_folds",
    )
    if any(not isinstance(value.get(field), list) or not value[field] for field in required_arrays):
        raise PipelineSearchError("Executable capability registry is incomplete")
    if value.get("channel_strategies") != ["all", "named"]:
        raise PipelineSearchError("Executable capability registry lacks safe channel strategies")
    policy = value.get("profile_conditioned_policy") or {}
    if not isinstance(policy.get("top_channel_set_sizes"), list):
        raise PipelineSearchError("Capability registry lacks profile-conditioned policy")
    return value


def _derive_profile_conditioned_capabilities(
    *,
    subject_profile: dict[str, Any],
    executor: DeterministicPipelineExecutor,
    capabilities: dict[str, Any],
) -> dict[str, Any]:
    policy = capabilities["profile_conditioned_policy"]
    lower_bound, upper_bound = (
        float(item) for item in policy["frequency_bounds_hz"]
    )
    half_width = float(policy["peak_half_width_hz"])
    minimum_width = float(policy["minimum_bandwidth_hz"])
    available_channels = tuple(executor.sessions[0].channel_names)
    bad_channels: set[str] = set()
    peak_bands: dict[tuple[float, float], set[str]] = {}
    ranked_channels: list[tuple[str, str]] = []

    for measurement in subject_profile.get("measurements") or []:
        measurement_id = str(measurement.get("measurement_id") or "")
        payload = measurement.get("payload") or {}
        kind = measurement.get("kind")
        if kind == "signal_quality":
            bad_channels.update(str(item) for item in payload.get("flat_channel_names") or [])
            bad_channels.update(
                str(item) for item in payload.get("robust_outlier_channel_names") or []
            )
        elif kind == "spectral_profile":
            for peak_name in ("mu_peak", "beta_peak"):
                value = (payload.get(peak_name) or {}).get("frequency_hz")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    low = max(lower_bound, round(float(value) - half_width, 3))
                    high = min(upper_bound, round(float(value) + half_width, 3))
                    if high - low >= minimum_width:
                        peak_bands.setdefault((low, high), set()).add(measurement_id)
        elif kind == "class_separability":
            groups = [payload.get("top_channels") or []]
            groups.extend(
                (item or {}).get("top_channels") or []
                for item in (payload.get("bandwise") or {}).values()
            )
            for group in groups:
                for item in group:
                    name = str((item or {}).get("channel_name") or "")
                    if name and name in available_channels and name not in bad_channels:
                        ranked_channels.append((name, measurement_id))

    ordered_channels: list[str] = []
    channel_evidence: dict[str, set[str]] = {}
    for name, measurement_id in ranked_channels:
        channel_evidence.setdefault(name, set()).add(measurement_id)
        if name not in ordered_channels:
            ordered_channels.append(name)
    maximum = min(int(policy["maximum_named_channels"]), len(available_channels))
    channel_sets: list[dict[str, Any]] = []
    seen_sets: set[tuple[str, ...]] = set()
    for requested_size in policy["top_channel_set_sizes"]:
        size = min(int(requested_size), maximum, len(ordered_channels))
        selected = tuple(ordered_channels[:size])
        if size < 1 or selected in seen_sets:
            continue
        seen_sets.add(selected)
        channel_sets.append(
            {
                "channel_names": list(selected),
                "evidence_measurement_ids": sorted(
                    {item for name in selected for item in channel_evidence.get(name, set())}
                ),
                "derivation": f"top_{size}_profile_ranked_nonflagged_channels",
            }
        )
    return {
        "fixed_bandpasses_hz": capabilities["bandpasses_hz"],
        "individualized_bandpasses": [
            {
                "bandpass_hz": list(band),
                "evidence_measurement_ids": sorted(evidence_ids),
                "derivation": "profile_peak_plus_minus_half_width",
            }
            for band, evidence_ids in sorted(peak_bands.items())
        ],
        "channel_options": [
            {
                "channel_strategy": "all",
                "selected_channels": [],
                "evidence_measurement_ids": [],
                "derivation": "dataset_signal_contract_baseline",
            },
            *[
                {
                    "channel_strategy": "named",
                    "selected_channels": item["channel_names"],
                    "evidence_measurement_ids": item["evidence_measurement_ids"],
                    "derivation": item["derivation"],
                }
                for item in channel_sets
            ],
        ],
        "excluded_profile_flagged_channels": sorted(
            bad_channels.intersection(available_channels)
        ),
        "policy": policy,
    }


def _validate_capability(
    spec: PipelineSpec,
    capabilities: dict[str, Any],
    profile_capabilities: dict[str, Any],
) -> None:
    if spec.family not in capabilities["families"]:
        raise PipelineSearchError(f"Pipeline family is not executable: {spec.family}")
    bands = {tuple(float(item) for item in band) for band in capabilities["bandpasses_hz"]}
    bands.update(
        tuple(float(item) for item in entry["bandpass_hz"])
        for entry in profile_capabilities["individualized_bandpasses"]
    )
    if tuple(spec.bandpass_hz) not in bands:
        raise PipelineSearchError(f"Bandpass is outside executable capabilities: {spec.bandpass_hz}")
    if spec.family == "csp_lda" and spec.csp_components not in capabilities["csp_components"]:
        raise PipelineSearchError(
            f"CSP component count is outside executable capabilities: {spec.csp_components}"
        )
    if spec.lda_shrinkage not in {
        float(item) for item in capabilities["lda_shrinkage"]
    }:
        raise PipelineSearchError("LDA shrinkage is outside executable capabilities")
    if spec.cv_folds not in capabilities["cv_folds"]:
        raise PipelineSearchError("CV fold count is outside executable capabilities")
    channel_options = {
        (
            str(item["channel_strategy"]),
            tuple(str(name) for name in item["selected_channels"]),
        )
        for item in profile_capabilities["channel_options"]
    }
    if (spec.channel_strategy, tuple(spec.selected_channels)) not in channel_options:
        raise PipelineSearchError("Channel selection is outside profile-conditioned capabilities")


def create_pipeline_search_tools(
    *,
    executor: DeterministicPipelineExecutor,
    subject_profile_path: Path,
    frozen_protocol_path: Path,
    autonomy_envelope_path: Path,
    capability_registry_path: Path,
    literature_store_path: Path | None = None,
    literature_sources: dict[str, Any] | None = None,
    literature_search_run_id: str | None = None,
    budget_ledger: BudgetLedger | None = None,
    dataset_incumbent_path: Path | None = None,
) -> tuple[ToolRegistry, dict[str, Any]]:
    subject_path = Path(subject_profile_path).expanduser().resolve()
    protocol_path = Path(frozen_protocol_path).expanduser().resolve()
    envelope_path = Path(autonomy_envelope_path).expanduser().resolve()
    capability_path = Path(capability_registry_path).expanduser().resolve()
    subject_profile = load_json_object(subject_path)
    protocol = load_json_object(protocol_path)
    capabilities = _load_capabilities(capability_path)
    if subject_profile.get("profile_complete") is not True:
        raise PipelineSearchError("Pipeline search requires a completed SubjectProfile")
    if subject_profile.get("subject_id") != executor.subject_id:
        raise PipelineSearchError("SubjectProfile and executor refer to different subjects")
    if protocol.get("status") != "frozen_autonomous":
        raise PipelineSearchError("Pipeline search requires a frozen autonomous protocol")
    dataset_id = str(protocol.get("dataset_id") or "")
    if not dataset_id:
        raise PipelineSearchError("Frozen protocol lacks dataset_id")
    incumbent_path = (
        Path(dataset_incumbent_path).expanduser().resolve()
        if dataset_incumbent_path is not None
        else None
    )
    incumbent: dict[str, Any] | None = None
    incumbent_configuration_hash: str | None = None
    minimum_personalization_gain: float | None = None
    if incumbent_path is not None:
        if not incumbent_path.is_file():
            raise PipelineSearchError("Dataset incumbent artifact is unavailable")
        incumbent = load_json_object(incumbent_path)
        try:
            validate_dataset_incumbent(incumbent)
        except DatasetIncumbentError as exc:
            raise PipelineSearchError(f"Dataset incumbent is invalid: {exc}") from exc
        if incumbent.get("dataset_id") != dataset_id:
            raise PipelineSearchError("Dataset incumbent belongs to another dataset")
        incumbent_configuration_hash = str(incumbent["pipeline_configuration_sha256"])
        minimum_personalization_gain = float(
            incumbent["personalization_policy"]["minimum_search_gain_over_incumbent"]
        )
    capabilities, dataset_contract_policy = _constrain_capabilities_by_dataset_contract(
        protocol=protocol,
        capabilities=capabilities,
    )
    if incumbent is not None:
        incumbent_spec = PipelineSpec.from_dict(incumbent["selected_pipeline"])
        if incumbent_spec.family not in capabilities["families"]:
            raise PipelineSearchError(
                "Dataset incumbent is outside the frozen dataset capability contract"
            )
        bound_contract = (incumbent.get("source_contracts") or {}).get(
            "dataset_level_contract"
        ) or {}
        if dataset_contract_policy is not None and (
            bound_contract.get("sha256") != dataset_contract_policy.get("sha256")
        ):
            raise PipelineSearchError(
                "Dataset incumbent is not bound to the active DatasetLevelContract"
            )
    envelope = load_autonomy_envelope(envelope_path, expected_dataset_id=dataset_id)
    literature_authorized = bool(
        envelope["permissions"].get("allow_network_literature")
    )
    literature_enabled = literature_authorized and literature_store_path is not None
    literature_store = (
        LiteratureStore(Path(literature_store_path)) if literature_enabled else None
    )
    sources = literature_sources or {
        "crossref": CrossrefSource(timeout_seconds=20.0),
        "openalex": OpenAlexSource(timeout_seconds=20.0),
    }
    search_run_id = literature_search_run_id or (
        f"subject-method-{dataset_id}-{executor.subject_id}"
    )
    search_sessions = tuple(
        str(item) for item in protocol["data_roles"]["pipeline_search_and_lock"]
    )
    executor_sessions = tuple(session.session_id for session in executor.sessions)
    if set(search_sessions) != set(executor_sessions):
        raise PipelineSearchError(
            "Executor sessions do not match the frozen pipeline_search_and_lock role"
        )
    profile_capabilities = _derive_profile_conditioned_capabilities(
        subject_profile=subject_profile,
        executor=executor,
        capabilities=capabilities,
    )

    protocol_budget = protocol.get("resource_budget") or {}
    max_executions = int(protocol_budget.get("max_candidate_executions", 0))
    max_cycles = int(protocol_budget.get("max_research_cycles", 0))
    if max_executions < 1 or max_cycles < 1:
        raise PipelineSearchError("Frozen protocol lacks a positive search budget")
    maximum = min(max_executions, max_cycles)
    minimum_before_lock = int(capabilities["minimum_distinct_candidates_before_lock"])
    primary_metric = str((protocol.get("evaluation") or {}).get("primary_metric") or "")
    if not primary_metric:
        raise PipelineSearchError("Frozen protocol lacks a primary evaluation metric")

    registry = ToolRegistry()
    context_read = False
    locked = False
    experiments: dict[str, dict[str, Any]] = {}
    configuration_hashes: set[str] = set()
    literature_queries: dict[str, dict[str, Any]] = {}
    discovered_paper_ids: set[str] = set()

    def require_context() -> None:
        if not context_read:
            raise PipelineSearchError("read_pipeline_search_context must be called first")
        if locked:
            raise PipelineSearchError("Pipeline has already been locked")

    def state() -> dict[str, Any]:
        summaries = [
            {
                "experiment_id": item["experiment_id"],
                "pipeline_id": item["pipeline"]["pipeline_id"],
                "family": item["pipeline"]["family"],
                "metrics": item["metrics"],
                "elapsed_seconds": item["elapsed_seconds"],
                "research_cycle_index": item["research_cycle_index"],
            }
            for item in experiments.values()
        ]
        incumbent_experiment = next(
            (
                item
                for item in experiments.values()
                if item.get("configuration_sha256") == incumbent_configuration_hash
            ),
            None,
        )
        return {
            "subject_id": executor.subject_id,
            "primary_metric": primary_metric,
            "executions_used": len(experiments),
            "executions_remaining": maximum - len(experiments),
            "maximum_research_cycles": maximum,
            "minimum_distinct_candidates_before_lock": minimum_before_lock,
            "experiments": summaries,
            "literature": {
                "authorized": literature_authorized,
                "enabled": literature_enabled,
                "queries_used": len(literature_queries),
                "queries_remaining": max(0, 3 - len(literature_queries)),
                "discovered_paper_count": len(discovered_paper_ids),
                "evidence_scope": "scholarly_metadata_or_abstract_discovery_only",
            },
            "confirmation_data_accessed": False,
            "selective_personalization": {
                "enabled": incumbent is not None,
                "incumbent_evaluated": incumbent_experiment is not None,
                "minimum_search_gain_over_incumbent": minimum_personalization_gain,
            },
        }

    def read_context() -> dict[str, Any]:
        nonlocal context_read
        context_read = True
        return {
            "dataset_id": dataset_id,
            "subject_id": executor.subject_id,
            "subject_profile": subject_profile,
            "frozen_research_design": {
                "protocol_id": protocol["protocol_id"],
                "evaluation": protocol["evaluation"],
                "individual_oracle": protocol["individual_oracle"],
                "stopping_policy": protocol.get("stopping_policy")
                or {"legacy_conditions": protocol.get("stopping_conditions", [])},
                "resource_budget": protocol["resource_budget"],
                "search_sessions": list(search_sessions),
            },
            "executable_capabilities": capabilities,
            "dataset_contract_capability_policy": dataset_contract_policy,
            "profile_conditioned_capabilities": profile_capabilities,
            "dataset_pipeline_incumbent": (
                {
                    "incumbent_id": incumbent["incumbent_id"],
                    "selected_pipeline": incumbent["selected_pipeline"],
                    "pipeline_configuration_sha256": incumbent_configuration_hash,
                    "selected_score_summary": incumbent["selected_score_summary"],
                    "personalization_policy": incumbent["personalization_policy"],
                    "mandatory_control": True,
                }
                if incumbent is not None
                else None
            ),
            "literature_discovery": {
                "authorized": literature_authorized,
                "enabled": literature_enabled,
                "maximum_queries": 3 if literature_enabled else 0,
                "sources": sorted(sources) if literature_enabled else [],
                "evidence_scope": "scholarly_metadata_or_abstract_discovery_only",
            },
            "search_state": state(),
            "provenance": {
                "subject_profile": {"path": str(subject_path), "sha256": sha256_path(subject_path)},
                "frozen_protocol": {"path": str(protocol_path), "sha256": sha256_path(protocol_path)},
                "autonomy_envelope": {"path": str(envelope_path), "sha256": sha256_path(envelope_path)},
                "capability_registry": {"path": str(capability_path), "sha256": sha256_path(capability_path)},
                **(
                    {"dataset_level_contract": dataset_contract_policy}
                    if dataset_contract_policy is not None
                    else {}
                ),
                **(
                    {
                        "dataset_pipeline_incumbent": {
                            "path": str(incumbent_path),
                            "sha256": sha256_path(incumbent_path),
                        }
                    }
                    if incumbent_path is not None
                    else {}
                ),
            },
        }

    def search_method_evidence(
        query_id: str,
        query_text: str,
        rationale: str,
    ) -> dict[str, Any]:
        require_context()
        if not literature_enabled or literature_store is None:
            raise PipelineSearchError("Network literature discovery is not enabled for this run")
        query_id = query_id.strip()
        query_text = query_text.strip()
        rationale = rationale.strip()
        if not query_id or not query_text or not rationale:
            raise PipelineSearchError("Literature query requires non-empty ID, text, and rationale")
        if query_id in literature_queries:
            raise PipelineSearchError("Literature query_id was already used")
        if len(literature_queries) >= 3:
            raise PipelineSearchError("Subject-level literature query budget is exhausted")
        query = LiteratureQuery(
            query_id=query_id,
            text=query_text,
            rationale=rationale,
            source_names=tuple(sorted(sources)),
            limit_per_source=6,
        )
        attempts: list[dict[str, Any]] = []
        papers: dict[str, dict[str, Any]] = {}
        for source_name, source in sources.items():
            try:
                records = source.search(query)
                literature_store.record_search(
                    search_run_id=search_run_id,
                    query=query,
                    source=source_name,
                    papers=records,
                )
                attempts.append(
                    {"source": source_name, "status": "completed", "result_count": len(records)}
                )
                for record in records:
                    discovered_paper_ids.add(record.stable_id)
                    compact = record.to_dict()
                    abstract = compact.get("abstract")
                    compact["stable_id"] = record.stable_id
                    compact["abstract"] = (
                        abstract[:800] if isinstance(abstract, str) else None
                    )
                    compact["evidence_boundary"] = (
                        "Discovery metadata or abstract only; empirical suitability must be tested locally."
                    )
                    papers.setdefault(record.stable_id, compact)
            except Exception as exc:  # Source failures are evidence and may be recovered by another query.
                literature_store.record_search_failure(
                    search_run_id=search_run_id,
                    query=query,
                    source=source_name,
                    error=f"{type(exc).__name__}: {exc}",
                )
                attempts.append(
                    {"source": source_name, "status": "failed", "error": str(exc)[:500]}
                )
        literature_queries[query_id] = {
            "query": query.to_dict(),
            "attempts": attempts,
            "paper_ids": sorted(papers),
        }
        return {
            "query": query.to_dict(),
            "attempts": attempts,
            "papers": list(papers.values()),
            "evidence_scope": "scholarly_metadata_or_abstract_discovery_only",
            "search_state": state(),
        }

    def evaluate_candidate(pipeline: dict[str, Any]) -> dict[str, Any]:
        require_context()
        if len(experiments) >= maximum:
            raise PipelineSearchError("Autonomous search budget is exhausted")
        if budget_ledger is not None:
            budget_ledger.precheck(
                f"{executor.subject_id}:pipeline_candidate",
                {"candidate_executions": 1, "research_cycles": 1},
            )
        spec = PipelineSpec.from_dict(pipeline)
        _validate_capability(spec, capabilities, profile_capabilities)
        configuration_hash = pipeline_configuration_hash(spec)
        if (
            incumbent_configuration_hash is not None
            and not experiments
            and configuration_hash != incumbent_configuration_hash
        ):
            raise PipelineSearchError(
                "The frozen dataset incumbent must be evaluated as the first control"
            )
        if configuration_hash in configuration_hashes:
            raise PipelineSearchError("Equivalent pipeline configuration was already evaluated")
        result = executor.evaluate(spec)
        if budget_ledger is not None:
            budget_ledger.account(
                f"{executor.subject_id}:pipeline_candidate",
                {
                    "candidate_executions": 1,
                    "research_cycles": 1,
                    "compute_seconds": float(result["elapsed_seconds"]),
                },
                metadata={
                    "pipeline_id": spec.pipeline_id,
                    "pipeline_configuration_sha256": configuration_hash,
                },
            )
        if primary_metric not in result["metrics"]:
            raise PipelineSearchError(
                f"Executor result does not contain primary metric {primary_metric!r}"
            )
        value = float(result["metrics"][primary_metric])
        if not 0 <= value <= 1:
            raise PipelineSearchError("Primary metric is outside [0, 1]")
        result["research_cycle_index"] = len(experiments) + 1
        result["configuration_sha256"] = configuration_hash
        experiments[result["experiment_id"]] = result
        configuration_hashes.add(configuration_hash)
        result["budget_after_execution"] = state()
        return result

    def inspect_state() -> dict[str, Any]:
        require_context()
        return state()

    def lock_pipeline(
        selected_experiment_id: str,
        evidence_experiment_ids: list[str],
        selection_rationale: list[str],
        rejected_alternatives: list[str],
        uncertainty: list[str],
        stop_reason: str,
        literature_paper_ids: list[str] | None = None,
        profile_hypothesis_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        nonlocal locked
        require_context()
        if len(configuration_hashes) < minimum_before_lock:
            raise PipelineSearchError(
                f"At least {minimum_before_lock} distinct candidates are required before lock"
            )
        if selected_experiment_id not in experiments:
            raise PipelineSearchError("Selected experiment is not part of this search run")
        cited = set(evidence_experiment_ids)
        if selected_experiment_id not in cited or not cited.issubset(experiments):
            raise PipelineSearchError("Lock evidence must cite available experiments including selection")
        if not selection_rationale or not stop_reason.strip():
            raise PipelineSearchError("Pipeline lock requires rationale and stop reason")
        cited_papers = set(literature_paper_ids or [])
        if literature_enabled and not literature_queries:
            raise PipelineSearchError("At least one subject-level literature query is required")
        if literature_enabled and (not cited_papers or not cited_papers.issubset(discovered_paper_ids)):
            raise PipelineSearchError("Pipeline lock must cite returned literature paper IDs")
        available_hypotheses = {
            str(item.get("hypothesis_id"))
            for item in subject_profile.get("hypotheses") or []
            if item.get("hypothesis_id")
        }
        cited_hypotheses = set(profile_hypothesis_ids or [])
        if not cited_hypotheses.issubset(available_hypotheses):
            raise PipelineSearchError("Pipeline lock cites unavailable subject hypotheses")
        selected = experiments[selected_experiment_id]
        route_decision: dict[str, Any] | None = None
        if incumbent is not None:
            try:
                route_decision = expected_selective_route(
                    incumbent_configuration_sha256=str(incumbent_configuration_hash),
                    minimum_gain=float(minimum_personalization_gain),
                    experiments=list(experiments.values()),
                    primary_metric=primary_metric,
                )
            except DatasetIncumbentError as exc:
                raise PipelineSearchError(
                    f"Selective personalization gate failed: {exc}"
                ) from exc
            if selected_experiment_id != route_decision["required_selected_experiment_id"]:
                raise PipelineSearchError(
                    "Selected experiment violates the frozen selective personalization gate"
                )
            required_route_evidence = {
                route_decision["incumbent_experiment_id"],
                route_decision["best_personalized_experiment_id"],
            }
            if not required_route_evidence.issubset(cited):
                raise PipelineSearchError(
                    "Selective route lock must cite incumbent and best personalized evidence"
                )
        observed_best = max(
            float(item["metrics"][primary_metric]) for item in experiments.values()
        )
        selected_score = float(selected["metrics"][primary_metric])
        tolerance = float(capabilities["observed_best_selection_tolerance"])
        locked = True
        return {
            "schema_version": "1.0",
            "lock_id": f"lock-{selected['pipeline_sha256'][:16]}",
            "status": "locked_awaiting_confirmation",
            "dataset_id": dataset_id,
            "subject_id": executor.subject_id,
            "protocol_id": protocol["protocol_id"],
            "selected_experiment_id": selected_experiment_id,
            "selected_pipeline": selected["pipeline"],
            "pipeline_sha256": selected["pipeline_sha256"],
            "primary_metric": primary_metric,
            "selected_search_score": selected_score,
            "observed_best_search_score": observed_best,
            "gap_to_observed_best": observed_best - selected_score,
            "within_capability_selection_tolerance": (
                observed_best - selected_score <= tolerance
            ),
            "selection_rationale": selection_rationale,
            "rejected_alternatives": rejected_alternatives,
            "uncertainty": uncertainty,
            "stop_reason": stop_reason,
            "evidence_experiment_ids": evidence_experiment_ids,
            "evidence_literature_paper_ids": sorted(cited_papers),
            "evidence_subject_hypothesis_ids": sorted(cited_hypotheses),
            "literature_search": {
                "search_run_id": search_run_id if literature_enabled else None,
                "evidence_scope": "scholarly_metadata_or_abstract_discovery_only",
                "queries": list(literature_queries.values()),
            },
            "search_trace": list(experiments.values()),
            "budget_usage": {
                "research_cycles": len(experiments),
                "candidate_executions": len(experiments),
                "authorized_maximum": maximum,
            },
            "source_contracts": {
                "subject_profile": {"path": str(subject_path), "sha256": sha256_path(subject_path)},
                "frozen_protocol": {"path": str(protocol_path), "sha256": sha256_path(protocol_path)},
                "autonomy_envelope": {"path": str(envelope_path), "sha256": sha256_path(envelope_path), "envelope_id": envelope["envelope_id"]},
                "capability_registry": {"path": str(capability_path), "sha256": sha256_path(capability_path)},
                **(
                    {"dataset_level_contract": dataset_contract_policy}
                    if dataset_contract_policy is not None
                    else {}
                ),
                **(
                    {
                        "literature_store": {
                            "path": str(literature_store.path),
                            "sha256": sha256_path(literature_store.path),
                        }
                    }
                    if literature_enabled and literature_store is not None
                    else {}
                ),
                **(
                    {
                        "dataset_pipeline_incumbent": {
                            "path": str(incumbent_path),
                            "sha256": sha256_path(incumbent_path),
                            "incumbent_id": incumbent["incumbent_id"],
                        }
                    }
                    if incumbent_path is not None and incumbent is not None
                    else {}
                ),
            },
            "profile_conditioned_capabilities": profile_capabilities,
            "route_decision": route_decision,
            "confirmation_accessed": False,
            "search_reopen_allowed_after_confirmation": False,
        }

    registry.register(
        ToolDefinition(
            name="search_subject_method_evidence",
            description="Search scholarly metadata/abstracts for one subject-profile-linked method question.",
            input_schema={
                "type": "object",
                "properties": {
                    "query_id": {"type": "string"},
                    "query_text": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["query_id", "query_text", "rationale"],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="autonomous_subject_literature_discovery",
            tags=("network", "literature", "profile-linked"),
        ),
        search_method_evidence,
    )
    registry.register(
        ToolDefinition(
            name="read_pipeline_search_context",
            description="Read subject evidence, frozen design, executable capabilities, and budget.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            approval="never",
            decision_kind="read_only_pipeline_search_contract",
            tags=("read-only", "search", "no-confirmation"),
        ),
        read_context,
    )
    registry.register(
        ToolDefinition(
            name="evaluate_pipeline_candidate",
            description="Execute one distinct complete pipeline on authorized search data.",
            input_schema={
                "type": "object",
                "properties": {"pipeline": pipeline_spec_schema()},
                "required": ["pipeline"],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="autonomous_budgeted_experiment",
            tags=("numeric", "budgeted", "search-data-only"),
        ),
        evaluate_candidate,
    )
    registry.register(
        ToolDefinition(
            name="inspect_pipeline_search_state",
            description="Inspect completed experiments and remaining budget without new computation.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            approval="never",
            decision_kind="read_only_search_state",
            tags=("read-only", "budget"),
        ),
        inspect_state,
    )
    registry.register(
        ToolDefinition(
            name="lock_pipeline",
            description="Lock one evidence-backed pipeline; no human itemized approval is used.",
            input_schema={
                "type": "object",
                "properties": {
                    "selected_experiment_id": {"type": "string"},
                    "evidence_experiment_ids": {"type": "array", "items": {"type": "string"}},
                    "selection_rationale": {"type": "array", "items": {"type": "string"}},
                    "rejected_alternatives": {"type": "array", "items": {"type": "string"}},
                    "uncertainty": {"type": "array", "items": {"type": "string"}},
                    "stop_reason": {"type": "string"},
                    "literature_paper_ids": {"type": "array", "items": {"type": "string"}},
                    "profile_hypothesis_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "selected_experiment_id",
                    "evidence_experiment_ids",
                    "selection_rationale",
                    "rejected_alternatives",
                    "uncertainty",
                    "stop_reason",
                ],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="autonomous_pipeline_lock",
            tags=("local-write", "pipeline-lock", "confirmation-prerequisite"),
        ),
        lock_pipeline,
    )
    return registry, {
        "dataset_id": dataset_id,
        "subject_id": executor.subject_id,
        "task": "budgeted_individualized_pipeline_search_and_lock",
        "maximum_research_cycles": maximum,
        "primary_metric": primary_metric,
    }


@dataclass
class PipelineSearchAgent:
    runtime: AgentRuntime
    context: dict[str, Any]

    def run(self) -> AgentRunResult:
        return self.runtime.run(
            system_prompt=PIPELINE_SEARCH_SYSTEM_PROMPT,
            user_prompt=json.dumps(self.context, ensure_ascii=False, sort_keys=True),
        )
