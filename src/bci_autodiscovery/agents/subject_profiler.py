"""Autonomous subject profiler over deterministic raw-free measurement tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci_autodiscovery.profiling import (
    SubjectMeasurementEngine,
    validate_dataset_profile,
)
from bci_autodiscovery.workflow.autonomy import load_json_object, sha256_path

from .contracts import AgentRunResult
from .language import DEFAULT_LANGUAGE_INSTRUCTION
from .runtime import AgentRuntime
from .tools import ToolDefinition, ToolRegistry


SUBJECT_PROFILER_SYSTEM_PROMPT = f"""{DEFAULT_LANGUAGE_INSTRUCTION}
You are the autonomous Subject Profiler Agent. First call read_subject_profile_context.
You cannot read raw EEG arrays. Use deterministic tools to measure signal quality, spectral
structure, training-side class separability, and within/cross-session stability only from
the protocol-authorized profiling sessions.

Call every baseline measurement family at least once. You may request measurements for
additional authorized profiling sessions when uncertainty warrants it. Do not call a
diagnostic a validated mechanism: ERD/ERS requires an explicit baseline, quality flags are
not automatic exclusions, and univariate separability is not decoding performance.

Finally call record_subject_profile with evidence-linked hypotheses, uncertainties, and
proposed search implications. If evidence is insufficient, record that status instead of
asking a human to decide or fabricating a strong profile."""


class SubjectProfileError(ValueError):
    pass


def subject_profile_schema() -> dict[str, Any]:
    evidence_ids = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "string", "enum": ["1.0"]},
            "subject_id": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["complete", "insufficient_evidence"],
            },
            "measurement_ids": evidence_ids,
            "quality_summary": {"type": "array", "items": {"type": "string"}},
            "spectral_summary": {"type": "array", "items": {"type": "string"}},
            "erd_ers_summary": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "statement": {"type": "string"},
                    "evidence_measurement_ids": evidence_ids,
                },
                "required": ["status", "statement", "evidence_measurement_ids"],
                "additionalProperties": False,
            },
            "stability_summary": {"type": "array", "items": {"type": "string"}},
            "class_separability_summary": {
                "type": "array",
                "items": {"type": "string"},
            },
            "hypotheses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "hypothesis_id": {"type": "string"},
                        "statement": {"type": "string"},
                        "evidence_measurement_ids": evidence_ids,
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "hypothesis_id",
                        "statement",
                        "evidence_measurement_ids",
                        "confidence",
                    ],
                    "additionalProperties": False,
                },
            },
            "uncertainties": {"type": "array", "items": {"type": "string"}},
            "search_implications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target_stage": {"type": "string"},
                        "recommendation": {"type": "string"},
                        "evidence_measurement_ids": evidence_ids,
                        "status": {
                            "type": "string",
                            "enum": ["proposed_for_budgeted_search"],
                        },
                    },
                    "required": [
                        "target_stage",
                        "recommendation",
                        "evidence_measurement_ids",
                        "status",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "schema_version",
            "subject_id",
            "status",
            "measurement_ids",
            "quality_summary",
            "spectral_summary",
            "erd_ers_summary",
            "stability_summary",
            "class_separability_summary",
            "hypotheses",
            "uncertainties",
            "search_implications",
        ],
        "additionalProperties": False,
    }


def _load_dataset_profile(path: Path) -> dict[str, Any]:
    profile = load_json_object(path)
    validate_dataset_profile(profile)
    return profile


def _normalize_session_ids(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list) or not values:
        raise SubjectProfileError("Protocol profiling role must contain sessions")
    return tuple(str(item) for item in values)


def create_subject_profiler_tools(
    *,
    engine: SubjectMeasurementEngine,
    dataset_profile_path: Path,
    frozen_protocol_path: Path,
) -> tuple[ToolRegistry, dict[str, Any]]:
    profile_path = Path(dataset_profile_path).expanduser().resolve()
    protocol_path = Path(frozen_protocol_path).expanduser().resolve()
    dataset_profile = _load_dataset_profile(profile_path)
    protocol = load_json_object(protocol_path)
    dataset_id = str(dataset_profile["dataset"]["id"])
    if protocol.get("dataset_id") != dataset_id:
        raise SubjectProfileError("DatasetProfile and frozen protocol refer to different datasets")
    if protocol.get("status") != "frozen_autonomous":
        raise SubjectProfileError("Subject profiling requires a frozen autonomous protocol")
    if protocol.get("split_unit") != "session":
        raise SubjectProfileError("Current Subject Profiler implementation requires session roles")
    allowed_sessions = _normalize_session_ids(
        protocol["data_roles"]["profiling_and_calibration"]
    )
    if set(engine.allowed_session_ids) != set(allowed_sessions):
        raise SubjectProfileError(
            "Measurement engine authorization does not match the frozen profiling role"
        )

    registry = ToolRegistry()
    context_read = False
    profile_recorded = False
    measurements: dict[str, dict[str, Any]] = {}
    measured_kinds: set[str] = set()

    def retain(value: dict[str, Any]) -> dict[str, Any]:
        measurement_id = str(value["measurement_id"])
        measurements[measurement_id] = value
        measured_kinds.add(str(value["kind"]))
        return value

    def require_context() -> None:
        if not context_read:
            raise SubjectProfileError("read_subject_profile_context must be called first")
        if profile_recorded:
            raise SubjectProfileError("Subject profile has already been recorded")

    def read_context() -> dict[str, Any]:
        nonlocal context_read
        context_read = True
        return {
            "dataset_id": dataset_id,
            "subject_id": engine.subject_id,
            "authorized_profiling_sessions": list(allowed_sessions),
            "measurement_catalog": [
                "signal_quality",
                "spectral_profile",
                "class_separability",
                "stability",
            ],
            "interpretation_rules": [
                "raw EEG arrays never enter model context",
                "quality flags do not automatically exclude data",
                "ERD/ERS requires a documented baseline",
                "class separability is not cross-validated decoding performance",
                "search implications remain hypotheses until budgeted execution",
            ],
            "dataset_profile": dataset_profile,
            "protocol_summary": {
                "protocol_id": protocol["protocol_id"],
                "status": protocol["status"],
                "split_unit": protocol["split_unit"],
                "profiling_sessions": list(allowed_sessions),
            },
            "provenance": {
                "dataset_profile": {
                    "path": str(profile_path),
                    "sha256": sha256_path(profile_path),
                },
                "frozen_protocol": {
                    "path": str(protocol_path),
                    "sha256": sha256_path(protocol_path),
                },
            },
        }

    def quality(session_id: str) -> dict[str, Any]:
        require_context()
        return retain(engine.quality(session_id))

    def spectral(session_id: str) -> dict[str, Any]:
        require_context()
        return retain(engine.spectral(session_id))

    def separability(session_id: str) -> dict[str, Any]:
        require_context()
        return retain(engine.separability(session_id))

    def stability(session_ids: list[str]) -> dict[str, Any]:
        require_context()
        return retain(engine.stability(session_ids))

    def record_subject_profile(subject_profile: dict[str, Any]) -> dict[str, Any]:
        nonlocal profile_recorded
        require_context()
        required_kinds = {
            "signal_quality",
            "spectral_profile",
            "class_separability",
            "stability",
        }
        missing_kinds = sorted(required_kinds.difference(measured_kinds))
        if missing_kinds:
            raise SubjectProfileError(
                f"Baseline deterministic measurements are incomplete: {missing_kinds}"
            )
        if subject_profile.get("subject_id") != engine.subject_id:
            raise SubjectProfileError("Subject profile belongs to another subject")
        cited = subject_profile.get("measurement_ids")
        if not isinstance(cited, list) or set(cited) != set(measurements):
            raise SubjectProfileError(
                "Subject profile must cite every measurement from the current run exactly"
            )
        hypothesis_ids: set[str] = set()
        for hypothesis in subject_profile.get("hypotheses") or []:
            hypothesis_id = str(hypothesis.get("hypothesis_id") or "").strip()
            if not hypothesis_id or hypothesis_id in hypothesis_ids:
                raise SubjectProfileError("Hypothesis IDs must be non-empty and unique")
            hypothesis_ids.add(hypothesis_id)
            confidence = hypothesis.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise SubjectProfileError(f"Hypothesis {hypothesis_id} confidence must be numeric")
            if not 0 <= float(confidence) <= 1:
                raise SubjectProfileError(
                    f"Hypothesis {hypothesis_id} confidence must be in [0, 1]"
                )
            refs = set(hypothesis.get("evidence_measurement_ids") or [])
            if not refs or not refs.issubset(measurements):
                raise SubjectProfileError(
                    f"Hypothesis {hypothesis_id} cites unavailable measurements"
                )
        for implication in subject_profile.get("search_implications") or []:
            refs = set(implication.get("evidence_measurement_ids") or [])
            if not refs or not refs.issubset(measurements):
                raise SubjectProfileError("Search implication cites unavailable measurements")
            if implication.get("status") != "proposed_for_budgeted_search":
                raise SubjectProfileError("Subject profile cannot directly activate a search choice")
        erd_refs = set(
            (subject_profile.get("erd_ers_summary") or {}).get(
                "evidence_measurement_ids"
            )
            or []
        )
        if not erd_refs or not erd_refs.issubset(measurements):
            raise SubjectProfileError("ERD/ERS summary must cite available measurements")

        profile_recorded = True
        result = json.loads(json.dumps(subject_profile))
        result["profile_complete"] = True
        result["measurements"] = list(measurements.values())
        result["source_contracts"] = {
            "dataset_profile": {
                "path": str(profile_path),
                "sha256": sha256_path(profile_path),
            },
            "frozen_protocol": {
                "path": str(protocol_path),
                "sha256": sha256_path(protocol_path),
            },
        }
        result["raw_arrays_exposed_to_model"] = False
        return result

    registry.register(
        ToolDefinition(
            name="read_subject_profile_context",
            description="Read subject identity, authorized sessions, contracts, and measurement catalog.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            approval="never",
            decision_kind="read_only_subject_contract",
            tags=("read-only", "subject", "no-raw-arrays"),
        ),
        read_context,
    )
    for name, description, handler in (
        (
            "measure_subject_signal_quality",
            "Measure finite values, robust channel/trial flags, and dispersion for one authorized session.",
            quality,
        ),
        (
            "measure_subject_spectral_profile",
            "Measure deterministic bandpower and individual peak candidates for one authorized session.",
            spectral,
        ),
        (
            "measure_subject_class_separability",
            "Measure a training-side univariate class-effect diagnostic for one authorized session.",
            separability,
        ),
    ):
        registry.register(
            ToolDefinition(
                name=name,
                description=description,
                input_schema={
                    "type": "object",
                    "properties": {"session_id": {"type": "string"}},
                    "required": ["session_id"],
                    "additionalProperties": False,
                },
                approval="never",
                decision_kind="deterministic_subject_measurement",
                tags=("read-only", "numeric", "no-raw-arrays"),
            ),
            handler,
        )
    registry.register(
        ToolDefinition(
            name="measure_subject_stability",
            description="Measure split-half stability and authorized cross-session drift.",
            input_schema={
                "type": "object",
                "properties": {
                    "session_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["session_ids"],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="deterministic_subject_measurement",
            tags=("read-only", "numeric", "no-raw-arrays"),
        ),
        stability,
    )
    registry.register(
        ToolDefinition(
            name="record_subject_profile",
            description="Record an evidence-linked subject profile and search hypotheses.",
            input_schema={
                "type": "object",
                "properties": {"subject_profile": subject_profile_schema()},
                "required": ["subject_profile"],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="autonomous_subject_profile",
            tags=("local-write", "subject-profile", "evidence-linked"),
        ),
        record_subject_profile,
    )
    return registry, {
        "dataset_id": dataset_id,
        "subject_id": engine.subject_id,
        "task": "measure_interpret_and_record_subject_profile",
        "authorized_profiling_sessions": list(allowed_sessions),
    }


@dataclass
class SubjectProfilerAgent:
    runtime: AgentRuntime
    context: dict[str, Any]

    def run(self) -> AgentRunResult:
        return self.runtime.run(
            system_prompt=SUBJECT_PROFILER_SYSTEM_PROMPT,
            user_prompt=json.dumps(self.context, ensure_ascii=False, sort_keys=True),
        )
