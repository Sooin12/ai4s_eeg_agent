"""Deterministic validation for explicit non-BIDS EEG semantic sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .dataset import DatasetProfileError, validate_dataset_profile


if os.name == "nt":
    os.environ.setdefault("WINDIR", str(Path(os.environ.get("SystemRoot", r"C:\Windows"))))
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mpl-config"))


SEMANTICS_FILENAME = "dataset_semantics.json"
VALIDATION_SCHEMAS = {
    "mne_raw": "mne_eeg_semantic_validation_v1",
    "declarative": "declarative_eeg_semantic_validation_v1",
    "mat_epochs": "mat_epoch_semantic_validation_v1",
}


def sha256_path(path: Path) -> str:
    target = Path(path).resolve()
    digest = hashlib.sha256()
    if target.is_file():
        with target.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    if not target.is_dir():
        raise DatasetProfileError(f"Cannot hash missing source: {target}")
    for item in sorted(path for path in target.rglob("*") if path.is_file()):
        digest.update(item.relative_to(target).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_path(item)))
    return digest.hexdigest()


def load_semantics(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetProfileError(f"Cannot read semantic sidecar: {exc}") from exc
    if payload.get("schema_version") != "1.0":
        raise DatasetProfileError("Unsupported dataset_semantics schema_version")
    kind = payload.get("adapter_kind")
    if kind not in VALIDATION_SCHEMAS:
        raise DatasetProfileError(f"Unsupported semantic adapter_kind: {kind}")
    globs = payload.get("source_globs")
    if not isinstance(globs, list) or not globs or any(
        not isinstance(item, str) or not item.strip() for item in globs
    ):
        raise DatasetProfileError("dataset_semantics.source_globs must be non-empty")
    if kind in {"mne_raw", "declarative"}:
        profile = payload.get("profile")
        if not isinstance(profile, dict):
            raise DatasetProfileError("dataset_semantics.profile must be an object")
        candidate = json.loads(json.dumps(profile))
        candidate["schema_version"] = "1.0"
        candidate["evidence"] = [{"source": str(resolved), "claim": "declared semantics"}]
        validate_dataset_profile(candidate)
    if kind == "mne_raw":
        mne_mapping = payload.get("mne_mapping")
        if not isinstance(mne_mapping, dict):
            raise DatasetProfileError("mne_raw semantics require mne_mapping")
        for field in ("subject_regex", "event_mapping"):
            if field not in mne_mapping:
                raise DatasetProfileError(f"mne_mapping.{field} is required")
        _compile_identity_regex(str(mne_mapping["subject_regex"]), "subject")
        if mne_mapping.get("session_regex"):
            _compile_identity_regex(str(mne_mapping["session_regex"]), "session")
        mapping = mne_mapping["event_mapping"]
        if not isinstance(mapping, dict) or not mapping:
            raise DatasetProfileError("mne_mapping.event_mapping must be non-empty")
    elif kind == "declarative":
        container = payload.get("container_mapping")
        if not isinstance(container, dict):
            raise DatasetProfileError("declarative semantics require container_mapping")
        for field in (
            "signal_object_or_stream",
            "axis_order",
            "event_source",
            "event_code_mapping",
            "data_role_policy",
        ):
            if field not in container:
                raise DatasetProfileError(f"container_mapping.{field} is required")
        axes = container["axis_order"]
        if (
            not isinstance(axes, list)
            or len(axes) < 2
            or any(not isinstance(item, str) or not item for item in axes)
            or len(axes) != len(set(axes))
            or not {"channel", "sample"}.issubset(axes)
        ):
            raise DatasetProfileError(
                "container_mapping.axis_order must uniquely include channel and sample"
            )
        event_mapping = container["event_code_mapping"]
        if not isinstance(event_mapping, dict) or not event_mapping:
            raise DatasetProfileError(
                "container_mapping.event_code_mapping must be non-empty"
            )
        if not isinstance(container["data_role_policy"], str) or not container[
            "data_role_policy"
        ].strip():
            raise DatasetProfileError("container_mapping.data_role_policy must be text")
    else:
        _validate_mat_epoch_mapping(payload)
    payload["_resolved_path"] = str(resolved)
    return payload


def discover_sources(dataset_root: Path, globs: list[str]) -> list[Path]:
    root = Path(dataset_root).expanduser().resolve()
    matches: dict[str, Path] = {}
    for pattern in globs:
        for path in root.glob(pattern):
            resolved = path.resolve()
            if resolved == root or root not in resolved.parents:
                raise DatasetProfileError(f"Semantic source escaped dataset root: {path}")
            matches[str(resolved).lower()] = resolved
    sources = sorted(matches.values(), key=lambda item: str(item).lower())
    if not sources:
        raise DatasetProfileError("Semantic sidecar source_globs matched no files")
    return sources


def validate_semantic_dataset(
    *, dataset_root: Path, semantics_path: Path, output_path: Path
) -> dict[str, Any]:
    root = Path(dataset_root).expanduser().resolve()
    semantics_file = Path(semantics_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise DatasetProfileError(f"Refusing to overwrite validation artifact: {output}")
    semantics = load_semantics(semantics_file)
    sources = discover_sources(root, semantics["source_globs"])
    source_records = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "kind": "directory" if path.is_dir() else "file",
            "size_bytes": _source_size_bytes(path),
            "sha256": _source_sha256(path),
            "dependencies": [
                dependency.relative_to(root).as_posix()
                for dependency in _source_dependencies(path)
            ],
        }
        for path in sources
    ]
    validation: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMAS[semantics["adapter_kind"]],
        "dataset_root": str(root),
        "semantic_sidecar": {
            "path": str(semantics_file),
            "sha256": sha256_path(semantics_file),
        },
        "adapter_kind": semantics["adapter_kind"],
        "sources": source_records,
        "summary": {
            "source_count": len(sources),
            "total_size_bytes": sum(item["size_bytes"] for item in source_records),
        },
        "checks": {
            "semantic_schema_valid": True,
            "all_sources_hash_bound": True,
            "mne_header_and_event_match": None,
        },
    }
    if semantics["adapter_kind"] == "mne_raw":
        observed = _inspect_mne_sources(root, sources, semantics)
        _assert_mne_profile_matches(semantics["profile"], observed)
        validation["observed"] = observed
        validation["checks"]["mne_header_and_event_match"] = True
    elif semantics["adapter_kind"] == "mat_epochs":
        validation["observed"] = _inspect_mat_epoch_sources(root, sources, semantics)
        validation["checks"]["mat_arrays_and_events_match"] = True
    else:
        validation["limitations"] = [
            "Container hashes and the explicit standard profile are validated, but arbitrary "
            "container internals require a future deterministic reader for independent semantic verification."
        ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validation


def verify_semantic_validation(
    *, dataset_root: Path, semantics_path: Path, validation_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(dataset_root).expanduser().resolve()
    semantics_file = Path(semantics_path).expanduser().resolve()
    validation_file = Path(validation_path).expanduser().resolve()
    semantics = load_semantics(semantics_file)
    try:
        validation = json.loads(validation_file.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetProfileError(f"Cannot read semantic validation: {exc}") from exc
    expected_schema = VALIDATION_SCHEMAS[semantics["adapter_kind"]]
    if validation.get("schema_version") != expected_schema:
        raise DatasetProfileError("Semantic validation schema or adapter kind disagrees")
    if Path(str(validation.get("dataset_root"))).resolve() != root:
        raise DatasetProfileError("Semantic validation belongs to another dataset root")
    sidecar = validation.get("semantic_sidecar") or {}
    if (
        Path(str(sidecar.get("path"))).resolve() != semantics_file
        or sidecar.get("sha256") != sha256_path(semantics_file)
    ):
        raise DatasetProfileError("Semantic sidecar hash binding failed")
    checks = validation.get("checks") or {}
    records = validation.get("sources")
    if (
        not isinstance(records, list)
        or not records
        or checks.get("semantic_schema_valid") is not True
        or checks.get("all_sources_hash_bound") is not True
    ):
        raise DatasetProfileError("Semantic validation contains no verified source inventory")
    if semantics["adapter_kind"] == "mne_raw" and checks.get(
        "mne_header_and_event_match"
    ) is not True:
        raise DatasetProfileError("MNE header and event validation did not pass")
    if semantics["adapter_kind"] == "mat_epochs" and checks.get(
        "mat_arrays_and_events_match"
    ) is not True:
        raise DatasetProfileError("MAT epoch array and event validation did not pass")
    for record in records:
        if not isinstance(record, dict) or not record.get("relative_path"):
            raise DatasetProfileError("Semantic validation source record is malformed")
        source = (root / str(record["relative_path"])).resolve()
        if root not in source.parents or not source.exists():
            raise DatasetProfileError(f"Validated semantic source is unavailable: {source}")
        if _source_sha256(source) != record.get("sha256"):
            raise DatasetProfileError(f"Validated semantic source changed: {source}")
    return semantics, validation


def profile_from_semantic_validation(
    *,
    semantics: dict[str, Any],
    validation_path: Path,
    dataset_id_hint: str | None,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if semantics["adapter_kind"] == "mat_epochs":
        if validation is None:
            raise DatasetProfileError("MAT epoch profiling requires verified observations")
        profile = _profile_from_mat_epoch_observations(semantics, validation)
    else:
        profile = json.loads(json.dumps(semantics["profile"]))
        profile["schema_version"] = "1.0"
    if dataset_id_hint and dataset_id_hint != "auto":
        profile["dataset"]["id"] = dataset_id_hint
    semantics_path = Path(semantics["_resolved_path"])
    validation_file = Path(validation_path).resolve()
    profile["evidence"] = [
        {
            "claim": "explicit dataset semantics",
            "source": str(semantics_path),
            "sha256": sha256_path(semantics_path),
        },
        {
            "claim": "source hashes and deterministic adapter validation",
            "source": str(validation_file),
            "sha256": sha256_path(validation_file),
        },
    ]
    return profile


def _validate_mat_epoch_mapping(payload: dict[str, Any]) -> None:
    mapping = payload.get("mat_mapping")
    if not isinstance(mapping, dict):
        raise DatasetProfileError("mat_epochs semantics require mat_mapping")
    required = (
        "signal_key",
        "label_key",
        "axis_order",
        "subject_regex",
        "session_regex",
        "event_code_mapping",
        "sampling_frequency_hz",
        "channel_names",
        "channel_units",
        "data_role_policy",
    )
    for field in required:
        if field not in mapping:
            raise DatasetProfileError(f"mat_mapping.{field} is required")
    axes = mapping["axis_order"]
    if (
        not isinstance(axes, list)
        or len(axes) != 3
        or set(axes) != {"channel", "sample", "trial"}
    ):
        raise DatasetProfileError(
            "mat_mapping.axis_order must contain channel, sample and trial exactly once"
        )
    _compile_identity_regex(str(mapping["subject_regex"]), "subject")
    _compile_identity_regex(str(mapping["session_regex"]), "session")
    if mapping.get("run_regex"):
        _compile_identity_regex(str(mapping["run_regex"]), "run")
    for field, group in (
        ("subject_assertion_regex", "subject"),
        ("session_assertion_regex", "session"),
        ("run_assertion_regex", "run"),
    ):
        if mapping.get(field):
            _compile_identity_regex(str(mapping[field]), group)
    if mapping.get("run_assertion_regex") and not mapping.get("run_regex"):
        raise DatasetProfileError(
            "mat_mapping.run_assertion_regex requires an authoritative run_regex"
        )
    event_mapping = mapping["event_code_mapping"]
    if not isinstance(event_mapping, dict) or not event_mapping:
        raise DatasetProfileError("mat_mapping.event_code_mapping must be non-empty")
    if float(mapping["sampling_frequency_hz"]) <= 0:
        raise DatasetProfileError("mat_mapping.sampling_frequency_hz must be positive")
    channels = mapping["channel_names"]
    if (
        not isinstance(channels, list)
        or not channels
        or any(not isinstance(item, str) or not item.strip() for item in channels)
        or len(channels) != len(set(channels))
    ):
        raise DatasetProfileError("mat_mapping.channel_names must be unique text labels")
    for field in ("signal_key", "label_key", "channel_units", "data_role_policy"):
        if not isinstance(mapping[field], str) or not mapping[field].strip():
            raise DatasetProfileError(f"mat_mapping.{field} must be non-empty text")
    hints = payload.get("profile_hints")
    if not isinstance(hints, dict):
        raise DatasetProfileError("mat_epochs semantics require profile_hints")
    for section in (
        "dataset",
        "paradigm",
        "resting_state",
        "signal",
        "equipment",
        "events",
        "constraints",
    ):
        if not isinstance(hints.get(section), dict):
            raise DatasetProfileError(f"profile_hints.{section} must be an object")
    if not str(hints["dataset"].get("id") or "").strip():
        raise DatasetProfileError("profile_hints.dataset.id is required")
    if not str(hints["paradigm"].get("family") or "").strip():
        raise DatasetProfileError("profile_hints.paradigm.family is required")
    for field in ("allowed", "forbidden", "requires_research_design_decision"):
        values = hints["constraints"].get(field)
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise DatasetProfileError(f"profile_hints.constraints.{field} must be text array")


def _inspect_mat_epoch_sources(
    root: Path, sources: list[Path], semantics: dict[str, Any]
) -> dict[str, Any]:
    try:
        import numpy as np
        from scipy.io import loadmat, whosmat
    except Exception as exc:
        raise DatasetProfileError(f"NumPy and SciPy are required for MAT validation: {exc}") from exc

    mapping = semantics["mat_mapping"]
    signal_key = str(mapping["signal_key"])
    label_key = str(mapping["label_key"])
    axes = list(mapping["axis_order"])
    channel_axis = axes.index("channel")
    sample_axis = axes.index("sample")
    trial_axis = axes.index("trial")
    subject_regex = _compile_identity_regex(str(mapping["subject_regex"]), "subject")
    session_regex = _compile_identity_regex(str(mapping["session_regex"]), "session")
    run_regex = (
        _compile_identity_regex(str(mapping["run_regex"]), "run")
        if mapping.get("run_regex")
        else None
    )
    assertion_regexes = {
        group: _compile_identity_regex(str(mapping[field]), group)
        for field, group in (
            ("subject_assertion_regex", "subject"),
            ("session_assertion_regex", "session"),
            ("run_assertion_regex", "run"),
        )
        if mapping.get(field)
    }
    event_mapping = {str(key): str(value) for key, value in mapping["event_code_mapping"].items()}
    expected_channels = len(mapping["channel_names"])
    records: list[dict[str, Any]] = []
    signatures: set[tuple[int, int]] = set()
    identities: set[tuple[str, str, str]] = set()

    for source in sources:
        relative = source.relative_to(root).as_posix()
        subject_match = subject_regex.search(relative)
        session_match = session_regex.search(relative)
        if not subject_match or not session_match:
            raise DatasetProfileError(f"Identity regex did not match MAT source: {relative}")
        subject = subject_match.group("subject")
        session = session_match.group("session")
        if run_regex:
            run_match = run_regex.search(relative)
            if not run_match:
                raise DatasetProfileError(f"run_regex did not match MAT source: {relative}")
            run = run_match.group("run")
        else:
            run = "default"
        authoritative_identity = {"subject": subject, "session": session, "run": run}
        asserted_identity: dict[str, str | None] = {}
        identity_conflicts: list[dict[str, str | None]] = []
        for group, assertion_regex in assertion_regexes.items():
            assertion_match = assertion_regex.search(relative)
            asserted = assertion_match.group(group) if assertion_match else None
            asserted_identity[group] = asserted
            if asserted != authoritative_identity[group]:
                identity_conflicts.append(
                    {
                        "field": group,
                        "authoritative_value": authoritative_identity[group],
                        "asserted_value": asserted,
                    }
                )
        identity = (subject, session, run)
        if identity in identities:
            raise DatasetProfileError(f"Duplicate subject/session/run MAT source: {identity}")
        identities.add(identity)

        inventory = {name: tuple(shape) for name, shape, _kind in whosmat(source)}
        if signal_key not in inventory or label_key not in inventory:
            raise DatasetProfileError(
                f"MAT source lacks declared objects {signal_key!r}/{label_key!r}: {relative}"
            )
        shape = inventory[signal_key]
        if len(shape) != 3:
            raise DatasetProfileError(f"MAT signal must be 3-D: {relative} has {shape}")
        if int(shape[channel_axis]) != expected_channels:
            raise DatasetProfileError(
                f"MAT channel count disagrees with semantics: {relative} has {shape[channel_axis]}"
            )
        payload = loadmat(source, variable_names=[signal_key, label_key])
        signal = np.asarray(payload[signal_key])
        labels = np.asarray(payload[label_key]).reshape(-1)
        trials = int(shape[trial_axis])
        samples = int(shape[sample_axis])
        if labels.size != trials:
            raise DatasetProfileError(
                f"MAT labels/trials disagree: {relative} has {labels.size}/{trials}"
            )
        label_counts: Counter[str] = Counter()
        for value in labels.tolist():
            key = (
                str(int(value))
                if isinstance(value, (int, float, np.integer, np.floating))
                and float(value).is_integer()
                else str(value)
            )
            if key not in event_mapping:
                raise DatasetProfileError(f"Unmapped MAT event code {key!r} in {relative}")
            label_counts[event_mapping[key]] += 1
        finite = bool(np.isfinite(signal).all())
        standardized = np.moveaxis(signal, (trial_axis, channel_axis, sample_axis), (0, 1, 2))
        flat_channels = int(np.count_nonzero(np.ptp(standardized, axis=(0, 2)) == 0))
        signatures.add((expected_channels, samples))
        identity_consistent = not identity_conflicts
        records.append(
            {
                "relative_path": relative,
                "subject_id": subject,
                "session_id": session,
                "run_id": run,
                "asserted_identity": asserted_identity,
                "identity_conflicts": identity_conflicts,
                "identity_consistent": identity_consistent,
                "shape": list(shape),
                "channels": expected_channels,
                "samples": samples,
                "trials": trials,
                "class_counts": dict(sorted(label_counts.items())),
                "all_finite": finite,
                "flat_channel_count": flat_channels,
                "valid": finite and flat_channels == 0 and identity_consistent,
            }
        )
    if len(signatures) != 1:
        raise DatasetProfileError(f"MAT epoch channel/sample signatures disagree: {sorted(signatures)}")

    trial_counts = Counter(record["trials"] for record in records)
    nominal_trials = trial_counts.most_common(1)[0][0]
    return {
        "reader": "scipy_mat_epoch_v1",
        "records": records,
        "channel_count": next(iter(signatures))[0],
        "sample_count": next(iter(signatures))[1],
        "nominal_trials_per_run": nominal_trials,
        "identity_conflict_count": sum(
            not bool(record["identity_consistent"]) for record in records
        ),
    }


def _profile_from_mat_epoch_observations(
    semantics: dict[str, Any], validation: dict[str, Any]
) -> dict[str, Any]:
    observed = validation.get("observed") or {}
    records = observed.get("records")
    if not isinstance(records, list) or not records:
        raise DatasetProfileError("MAT semantic validation contains no observed records")
    mapping = semantics["mat_mapping"]
    hints = json.loads(json.dumps(semantics["profile_hints"]))
    subjects = sorted({str(item["subject_id"]) for item in records})
    sessions = sorted({str(item["session_id"]) for item in records}, key=_identity_sort_key)
    sessions_by_subject: dict[str, set[str]] = defaultdict(set)
    runs_by_subject_session: Counter[tuple[str, str]] = Counter()
    class_counts: Counter[str] = Counter()
    for item in records:
        sessions_by_subject[str(item["subject_id"])].add(str(item["session_id"]))
        runs_by_subject_session[(str(item["subject_id"]), str(item["session_id"]))] += 1
        class_counts.update({str(key): int(value) for key, value in item["class_counts"].items()})
    nominal = int(observed["nominal_trials_per_run"])
    missing_subject_sessions = [
        {"subject_id": subject, "session_id": session}
        for subject in subjects
        for session in sessions
        if session not in sessions_by_subject[subject]
    ]
    anomalies = [
        {
            "subject_id": str(item["subject_id"]),
            "session_id": str(item["session_id"]),
            "relative_path": str(item["relative_path"]),
            "observed_trials": int(item["trials"]),
            "nominal_trials": nominal,
            "difference": int(item["trials"]) - nominal,
        }
        for item in records
        if int(item["trials"]) != nominal
    ]
    identity_conflicts = [
        {
            "relative_path": str(item["relative_path"]),
            "subject_id": str(item["subject_id"]),
            "session_id": str(item["session_id"]),
            "run_id": str(item.get("run_id", "default")),
            "conflicts": list(item.get("identity_conflicts") or []),
        }
        for item in records
        if not bool(item.get("identity_consistent", True))
    ]
    sampling_rate = float(mapping["sampling_frequency_hz"])
    duration = int(observed["sample_count"]) / sampling_rate
    actions = [
        {
            "label": label,
            "values": sorted(
                key for key, value in mapping["event_code_mapping"].items() if str(value) == label
            ),
            "trial_count": int(class_counts[label]),
        }
        for label in sorted(class_counts)
    ]
    signal = hints["signal"]
    signal.update(
        {
            "sampling_frequency_hz": sampling_rate,
            "recording_type": "epoched",
            "channel_count": len(mapping["channel_names"]),
            "channel_names": list(mapping["channel_names"]),
            "channel_units": [str(mapping["channel_units"])],
        }
    )
    events = hints["events"]
    events.update(
        {
            "labels": sorted(class_counts),
            "total": sum(class_counts.values()),
            "class_counts": dict(sorted(class_counts.items())),
            "nominal_durations_s": [duration],
            "common_analysis_window_s": [0.0, duration],
            "common_window_reason": "Validated common MAT sample count divided by declared sampling rate.",
        }
    )
    hints["paradigm"]["actions"] = actions
    profile = {
        "schema_version": "1.0",
        "dataset": hints["dataset"],
        "paradigm": hints["paradigm"],
        "resting_state": hints["resting_state"],
        "signal": signal,
        "equipment": hints["equipment"],
        "events": events,
        "sessions": {
            "subjects": len(subjects),
            "sessions_per_subject": max(len(value) for value in sessions_by_subject.values()),
            "total_subject_sessions": len(records),
            "session_indices": sessions,
            "runs_per_session": {
                session: max(
                    runs_by_subject_session[(subject, session)] for subject in subjects
                )
                for session in sessions
            },
            "longitudinal": len(sessions) > 1,
            "separate_days_declared": bool(hints.get("sessions", {}).get("separate_days_declared", False)),
        },
        "volume": {
            "subjects": len(subjects),
            "subject_sessions": len(records),
            "runs": len(records),
            "trials": sum(int(item["trials"]) for item in records),
            "signal_bytes": int(validation["summary"]["total_size_bytes"]),
        },
        "quality": {
            "core_valid_runs": sum(bool(item["valid"]) for item in records),
            "core_invalid_runs": sum(not bool(item["valid"]) for item in records),
            "nonfinite_runs": sum(not bool(item["all_finite"]) for item in records),
            "flat_channel_runs": sum(int(item["flat_channel_count"]) > 0 for item in records),
            "automatic_exclusions": 0,
            "nominal_trials_per_subject_session": nominal,
            "trial_count_anomalies": anomalies,
            "missing_trials": sum(max(0, -int(item["difference"])) for item in anomalies),
            "missing_subject_sessions": missing_subject_sessions,
            "identity_conflict_count": len(identity_conflicts),
            "identity_conflicts": identity_conflicts,
        },
        "constraints": hints["constraints"],
    }
    profile["constraints"].setdefault("external_authority_blockers", [])
    if identity_conflicts:
        blocker = (
            "reconcile conflicting source identity declarations before "
            "subject-level execution"
        )
        if blocker not in profile["constraints"]["external_authority_blockers"]:
            profile["constraints"]["external_authority_blockers"].append(blocker)
    validate_dataset_profile({
        **profile,
        "evidence": [{"source": semantics["_resolved_path"], "claim": "declared semantics"}],
    })
    return profile


def _identity_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _compile_identity_regex(pattern: str, group: str) -> re.Pattern[str]:
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise DatasetProfileError(f"Invalid {group} regex: {exc}") from exc
    if group not in compiled.groupindex:
        raise DatasetProfileError(f"Identity regex requires named group {group!r}")
    return compiled


def _inspect_mne_sources(
    root: Path, sources: list[Path], semantics: dict[str, Any]
) -> dict[str, Any]:
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    os.environ.setdefault("MNE_LOGGING_LEVEL", "ERROR")
    try:
        import mne
    except Exception as exc:
        raise DatasetProfileError(f"MNE is required for mne_raw validation: {exc}") from exc
    mapping = semantics["mne_mapping"]
    subject_regex = _compile_identity_regex(str(mapping["subject_regex"]), "subject")
    session_regex = (
        _compile_identity_regex(str(mapping["session_regex"]), "session")
        if mapping.get("session_regex")
        else None
    )
    event_mapping = {str(key): str(value) for key, value in mapping["event_mapping"].items()}
    class_counts: Counter[str] = Counter()
    durations: set[float] = set()
    subjects: set[str] = set()
    sessions_by_subject: dict[str, set[str]] = defaultdict(set)
    channel_signatures: set[tuple[tuple[str, str], ...]] = set()
    sampling_rates: set[float] = set()
    line_frequencies: set[float] = set()
    for source in sources:
        relative = source.relative_to(root).as_posix()
        subject_match = subject_regex.search(relative)
        if not subject_match:
            raise DatasetProfileError(f"subject_regex did not match {relative}")
        subject = subject_match.group("subject")
        if session_regex:
            session_match = session_regex.search(relative)
            if not session_match:
                raise DatasetProfileError(f"session_regex did not match {relative}")
            session = session_match.group("session")
        else:
            session = "default"
        raw = _read_mne_raw(mne, source)
        subjects.add(subject)
        sessions_by_subject[subject].add(session)
        sampling_rates.add(float(raw.info["sfreq"]))
        if raw.info.get("line_freq") is not None:
            line_frequencies.add(float(raw.info["line_freq"]))
        channel_signatures.add(tuple(zip(raw.ch_names, raw.get_channel_types(), strict=True)))
        for description, duration in zip(
            raw.annotations.description, raw.annotations.duration, strict=True
        ):
            key = str(description)
            if key not in event_mapping:
                raise DatasetProfileError(f"Unmapped MNE annotation {key!r} in {relative}")
            class_counts[event_mapping[key]] += 1
            durations.add(float(duration))
    if len(sampling_rates) != 1 or len(channel_signatures) != 1:
        raise DatasetProfileError("MNE recordings disagree on sampling rate or channel signature")
    signature = next(iter(channel_signatures))
    session_labels = sorted({item for values in sessions_by_subject.values() for item in values})
    per_subject = {len(values) for values in sessions_by_subject.values()}
    if len(per_subject) != 1:
        raise DatasetProfileError("Subjects disagree on session count")
    return {
        "sampling_frequency_hz": next(iter(sampling_rates)),
        "channel_names": [item[0] for item in signature],
        "channel_types": [item[1].upper() for item in signature],
        "channel_count": len(signature),
        "eog_channel_count": sum(1 for _name, kind in signature if kind == "eog"),
        "power_line_frequency_hz": (
            next(iter(line_frequencies)) if len(line_frequencies) == 1 else None
        ),
        "class_counts": dict(sorted(class_counts.items())),
        "event_total": sum(class_counts.values()),
        "event_durations_s": sorted(durations),
        "subjects": len(subjects),
        "sessions_per_subject": next(iter(per_subject)),
        "session_indices": session_labels,
        "runs": len(sources),
    }


def _read_mne_raw(mne: Any, path: Path) -> Any:
    lowered = path.name.lower()
    if lowered.endswith(".edf"):
        reader = mne.io.read_raw_edf
    elif lowered.endswith(".bdf"):
        reader = mne.io.read_raw_bdf
    elif lowered.endswith(".vhdr"):
        reader = mne.io.read_raw_brainvision
    elif lowered.endswith(".set"):
        reader = mne.io.read_raw_eeglab
    elif lowered.endswith(".fif") or lowered.endswith(".fif.gz"):
        reader = mne.io.read_raw_fif
    elif lowered.endswith(".gdf"):
        reader = mne.io.read_raw_gdf
    elif lowered.endswith(".cnt"):
        reader = mne.io.read_raw_cnt
    elif lowered.endswith(".mff"):
        reader = mne.io.read_raw_egi
    else:
        raise DatasetProfileError(f"No installed MNE raw reader for: {path}")
    try:
        return reader(path, preload=False, verbose="ERROR")
    except Exception as exc:
        raise DatasetProfileError(f"MNE could not inspect {path}: {exc}") from exc


def _assert_mne_profile_matches(profile: dict[str, Any], observed: dict[str, Any]) -> None:
    expected = {
        "sampling_frequency_hz": float(profile["signal"]["sampling_frequency_hz"]),
        "channel_names": profile["signal"]["channel_names"],
        "channel_count": int(profile["signal"]["channel_count"]),
        "eog_channel_count": int(profile["signal"].get("eog_channel_count", 0)),
        "class_counts": profile["events"]["class_counts"],
        "event_total": int(profile["events"]["total"]),
        "subjects": int(profile["volume"]["subjects"]),
        "sessions_per_subject": int(profile["sessions"]["sessions_per_subject"]),
        "session_indices": profile["sessions"]["session_indices"],
        "runs": int(profile["volume"]["runs"]),
    }
    for field, value in expected.items():
        if observed[field] != value:
            raise DatasetProfileError(
                f"MNE observation disagrees with semantic profile at {field}: "
                f"observed={observed[field]!r}, declared={value!r}"
            )
    declared_line = profile["signal"].get("power_line_frequency_hz")
    if observed["power_line_frequency_hz"] is not None:
        if declared_line is None or float(declared_line) != observed[
            "power_line_frequency_hz"
        ]:
            raise DatasetProfileError("MNE line frequency disagrees with semantic profile")


def _size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _source_dependencies(path: Path) -> list[Path]:
    if path.suffix.lower() == ".vhdr":
        return [
            companion
            for companion in (path.with_suffix(".vmrk"), path.with_suffix(".eeg"))
            if companion.is_file()
        ]
    if path.suffix.lower() == ".set" and path.with_suffix(".fdt").is_file():
        return [path.with_suffix(".fdt")]
    return []


def _source_sha256(path: Path) -> str:
    dependencies = _source_dependencies(path)
    if not dependencies:
        return sha256_path(path)
    digest = hashlib.sha256()
    for item in [path, *dependencies]:
        digest.update(item.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_path(item)))
    return digest.hexdigest()


def _source_size_bytes(path: Path) -> int:
    return _size_bytes(path) + sum(_size_bytes(item) for item in _source_dependencies(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an explicit semantic sidecar and bind it to local EEG sources."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--semantics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.dataset_root.expanduser().resolve()
    semantics_path = (
        args.semantics.expanduser().resolve()
        if args.semantics
        else root / SEMANTICS_FILENAME
    )
    validation = validate_semantic_dataset(
        dataset_root=root, semantics_path=semantics_path, output_path=args.output
    )
    print(f"status: validated")
    print(f"adapter_kind: {validation['adapter_kind']}")
    print(f"sources: {validation['summary']['source_count']}")
    print(f"output: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
