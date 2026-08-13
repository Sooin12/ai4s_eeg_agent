"""Deterministic dataset-level profiling used by the Dataset Profiler Agent."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


class DatasetProfileError(RuntimeError):
    pass


DATASET_PROFILE_BINDING_ROOTS = (
    "dataset",
    "paradigm",
    "resting_state",
    "signal",
    "equipment",
    "events",
    "sessions",
    "volume",
    "quality",
    "constraints",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetProfileError(f"Cannot read JSON {path}: {exc}") from exc


def _read_tsv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    except OSError as exc:
        raise DatasetProfileError(f"Cannot read TSV {path}: {exc}") from exc


def _one(values: set[Any], field: str) -> Any:
    if len(values) != 1:
        raise DatasetProfileError(f"Expected one value for {field}; observed {sorted(values)!r}")
    return next(iter(values))


def _one_json(values: list[Any], field: str) -> Any:
    encoded = {json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values}
    if len(encoded) != 1:
        raise DatasetProfileError(f"Expected one value for {field}; observed disagreement")
    return values[0]


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "n/a", "unknown"}:
        return None
    return float(value)


def _entity_value(path: Path, prefix: str) -> str | None:
    part = next((part for part in path.parts if part.startswith(prefix)), None)
    return part.split("-", 1)[1] if part else None


def _sortable_label(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _parse_identity(path: Path) -> tuple[str, str]:
    subject_part = next((part for part in path.parts if part.startswith("sub-")), None)
    session_part = next((part for part in path.parts if part.startswith("ses-")), None)
    if not subject_part:
        raise DatasetProfileError(f"Cannot parse subject from {path}")
    return subject_part.split("-", 1)[1], (
        session_part.split("-", 1)[1] if session_part else "default"
    )


def _bids_eeg_files(root: Path, suffix: str) -> list[Path]:
    return sorted(
        path
        for path in root.rglob(f"*{suffix}")
        if path.is_file() and "eeg" in path.parts and _entity_value(path, "sub-")
    )


def _bids_signal_files(root: Path) -> list[Path]:
    extensions = (".edf", ".bdf", ".vhdr", ".set")
    return sorted(
        path
        for path in root.rglob("*_eeg.*")
        if path.is_file()
        and "eeg" in path.parts
        and path.suffix.lower() in extensions
        and _entity_value(path, "sub-")
    )


def _validate_bids_signal_companions(signal_files: list[Path]) -> None:
    for path in signal_files:
        if path.suffix.lower() == ".vhdr" and not all(
            path.with_suffix(extension).is_file() for extension in (".vmrk", ".eeg")
        ):
            raise DatasetProfileError(
                f"BrainVision recording lacks required VMRK/EEG companions: {path}"
            )


def _paradigm_family(*, task_name: str, semantic_text: str, keywords: set[str]) -> str:
    if "motor imagery" in keywords or "motor imagery" in semantic_text:
        return "motor_imagery"
    normalized = "_".join(
        part for part in "".join(
            character.lower() if character.isalnum() else " " for character in task_name
        ).split()
    )
    return f"bids_task_{normalized or 'unspecified'}"


def profile_bids_eeg_dataset(
    *,
    dataset_root: str,
    validation_path: str,
    dataset_id: str,
    common_analysis_window_s: list[float] | None = None,
) -> dict[str, Any]:
    """Build a normalized profile from BIDS sidecars plus audited validation.

    No signal samples are sent to or interpreted by a language model. The expensive
    signal acceptance run is consumed as a hashed, read-only evidence artifact.
    """

    root = Path(dataset_root).expanduser().resolve()
    validation_file = Path(validation_path).expanduser().resolve()
    if not root.is_dir():
        raise DatasetProfileError(f"Dataset root does not exist: {root}")
    if not validation_file.is_file():
        raise DatasetProfileError(f"Validation artifact does not exist: {validation_file}")

    validation = _read_json(validation_file)
    expected_root = Path(validation.get("dataset_root", "")).expanduser().resolve()
    if expected_root != root:
        raise DatasetProfileError(
            f"Validation artifact belongs to {expected_root}, not requested root {root}"
        )
    core = validation.get("validation") or {}
    if int(core.get("invalid_runs", -1)) != 0:
        raise DatasetProfileError("Dataset has failed core validation runs")

    description_file = root / "dataset_description.json"
    description = _read_json(description_file)
    event_files = _bids_eeg_files(root, "_events.tsv")
    channel_files = _bids_eeg_files(root, "_channels.tsv")
    eeg_json_files = _bids_eeg_files(root, "_eeg.json")
    signal_files = _bids_signal_files(root)
    if not event_files or not channel_files or not eeg_json_files or not signal_files:
        raise DatasetProfileError("Required BIDS EEG sidecars or signal files are missing")
    if len({len(event_files), len(channel_files), len(eeg_json_files), len(signal_files)}) != 1:
        raise DatasetProfileError(
            "BIDS run sidecar counts disagree: "
            f"events={len(event_files)}, channels={len(channel_files)}, "
            f"eeg_json={len(eeg_json_files)}, signal={len(signal_files)}"
        )
    _validate_bids_signal_companions(signal_files)

    event_counts: Counter[str] = Counter()
    event_values: dict[str, set[str]] = defaultdict(set)
    event_durations: set[float] = set()
    subjects: set[str] = set()
    sessions_by_subject: dict[str, set[str]] = defaultdict(set)
    runs_by_session: Counter[tuple[str, str]] = Counter()
    for path in event_files:
        subject, session = _parse_identity(path)
        subjects.add(subject)
        sessions_by_subject[subject].add(session)
        runs_by_session[(subject, session)] += 1
        for row in _read_tsv(path):
            label = row.get("trial_type", "")
            event_counts[label] += 1
            event_values[label].add(row.get("value", ""))
            event_durations.add(float(row["duration"]))

    channel_signatures: set[tuple[tuple[str, str, str], ...]] = set()
    channel_statuses: Counter[str] = Counter()
    for path in channel_files:
        rows = _read_tsv(path)
        channel_signatures.add(
            tuple((row["name"], row["type"], row["units"]) for row in rows)
        )
        channel_statuses.update(row.get("status", "unknown") for row in rows)
    channel_signature = _one(channel_signatures, "channel signature")

    eeg_metadata = [_read_json(path) for path in eeg_json_files]
    metadata_fields = {
        name: [item.get(name) for item in eeg_metadata]
        for name in (
            "TaskName",
            "Manufacturer",
            "ManufacturersModelName",
            "CapManufacturer",
            "CapManufacturersModelName",
            "SamplingFrequency",
            "PowerLineFrequency",
            "RecordingType",
            "EEGReference",
            "EEGGround",
            "EEGPlacementScheme",
            "EEGChannelCount",
            "EOGChannelCount",
            "SoftwareFilters",
            "HardwareFilters",
            "TaskDescription",
            "Instructions",
        )
    }
    stable_metadata = {
        name: _one_json(values, name) for name, values in metadata_fields.items()
    }

    sessions_per_subject = {len(value) for value in sessions_by_subject.values()}
    runs_per_session_index: dict[str, int] = {}
    session_labels = sorted({item[1] for item in runs_by_session}, key=_sortable_label)
    for session in session_labels:
        counts = {
            runs_by_session[(subject, session)]
            for subject in subjects
            if (subject, session) in runs_by_session
        }
        runs_per_session_index[str(session)] = int(_one(counts, f"runs in session {session}"))

    labels = sorted(label for label in event_counts if label)
    task_description = str(stable_metadata["TaskDescription"] or "")
    instructions = str(stable_metadata["Instructions"] or "")
    semantic_text = f"{task_description} {instructions}".lower()
    keywords = {str(item).lower() for item in description.get("Keywords", [])}
    paradigm_family = _paradigm_family(
        task_name=str(stable_metadata["TaskName"] or ""),
        semantic_text=semantic_text,
        keywords=keywords,
    )
    rest_markers = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and "rest" in path.name.lower()
    ]
    actual_eog_count = int(stable_metadata["EOGChannelCount"] or 0)
    if stable_metadata["EEGChannelCount"] is not None and int(
        stable_metadata["EEGChannelCount"]
    ) != len(channel_signature):
        raise DatasetProfileError("EEG metadata channel count disagrees with channels.tsv")
    if validation.get("run_count") is not None and int(validation["run_count"]) != len(signal_files):
        raise DatasetProfileError("Validation run count disagrees with discovered BIDS signal files")
    if core.get("total_trials") is not None and int(core["total_trials"]) != sum(
        event_counts.values()
    ):
        raise DatasetProfileError("Validation trial count disagrees with event sidecars")
    if validation.get("subject_count") is not None and int(
        validation["subject_count"]
    ) != len(subjects):
        raise DatasetProfileError("Validation subject count disagrees with BIDS hierarchy")
    quality_review = validation.get("quality_review") or {}
    diagnostic_review = validation.get("diagnostic_filtered_review") or {}

    evidence = [
        {
            "claim": "dataset identity and provenance",
            "source": str(description_file.resolve()),
            "sha256": _sha256(description_file),
            "method": "BIDS dataset_description.json",
        },
        {
            "claim": "actions, event counts, durations, subjects, sessions, and runs",
            "source": "sub-*/[ses-*/]eeg/*_events.tsv",
            "files_read": len(event_files),
            "method": "complete sidecar enumeration",
        },
        {
            "claim": "channel names, types, units, and declared status",
            "source": "sub-*/[ses-*/]eeg/*_channels.tsv",
            "files_read": len(channel_files),
            "method": "complete sidecar signature agreement",
        },
        {
            "claim": "equipment, sampling, reference, ground, and auxiliary channels",
            "source": "sub-*/[ses-*/]eeg/*_eeg.json",
            "files_read": len(eeg_json_files),
            "method": "complete sidecar field agreement",
        },
        {
            "claim": "signal integrity and usable common epoch boundary",
            "source": str(validation_file),
            "sha256": _sha256(validation_file),
            "method": "previous full read-only signal acceptance run",
        },
    ]

    if common_analysis_window_s is None:
        hinted_window = validation.get("common_analysis_window_s")
        if hinted_window is not None:
            common_analysis_window_s = [float(value) for value in hinted_window]
        elif int(core.get("truncated_event_runs", 0)) > 0:
            raise DatasetProfileError(
                "Validation reports truncated events but does not publish a safe common_analysis_window_s"
            )
        else:
            common_analysis_window_s = [0.0, min(event_durations)]

    profile: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset": {
            "id": dataset_id,
            "name": description.get("Name"),
            "version": description.get("Version"),
            "bids_version": description.get("BIDSVersion"),
            "dataset_type": description.get("DatasetType"),
            "license": description.get("License"),
            "format": "BIDS EEG with " + ", ".join(
                sorted({path.suffix.lower().lstrip(".") for path in signal_files})
            ),
            "doi": description.get("DatasetDOI"),
        },
        "paradigm": {
            "family": paradigm_family,
            "task_name": stable_metadata["TaskName"],
            "actions": [
                {
                    "label": label,
                    "values": sorted(event_values[label]),
                    "trial_count": event_counts[label],
                }
                for label in labels
            ],
            "cue_based": "cue" in semantic_text,
            "feedback_present": "feedback" in semantic_text,
            "feedback_note": task_description,
        },
        "resting_state": {
            "present": bool(rest_markers),
            "matching_files": rest_markers,
            "interpretation": (
                "Rest-like files were discovered and require semantic review."
                if rest_markers
                else "No independent resting-state recording was discovered."
            ),
        },
        "signal": {
            "modalities": sorted({item[1] for item in channel_signature}),
            "sampling_frequency_hz": float(stable_metadata["SamplingFrequency"]),
            "power_line_frequency_hz": _optional_float(
                stable_metadata["PowerLineFrequency"]
            ),
            "recording_type": stable_metadata["RecordingType"],
            "channel_count": len(channel_signature),
            "channel_names": [item[0] for item in channel_signature],
            "channel_units": sorted({item[2] for item in channel_signature}),
            "declared_channel_status_counts": dict(sorted(channel_statuses.items())),
            "reference": stable_metadata["EEGReference"],
            "ground": stable_metadata["EEGGround"],
            "placement_scheme": stable_metadata["EEGPlacementScheme"],
            "eog_channel_count": actual_eog_count,
            "software_filters": stable_metadata["SoftwareFilters"],
            "hardware_filters": stable_metadata["HardwareFilters"],
        },
        "equipment": {
            "manufacturer": stable_metadata["Manufacturer"],
            "model": stable_metadata["ManufacturersModelName"],
            "cap_manufacturer": stable_metadata["CapManufacturer"],
            "cap_model": stable_metadata["CapManufacturersModelName"],
        },
        "events": {
            "labels": labels,
            "nominal_durations_s": sorted(event_durations),
            "total": sum(event_counts.values()),
            "class_counts": dict(sorted(event_counts.items())),
            "common_analysis_window_s": common_analysis_window_s,
            "common_window_reason": (
                f"The safe common window {common_analysis_window_s} is supported by the "
                "validation artifact or adapter evidence."
            ),
        },
        "sessions": {
            "subjects": len(subjects),
            "sessions_per_subject": int(_one(sessions_per_subject, "sessions per subject")),
            "total_subject_sessions": sum(len(value) for value in sessions_by_subject.values()),
            "session_indices": session_labels,
            "runs_per_session": runs_per_session_index,
            "longitudinal": any(len(value) > 1 for value in sessions_by_subject.values()),
            "separate_days_declared": "separate days" in semantic_text,
        },
        "volume": {
            "subjects": int(validation.get("subject_count", len(subjects))),
            "subject_sessions": int(validation.get("session_count", 0)),
            "runs": len(signal_files),
            "trials": sum(event_counts.values()),
            "signal_bytes": sum(path.stat().st_size for path in signal_files),
        },
        "quality": {
            "core_valid_runs": int(core.get("valid_runs", 0)),
            "core_invalid_runs": int(core.get("invalid_runs", 0)),
            "nonfinite_runs": int(core.get("nonfinite_runs", 0)),
            "flat_channel_runs": int(core.get("flat_channel_runs", 0)),
            "truncated_event_runs": int(core.get("truncated_event_runs", 0)),
            "raw_review_candidates": int(quality_review.get("candidate_count", 0)),
            "diagnostic_review_candidates": int(diagnostic_review.get("candidate_count", 0)),
            "isolated_diagnostic_transient_runs": int(
                core.get("isolated_diagnostic_transient_runs", 0)
            ),
            "automatic_exclusions": 0,
        },
        "constraints": {
            "allowed": [
                f"EEG processing for the evidence-established paradigm {paradigm_family}",
                "classification or analysis over the event labels recorded in events.tsv",
                "trial-level quality control with an auditable exclusion rule",
                f"search windows fully contained in {common_analysis_window_s}",
                "cross-session stability and drift analysis when repeated sessions exist",
            ],
            "forbidden": [
                "claims based on auxiliary channels that are not present",
                "independent resting-state claims when no resting-state recording is evidenced",
                "claims based on channels absent from channels.tsv",
                "silent exclusion of subjects, runs, trials, or channels",
                "treating any future frozen-confirmation role as search data",
            ],
            "requires_research_design_decision": [
                "search versus frozen-confirmation session split",
                "trial/channel exclusion thresholds",
                "legal pipeline portfolio and experiment budget",
            ],
            "external_authority_blockers": [],
        },
        "evidence": evidence,
    }
    validate_dataset_profile(profile)
    return profile


def validate_dataset_profile(profile: dict[str, Any]) -> None:
    """Check the minimum machine contract required by downstream agents."""

    if profile.get("schema_version") != "1.0":
        raise DatasetProfileError("Unsupported DatasetProfile schema_version")
    required_sections = {
        "dataset",
        "paradigm",
        "resting_state",
        "signal",
        "equipment",
        "events",
        "sessions",
        "volume",
        "quality",
        "constraints",
        "evidence",
    }
    missing = sorted(required_sections.difference(profile))
    if missing:
        raise DatasetProfileError(f"Dataset profile is missing sections: {missing}")
    dataset_id = profile["dataset"].get("id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise DatasetProfileError("Dataset profile contains no stable dataset ID")
    paradigm = profile["paradigm"]
    if not isinstance(paradigm.get("family"), str) or not paradigm["family"].strip():
        raise DatasetProfileError("Dataset profile contains no paradigm family")
    actions = paradigm.get("actions")
    if not isinstance(actions, list) or not actions:
        raise DatasetProfileError("Dataset profile contains no actions")
    labels = [str(item.get("label") or "").strip() for item in actions if isinstance(item, dict)]
    if len(labels) != len(actions) or any(not label for label in labels):
        raise DatasetProfileError("Every DatasetProfile action requires a label")
    if len(labels) != len(set(labels)):
        raise DatasetProfileError("DatasetProfile action labels must be unique")
    signal = profile["signal"]
    if int(signal.get("channel_count", 0)) <= 0:
        raise DatasetProfileError("Dataset profile contains no channels")
    if float(signal.get("sampling_frequency_hz", 0)) <= 0:
        raise DatasetProfileError("Dataset profile has invalid sampling frequency")
    if int(profile["volume"].get("trials", 0)) <= 0:
        raise DatasetProfileError("Dataset profile contains no trials")
    events = profile["events"]
    bounds = events.get("common_analysis_window_s")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or not float(bounds[0]) < float(bounds[1])
    ):
        raise DatasetProfileError("Dataset profile has no valid common analysis window")
    sessions = profile["sessions"]
    if int(sessions.get("sessions_per_subject", 0)) <= 0:
        raise DatasetProfileError("Dataset profile contains no per-subject sessions")
    indices = sessions.get("session_indices")
    if not isinstance(indices, list) or not indices or len(indices) != len(set(indices)):
        raise DatasetProfileError("Dataset profile session indices must be non-empty and unique")
    constraints = profile["constraints"]
    if not isinstance(constraints, dict):
        raise DatasetProfileError("Dataset profile constraints must be an object")
    for field in ("allowed", "forbidden"):
        values = constraints.get(field)
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise DatasetProfileError(f"Dataset profile constraints.{field} must be text array")
    design = constraints.get("requires_research_design_decision")
    legacy = constraints.get("requires_human_decision")
    if design is None and legacy is None:
        raise DatasetProfileError(
            "Dataset profile must distinguish deferred research-design decisions"
        )
    if design is not None and (
        not isinstance(design, list) or any(not isinstance(item, str) for item in design)
    ):
        raise DatasetProfileError(
            "constraints.requires_research_design_decision must be a text array"
        )
    blockers = constraints.get("external_authority_blockers", [])
    if not isinstance(blockers, list) or any(not isinstance(item, str) for item in blockers):
        raise DatasetProfileError("constraints.external_authority_blockers must be a text array")
    evidence = profile["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise DatasetProfileError("Dataset profile contains no provenance evidence")
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("source"), str)
        or not item["source"].strip()
        or (
            item.get("claim") is not None
            and not isinstance(item.get("claim"), str)
        )
        for item in evidence
    ):
        raise DatasetProfileError(
            "Every DatasetProfile evidence item needs a source and any claim must be text"
        )


def validate_dataset_profile_provenance(
    profile: dict[str, Any],
    *,
    require_hashed_evidence: bool = False,
    require_current_constraints: bool = False,
) -> None:
    """Verify every evidence artifact that publishes a content hash.

    Pattern-based evidence entries intentionally have no hash and remain descriptive. Actual
    generated validation/manifest artifacts are hash-bound and must still exist unchanged.
    """

    validate_dataset_profile(profile)
    if require_current_constraints:
        constraints = profile["constraints"]
        if "requires_human_decision" in constraints:
            raise DatasetProfileError(
                "Legacy constraints.requires_human_decision is not valid for a formal run"
            )
        for field in (
            "requires_research_design_decision",
            "external_authority_blockers",
        ):
            values = constraints.get(field)
            if not isinstance(values, list) or any(
                not isinstance(item, str) for item in values
            ):
                raise DatasetProfileError(
                    f"Formal DatasetProfile requires constraints.{field} as a text array"
                )
    hashed_count = 0
    for item in profile["evidence"]:
        expected = item.get("sha256")
        if expected is None:
            continue
        hashed_count += 1
        if not isinstance(expected, str) or len(expected) != 64:
            raise DatasetProfileError("DatasetProfile evidence contains an invalid SHA-256")
        source = Path(item["source"]).expanduser()
        if not source.is_absolute() or not source.is_file():
            raise DatasetProfileError(
                f"Hashed DatasetProfile evidence is unavailable: {source}"
            )
        if _sha256(source.resolve()) != expected:
            raise DatasetProfileError(
                f"DatasetProfile evidence failed integrity verification: {source}"
            )
    if require_hashed_evidence and hashed_count == 0:
        raise DatasetProfileError(
            "DatasetProfile has no hash-bound provenance artifact"
        )


def dataset_profile_field_catalog(profile: dict[str, Any]) -> dict[str, Any]:
    """Return stable leaf paths that another contract may cite exactly."""

    validate_dataset_profile(profile)
    catalog: dict[str, Any] = {}

    def visit(path: str, value: Any) -> None:
        if isinstance(value, dict) and value:
            for key, nested in value.items():
                visit(f"{path}.{key}", nested)
            return
        catalog[path] = value

    for root in DATASET_PROFILE_BINDING_ROOTS:
        visit(root, profile[root])
    return dict(sorted(catalog.items()))
