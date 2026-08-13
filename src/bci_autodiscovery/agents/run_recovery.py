"""Fail-closed guards for resuming interrupted immutable Agent runs."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from bci_autodiscovery.workflow.protocol_artifacts import atomic_json


class AgentRecoveryError(RuntimeError):
    pass


def write_process_state(
    run_dir: Path, *, run_id: str, status: str, error: str | None = None
) -> None:
    atomic_json(
        Path(run_dir) / "run_process.json",
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "pid": os.getpid(),
            "status": status,
            "error": error,
            "updated_at_unix": time.time(),
        },
    )


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _source_run_dir(draft_path: Path) -> Path | None:
    resolved = Path(draft_path).expanduser().resolve()
    for parent in resolved.parents:
        if parent.name == "cycles":
            return parent.parent
    return None


def assert_source_run_recoverable(
    draft_path: Path, *, audit_quiescence_seconds: float = 120.0
) -> None:
    """Refuse recovery while the source is active or already terminal."""

    run_dir = _source_run_dir(draft_path)
    if run_dir is None:
        return
    result_path = run_dir / "dataset_level_run.json"
    if result_path.is_file():
        try:
            result: dict[str, Any] = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AgentRecoveryError(
                f"Source run result cannot be verified: {result_path}: {exc}"
            ) from exc
        if result.get("status") in {
            "completed",
            "rejected",
            "revision_limit_reached",
        }:
            raise AgentRecoveryError(
                f"Source Dataset-Level run is already terminal ({result.get('status')}): "
                f"{run_dir}"
            )

    process_path = run_dir / "run_process.json"
    if process_path.is_file():
        try:
            state = json.loads(process_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AgentRecoveryError(
                f"Source process state cannot be verified: {process_path}: {exc}"
            ) from exc
        if state.get("status") == "running" and _process_alive(
            int(state.get("pid") or 0)
        ):
            raise AgentRecoveryError(
                f"Source Dataset-Level process is still alive (pid={state.get('pid')}): "
                f"{run_dir}"
            )
        return

    audit_path = run_dir / "audit.jsonl"
    if audit_path.is_file():
        age = time.time() - audit_path.stat().st_mtime
        if age < audit_quiescence_seconds:
            raise AgentRecoveryError(
                "Source Dataset-Level audit is still changing or too recent for safe "
                f"recovery ({age:.1f}s old): {audit_path}"
            )


def assert_research_design_run_recoverable(
    run_dir: Path, *, audit_quiescence_seconds: float = 120.0
) -> dict[str, Any]:
    """Validate that an interrupted Research Design run can safely append recovery events."""

    resolved = Path(run_dir).expanduser().resolve()
    state_path = resolved / "research_design_state.json"
    if not state_path.is_file():
        raise AgentRecoveryError(f"Research Design state does not exist: {state_path}")
    try:
        state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentRecoveryError(f"Research Design state cannot be verified: {exc}") from exc
    if state.get("status") in {
        "completed",
        "rejected",
        "revision_limit_reached",
    }:
        raise AgentRecoveryError(
            f"Research Design run is already terminal ({state.get('status')}): {resolved}"
        )
    if (resolved / "research_design_run.json").exists():
        raise AgentRecoveryError("Research Design terminal manifest already exists")

    process_path = resolved / "run_process.json"
    if process_path.is_file():
        try:
            process = json.loads(process_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AgentRecoveryError(
                f"Research Design process state cannot be verified: {exc}"
            ) from exc
        if process.get("status") == "running" and _process_alive(
            int(process.get("pid") or 0)
        ):
            raise AgentRecoveryError(
                "Research Design process is still alive "
                f"(pid={process.get('pid')}): {resolved}"
            )
        return state

    audit_path = resolved / "audit.jsonl"
    if audit_path.is_file():
        age = time.time() - audit_path.stat().st_mtime
        if age < audit_quiescence_seconds:
            raise AgentRecoveryError(
                "Research Design audit is too recent for safe recovery "
                f"({age:.1f}s old): {audit_path}"
            )
    return state
