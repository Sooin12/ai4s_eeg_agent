"""Publish a provenance-bound, user-facing view of immutable Agent artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


STAGE_DIRECTORIES = {
    "dataset_understanding": "01_dataset_understanding",
    "research_design": "02_research_design",
    "method_capabilities": "03_method_capabilities",
    "subject_profiles": "04_subject_profiles",
    "pipeline_search": "05_pipeline_search",
    "confirmation": "06_confirmation",
    "reports": "07_reports",
}


class AgentOutputError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_identifier(value: str, *, field: str) -> str:
    normalized = value.strip().lower().replace(" ", "_")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,127}", normalized):
        raise AgentOutputError(f"Unsafe {field}: {value!r}")
    return normalized


class AgentOutputPublisher:
    """Maintain a convenient view while keeping artifacts/runs authoritative."""

    def __init__(self, output_root: Path = Path("agent_outputs")) -> None:
        self.output_root = Path(output_root).expanduser().resolve()

    def publish(
        self,
        *,
        dataset_id: str,
        stage: str,
        artifact_name: str,
        source_path: Path,
        run_id: str,
    ) -> dict[str, Any]:
        if stage not in STAGE_DIRECTORIES:
            raise AgentOutputError(f"Unknown Agent output stage: {stage}")
        safe_dataset = _safe_identifier(dataset_id, field="dataset_id")
        safe_name = _safe_identifier(artifact_name, field="artifact_name")
        source = Path(source_path).expanduser().resolve()
        if not source.is_file():
            raise AgentOutputError(f"Source artifact is unavailable: {source}")
        suffix = "".join(source.suffixes) or ".bin"
        dataset_root = self.output_root / safe_dataset
        stage_root = dataset_root / STAGE_DIRECTORIES[stage]
        stage_root.mkdir(parents=True, exist_ok=True)
        destination = stage_root / f"{safe_name}{suffix}"
        _atomic_copy(source, destination)
        source_hash = _sha256(source)
        published_hash = _sha256(destination)
        if published_hash != source_hash:
            raise AgentOutputError("Published Agent output failed SHA-256 verification")
        record = {
            "artifact_name": safe_name,
            "stage": stage,
            "run_id": run_id,
            "source_path": str(source),
            "source_sha256": source_hash,
            "published_path": str(destination.resolve()),
            "published_sha256": published_hash,
            "published_at": datetime.now(timezone.utc).isoformat(),
            "authoritative": False,
        }
        index_path = dataset_root / "index.json"
        index = _load_index(index_path, dataset_id=safe_dataset)
        key = f"{stage}/{safe_name}"
        index["artifacts"][key] = record
        _atomic_json(index_path, index)
        _write_dataset_readme(dataset_root, safe_dataset)
        return record

    def publish_many(
        self,
        *,
        dataset_id: str,
        run_id: str,
        artifacts: Iterable[tuple[str, str, Path]],
    ) -> list[dict[str, Any]]:
        return [
            self.publish(
                dataset_id=dataset_id,
                stage=stage,
                artifact_name=name,
                source_path=path,
                run_id=run_id,
            )
            for stage, name, path in artifacts
        ]


def _load_index(path: Path, *, dataset_id: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": "1.0",
            "dataset_id": dataset_id,
            "view_kind": "derived_user_view",
            "authoritative_store": "artifacts/runs",
            "artifacts": {},
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentOutputError(f"Cannot read Agent output index: {exc}") from exc
    if value.get("dataset_id") != dataset_id or not isinstance(value.get("artifacts"), dict):
        raise AgentOutputError("Agent output index identity or schema is invalid")
    return value


def _atomic_copy(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_dataset_readme(dataset_root: Path, dataset_id: str) -> None:
    lines = [
        f"# {dataset_id} Agent 输出",
        "",
        "此目录是面向用户的派生视图；不可变审计原件仍在 `artifacts/runs/`。",
        "每个文件的来源路径、运行 ID 和 SHA-256 见 `index.json`。",
        "",
    ]
    for stage_name in STAGE_DIRECTORIES.values():
        (dataset_root / stage_name).mkdir(exist_ok=True)
        lines.append(f"- `{stage_name}/`")
    (dataset_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

