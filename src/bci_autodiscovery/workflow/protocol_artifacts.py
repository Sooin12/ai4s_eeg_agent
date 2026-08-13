"""Versioned protocol artifacts and the deterministic human-approval boundary."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ProtocolArtifactError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolArtifactError(f"Cannot load protocol artifact {path}: {exc}") from exc


def atomic_json(path: Path, value: object, *, refuse_overwrite: bool = False) -> None:
    if refuse_overwrite and path.exists():
        raise ProtocolArtifactError(f"Refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    if not safe:
        raise ProtocolArtifactError("Dataset id cannot be converted to a safe path")
    return safe


class ProtocolArtifactRegistry:
    """Preserve immutable revisions and expose one verified approved contract."""

    def __init__(self, *, root: Path, dataset_id: str) -> None:
        self.root = Path(root).expanduser().resolve() / _safe_id(dataset_id)
        self.dataset_id = dataset_id
        self.index_path = self.root / "index.json"
        if self.index_path.exists():
            self._index = load_json(self.index_path)
            if self._index.get("dataset_id") != dataset_id:
                raise ProtocolArtifactError("Protocol registry belongs to another dataset")
        else:
            self._index = {
                "schema_version": "1.0",
                "dataset_id": dataset_id,
                "revisions": [],
                "review_turns": [],
                "approvals": [],
                "current_approved": None,
            }

    def register_revision(
        self,
        *,
        source_path: Path,
        kind: str,
        parent_sha256: str | None = None,
    ) -> dict[str, Any]:
        source = Path(source_path).expanduser().resolve()
        value = load_json(source)
        if value.get("dataset_id") != self.dataset_id:
            raise ProtocolArtifactError("Protocol revision belongs to another dataset")
        digest = sha256_file(source)
        existing = next(
            (item for item in self._index["revisions"] if item["sha256"] == digest),
            None,
        )
        if existing:
            return existing
        revision_number = len(self._index["revisions"]) + 1
        artifact_id = f"revision-{revision_number:04d}-{digest[:12]}"
        destination = self.root / "revisions" / f"{artifact_id}.json"
        atomic_json(destination, value, refuse_overwrite=True)
        record = {
            "artifact_id": artifact_id,
            "kind": kind,
            "revision_number": revision_number,
            "path": str(destination),
            "sha256": sha256_file(destination),
            "source_path": str(source),
            "source_sha256": digest,
            "parent_sha256": parent_sha256,
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._index["revisions"].append(record)
        self._write_index()
        return record

    def register_review_turn(self, *, review_path: Path) -> dict[str, Any]:
        source = Path(review_path).expanduser().resolve()
        value = load_json(source)
        if value.get("dataset_id") != self.dataset_id:
            raise ProtocolArtifactError("Protocol review belongs to another dataset")
        digest = sha256_file(source)
        existing = next(
            (item for item in self._index["review_turns"] if item["sha256"] == digest),
            None,
        )
        if existing:
            return existing
        number = len(self._index["review_turns"]) + 1
        artifact_id = f"review-{number:04d}-{digest[:12]}"
        destination = self.root / "reviews" / f"{artifact_id}.json"
        atomic_json(destination, value, refuse_overwrite=True)
        record = {
            "artifact_id": artifact_id,
            "path": str(destination),
            "sha256": sha256_file(destination),
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._index["review_turns"].append(record)
        self._write_index()
        return record

    def approve(
        self,
        *,
        proposal_path: Path,
        approved_by: str,
        decision_note: str,
    ) -> dict[str, Any]:
        if not approved_by.strip():
            raise ProtocolArtifactError("approved_by cannot be empty")
        source = Path(proposal_path).expanduser().resolve()
        proposal = load_json(source)
        if proposal.get("dataset_id") != self.dataset_id:
            raise ProtocolArtifactError("Protocol proposal belongs to another dataset")
        if proposal.get("status") != "proposed_requires_human_approval":
            raise ProtocolArtifactError("Only a pending proposal can be approved")
        source_profile = proposal.get("source_profile") or {}
        profile_path_text = source_profile.get("path")
        expected_profile_hash = source_profile.get("sha256")
        if not profile_path_text or not expected_profile_hash:
            raise ProtocolArtifactError("Proposal lacks its authoritative source_profile binding")
        profile_path = Path(profile_path_text)
        if not profile_path.is_file() or sha256_file(profile_path) != expected_profile_hash:
            raise ProtocolArtifactError("Proposal source profile failed integrity check")
        from bci_autodiscovery.agents.protocol_planner import (
            _load_profile,
            validate_protocol_proposal,
        )

        profile = _load_profile(profile_path)
        validate_protocol_proposal(
            proposal,
            profile=profile,
            dataset_id=self.dataset_id,
        )
        source_digest = sha256_file(source)
        revision = self.register_revision(
            source_path=source,
            kind="approval_source",
            parent_sha256=(proposal.get("revision") or {}).get("parent_sha256"),
        )
        approved_at = datetime.now(timezone.utc).isoformat()
        approved = dict(proposal)
        approved["status"] = "approved"
        approved["activation_performed"] = True
        approved["approved_at_utc"] = approved_at
        approved["approval"] = {
            "approved_by": approved_by.strip(),
            "decision_note": decision_note.strip(),
            "source_proposal_path": str(source),
            "source_proposal_sha256": source_digest,
            "registry_revision_id": revision["artifact_id"],
        }
        if approved.get("split_unit") == "session":
            approved["session_roles"] = approved["data_roles"]
        approval_id = f"approval-{len(self._index['approvals']) + 1:04d}-{source_digest[:12]}"
        destination = self.root / "approvals" / f"{approval_id}.json"
        atomic_json(destination, approved, refuse_overwrite=True)
        record = {
            "artifact_id": approval_id,
            "path": str(destination),
            "sha256": sha256_file(destination),
            "approved_at_utc": approved_at,
            "approved_by": approved_by.strip(),
            "source_revision_id": revision["artifact_id"],
        }
        self._index["approvals"].append(record)
        self._index["current_approved"] = record
        self._write_index()
        return approved

    def resolve_current_approved(self) -> Path:
        record = self._index.get("current_approved")
        if not record:
            raise ProtocolArtifactError(
                f"Dataset {self.dataset_id!r} has no human-approved protocol"
            )
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ProtocolArtifactError("Current approved protocol failed integrity check")
        value = load_json(path)
        if value.get("status") != "approved":
            raise ProtocolArtifactError("Current protocol is not approved")
        return path

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._index))

    def _write_index(self) -> None:
        atomic_json(self.index_path, self._index)
