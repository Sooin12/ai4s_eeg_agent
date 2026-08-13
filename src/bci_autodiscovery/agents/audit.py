"""Append-only JSONL audit traces for research-agent runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


class AuditSink(Protocol):
    def record(self, event_type: str, payload: dict[str, Any]) -> None: ...


class NullAuditSink:
    def record(self, event_type: str, payload: dict[str, Any]) -> None:
        del event_type, payload


class MemoryAuditSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append({"event_type": event_type, "payload": payload})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact(value: Any) -> Any:
    """Redact credential-shaped fields before serializing an audit event."""

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            normalized = key.lower().replace("-", "_")
            if normalized in {
                "api_key",
                "apikey",
                "authorization",
                "access_token",
                "secret",
                "password",
            }:
                result[key] = "<redacted>"
            else:
                result[key] = _redact(nested)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


class JsonlAuditSink:
    """Durable append-only sink; each line is independently parseable."""

    def __init__(self, path: Path, *, run_id: str, resume: bool = False) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._sequence = 0
        if resume:
            self._sequence = self._validated_existing_sequence()
        elif self.path.exists() and self.path.stat().st_size:
            raise ValueError(
                f"Refusing to append a new audit run to existing JSONL: {self.path}"
            )

    def _validated_existing_sequence(self) -> int:
        if not self.path.exists():
            return 0
        count = 0
        for count, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid existing audit line {count}: {exc}") from exc
            if event.get("run_id") != self.run_id:
                raise ValueError("Existing audit belongs to another run")
            if event.get("sequence") != count - 1:
                raise ValueError("Existing audit sequence is not contiguous")
        return count

    def record(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "sequence": self._sequence,
            "timestamp_utc": _utc_now(),
            "event_type": event_type,
            "payload": _redact(payload),
        }
        self._sequence += 1
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
