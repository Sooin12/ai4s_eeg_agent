"""Serializable contracts shared by providers, tools, and the runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


RunStatus = Literal["completed", "needs_approval", "failed", "limit_reached"]


@dataclass(frozen=True)
class ToolCall:
    """A provider request to invoke one registered local tool."""

    id: str
    name: str
    arguments: dict[str, Any]

    def to_message_dict(self) -> dict[str, Any]:
        import json

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False, sort_keys=True),
            },
        }


@dataclass(frozen=True)
class ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "ModelUsage":
        usage = value or {}
        details = usage.get("prompt_tokens_details") or {}
        return cls(
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            cached_tokens=int(usage.get("cached_tokens") or details.get("cached_tokens") or 0),
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ModelResponse:
    """Normalized response returned by every model provider."""

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: ModelUsage = field(default_factory=ModelUsage)
    model: str | None = None
    provider_response_id: str | None = None
    reasoning_content: str | None = None


@dataclass(frozen=True)
class ToolExecutionRecord:
    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: Any | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ApprovalRequest:
    """A fail-closed checkpoint emitted before a guarded tool can run."""

    call_id: str
    tool_name: str
    reason: str
    arguments: dict[str, Any]
    decision_kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentRunResult:
    run_id: str
    status: RunStatus
    final_text: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_executions: list[ToolExecutionRecord] = field(default_factory=list)
    approval_request: ApprovalRequest | None = None
    error: str | None = None
    iterations: int = 0
    total_usage: ModelUsage = field(default_factory=ModelUsage)
    audit_path: str | None = None

    def latest_tool_result(self, tool_name: str) -> Any | None:
        for item in reversed(self.tool_executions):
            if item.tool_name == tool_name and item.error is None:
                return item.result
        return None

    def to_dict(self, *, include_messages: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "run_id": self.run_id,
            "status": self.status,
            "final_text": self.final_text,
            "tool_executions": [item.to_dict() for item in self.tool_executions],
            "approval_request": (
                self.approval_request.to_dict() if self.approval_request else None
            ),
            "error": self.error,
            "iterations": self.iterations,
            "total_usage": self.total_usage.to_dict(),
            "audit_path": self.audit_path,
        }
        if include_messages:
            result["messages"] = self.messages
        return result
