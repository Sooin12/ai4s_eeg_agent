"""Bounded model/tool loop with approval gates and complete audit events."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from .audit import AuditSink, NullAuditSink
from .contracts import (
    AgentRunResult,
    ApprovalRequest,
    ModelUsage,
    ToolExecutionRecord,
)
from .providers import ModelProvider, ProviderError
from .tools import ToolError, ToolRegistry


@dataclass(frozen=True)
class RuntimeLimits:
    max_iterations: int = 8
    max_tool_calls: int = 16
    max_repeated_identical_calls: int = 2

    def __post_init__(self) -> None:
        if min(
            self.max_iterations,
            self.max_tool_calls,
            self.max_repeated_identical_calls,
        ) < 1:
            raise ValueError("All runtime limits must be positive")


def _add_usage(left: ModelUsage, right: ModelUsage) -> ModelUsage:
    return ModelUsage(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        cached_tokens=left.cached_tokens + right.cached_tokens,
    )


def _fingerprint_tool_call(name: str, arguments: dict[str, Any]) -> str:
    raw = json.dumps(
        {"name": name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AgentRuntime:
    def __init__(
        self,
        *,
        provider: ModelProvider,
        tools: ToolRegistry,
        audit: AuditSink | None = None,
        limits: RuntimeLimits | None = None,
        run_id: str | None = None,
        audit_path: str | None = None,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.audit = audit or NullAuditSink()
        self.limits = limits or RuntimeLimits()
        self.run_id = run_id or uuid.uuid4().hex
        self.audit_path = audit_path
        self.progress_callback = progress_callback

    def _progress(self, event: str, payload: dict[str, Any] | None = None) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event, payload or {})

    def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        completion_check: Callable[[], dict[str, Any]] | None = None,
        complete_on_tool_state: bool = False,
    ) -> AgentRunResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = AgentRunResult(
            run_id=self.run_id,
            status="failed",
            messages=messages,
            audit_path=self.audit_path,
        )
        total_usage = ModelUsage()
        call_count = 0
        repeated_calls: dict[str, int] = {}
        self.audit.record(
            "run_started",
            {
                "provider": self.provider.name,
                "model": self.provider.model,
                "provider_config": self.provider.audit_config(),
                "tool_count": len(self.tools),
                "limits": self.limits.__dict__,
                "initial_messages": messages,
            },
        )
        self._progress("run_started", {"run_id": self.run_id})

        for iteration in range(1, self.limits.max_iterations + 1):
            result.iterations = iteration
            self.audit.record(
                "model_request",
                {
                    "iteration": iteration,
                    "messages": messages,
                    "tools": self.tools.definitions(),
                },
            )
            self._progress("model_request", {"iteration": iteration})
            try:
                response = self.provider.complete(
                    messages=messages,
                    tools=self.tools.definitions(),
                )
            except Exception as exc:
                result.status = "failed"
                prefix = "" if isinstance(exc, ProviderError) else "Unexpected provider failure: "
                result.error = prefix + str(exc)
                result.total_usage = total_usage
                self.audit.record(
                    "run_failed", {"stage": "provider", "error": result.error}
                )
                self._progress("run_failed", {"stage": "provider", "error": result.error})
                return result

            total_usage = _add_usage(total_usage, response.usage)
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": response.content or "",
            }
            if response.reasoning_content:
                assistant["reasoning_content"] = response.reasoning_content
            if response.tool_calls:
                assistant["tool_calls"] = [call.to_message_dict() for call in response.tool_calls]
            messages.append(assistant)
            self.audit.record(
                "model_response",
                {
                    "iteration": iteration,
                    "model": response.model,
                    "provider_response_id": response.provider_response_id,
                    "content": response.content,
                    "tool_calls": [call.to_message_dict() for call in response.tool_calls],
                    "usage": response.usage.to_dict(),
                },
            )
            self._progress(
                "model_response",
                {"iteration": iteration, "tool_calls": len(response.tool_calls)},
            )

            if not response.tool_calls:
                if completion_check is not None:
                    try:
                        completion_state = completion_check()
                        if not isinstance(completion_state, dict):
                            raise TypeError(
                                "completion_check must return a JSON-object-like dict"
                            )
                        self.audit.record(
                            "completion_checked",
                            {
                                "iteration": iteration,
                                "state": completion_state,
                            },
                        )
                    except Exception as exc:
                        result.status = "failed"
                        result.error = (
                            "Deterministic completion check failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        result.total_usage = total_usage
                        self.audit.record(
                            "run_failed",
                            {"stage": "completion_check", "error": result.error},
                        )
                        self._progress(
                            "run_failed",
                            {"stage": "completion_check", "error": result.error},
                        )
                        return result
                    if completion_state.get("complete") is not True:
                        self.audit.record(
                            "completion_rejected",
                            {
                                "iteration": iteration,
                                "state": completion_state,
                                "model_final_text": response.content or "",
                            },
                        )
                        self._progress(
                            "completion_rejected",
                            {"iteration": iteration, "state": completion_state},
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "deterministic_completion_gate": "incomplete",
                                        "state": completion_state,
                                        "instruction": (
                                            "Do not finish yet. Diagnose the latest tool "
                                            "error and continue using the available tools "
                                            "until complete is true."
                                        ),
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                            }
                        )
                        continue
                result.status = "completed"
                result.final_text = response.content or ""
                result.total_usage = total_usage
                self.audit.record(
                    "run_completed",
                    {
                        "iterations": iteration,
                        "tool_calls": call_count,
                        "total_usage": total_usage.to_dict(),
                        "final_text": result.final_text,
                    },
                )
                self._progress("run_completed", {"iteration": iteration})
                return result

            for call in response.tool_calls:
                call_count += 1
                if call_count > self.limits.max_tool_calls:
                    result.status = "limit_reached"
                    result.error = "Maximum tool-call budget reached"
                    result.total_usage = total_usage
                    self.audit.record("run_halted", {"reason": result.error})
                    return result

                fingerprint = _fingerprint_tool_call(call.name, call.arguments)
                repeated_calls[fingerprint] = repeated_calls.get(fingerprint, 0) + 1
                if repeated_calls[fingerprint] > self.limits.max_repeated_identical_calls:
                    result.status = "limit_reached"
                    result.error = f"Repeated identical tool call blocked: {call.name}"
                    result.total_usage = total_usage
                    self.audit.record(
                        "run_halted",
                        {"reason": result.error, "call_id": call.id, "arguments": call.arguments},
                    )
                    return result

                try:
                    registered = self.tools.get(call.name)
                except ToolError as exc:
                    execution = ToolExecutionRecord(
                        call_id=call.id,
                        tool_name=call.name,
                        arguments=call.arguments,
                        error=str(exc),
                    )
                    result.tool_executions.append(execution)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps({"error": str(exc)}, ensure_ascii=False),
                        }
                    )
                    self.audit.record("tool_rejected", execution.to_dict())
                    continue

                definition = registered.definition
                self.audit.record(
                    "tool_requested",
                    {
                        "call_id": call.id,
                        "tool_name": call.name,
                        "arguments": call.arguments,
                        "approval": definition.approval,
                        "decision_kind": definition.decision_kind,
                    },
                )
                self._progress(
                    "tool_requested",
                    {"tool_name": call.name, "call_id": call.id},
                )
                if definition.approval == "always":
                    approval = ApprovalRequest(
                        call_id=call.id,
                        tool_name=call.name,
                        reason=definition.approval_reason or "Explicit user approval is required",
                        arguments=call.arguments,
                        decision_kind=definition.decision_kind,
                    )
                    result.status = "needs_approval"
                    result.approval_request = approval
                    result.total_usage = total_usage
                    self.audit.record("approval_required", approval.to_dict())
                    return result

                try:
                    tool_result = self.tools.execute(call.name, call.arguments)
                    execution = ToolExecutionRecord(
                        call_id=call.id,
                        tool_name=call.name,
                        arguments=call.arguments,
                        result=tool_result,
                    )
                    content = json.dumps(tool_result, ensure_ascii=False, sort_keys=True)
                    event_type = "tool_completed"
                except ToolError as exc:
                    execution = ToolExecutionRecord(
                        call_id=call.id,
                        tool_name=call.name,
                        arguments=call.arguments,
                        error=str(exc),
                    )
                    content = json.dumps({"error": str(exc)}, ensure_ascii=False)
                    event_type = "tool_failed"
                result.tool_executions.append(execution)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": content}
                )
                self.audit.record(event_type, execution.to_dict())
                self._progress(
                    event_type,
                    {"tool_name": call.name, "call_id": call.id},
                )

            if completion_check is not None and complete_on_tool_state:
                try:
                    completion_state = completion_check()
                    if not isinstance(completion_state, dict):
                        raise TypeError(
                            "completion_check must return a JSON-object-like dict"
                        )
                    self.audit.record(
                        "completion_checked",
                        {
                            "iteration": iteration,
                            "state": completion_state,
                            "trigger": "after_tool_execution",
                        },
                    )
                except Exception as exc:
                    result.status = "failed"
                    result.error = (
                        "Deterministic completion check failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    result.total_usage = total_usage
                    self.audit.record(
                        "run_failed",
                        {"stage": "completion_check", "error": result.error},
                    )
                    self._progress(
                        "run_failed",
                        {"stage": "completion_check", "error": result.error},
                    )
                    return result
                if completion_state.get("complete") is True:
                    result.status = "completed"
                    result.final_text = response.content or (
                        "确定性完成门已确认所有必要工具状态。"
                    )
                    result.total_usage = total_usage
                    self.audit.record(
                        "run_completed",
                        {
                            "iterations": iteration,
                            "tool_calls": call_count,
                            "total_usage": total_usage.to_dict(),
                            "final_text": result.final_text,
                            "completion_source": "deterministic_tool_state",
                        },
                    )
                    self._progress("run_completed", {"iteration": iteration})
                    return result

        result.status = "limit_reached"
        result.error = "Maximum model iteration budget reached"
        result.total_usage = total_usage
        self.audit.record("run_halted", {"reason": result.error})
        return result
