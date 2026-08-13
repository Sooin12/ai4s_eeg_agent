from __future__ import annotations

from bci_autodiscovery.agents.audit import MemoryAuditSink
from bci_autodiscovery.agents.contracts import ModelResponse, ToolCall
from bci_autodiscovery.agents.providers import ScriptedProvider
from bci_autodiscovery.agents.runtime import AgentRuntime, RuntimeLimits
from bci_autodiscovery.agents.tools import ToolDefinition, ToolRegistry


def _echo_registry(*, approval: str = "never", calls: list[str] | None = None) -> ToolRegistry:
    registry = ToolRegistry()

    def echo(value: str) -> dict[str, str]:
        if calls is not None:
            calls.append(value)
        return {"echo": value}

    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo a string.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            approval=approval,
            approval_reason="User must approve this scientific decision.",
            decision_kind="scientific_protocol",
        ),
        echo,
    )
    return registry


def test_runtime_executes_tool_and_returns_audited_result() -> None:
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(ToolCall(id="call-1", name="echo", arguments={"value": "ok"}),)
            ),
            ModelResponse(content="done"),
        ]
    )
    audit = MemoryAuditSink()
    result = AgentRuntime(
        provider=provider,
        tools=_echo_registry(),
        audit=audit,
        run_id="test-run",
    ).run(system_prompt="system", user_prompt="user")

    assert result.status == "completed"
    assert result.final_text == "done"
    assert result.latest_tool_result("echo") == {"echo": "ok"}
    assert [event["event_type"] for event in audit.events][-1] == "run_completed"


def test_runtime_stops_before_approval_guarded_tool() -> None:
    handler_calls: list[str] = []
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(id="call-approval", name="echo", arguments={"value": "locked"}),
                )
            )
        ]
    )
    result = AgentRuntime(
        provider=provider,
        tools=_echo_registry(approval="always", calls=handler_calls),
        run_id="approval-run",
    ).run(system_prompt="system", user_prompt="user")

    assert result.status == "needs_approval"
    assert result.approval_request is not None
    assert result.approval_request.decision_kind == "scientific_protocol"
    assert handler_calls == []


def test_runtime_blocks_repeated_identical_calls() -> None:
    repeated = ModelResponse(
        tool_calls=(ToolCall(id="call", name="echo", arguments={"value": "same"}),)
    )
    provider = ScriptedProvider([repeated, repeated])
    result = AgentRuntime(
        provider=provider,
        tools=_echo_registry(),
        limits=RuntimeLimits(max_iterations=3, max_tool_calls=4, max_repeated_identical_calls=1),
        run_id="repeat-run",
    ).run(system_prompt="system", user_prompt="user")

    assert result.status == "limit_reached"
    assert "Repeated identical" in (result.error or "")


def test_runtime_rejects_premature_final_and_allows_autonomous_repair() -> None:
    provider = ScriptedProvider(
        [
            ModelResponse(content="premature"),
            ModelResponse(
                tool_calls=(
                    ToolCall(id="repair", name="echo", arguments={"value": "fixed"}),
                )
            ),
            ModelResponse(content="done"),
        ]
    )
    audit = MemoryAuditSink()
    calls: list[str] = []

    result = AgentRuntime(
        provider=provider,
        tools=_echo_registry(calls=calls),
        audit=audit,
        limits=RuntimeLimits(max_iterations=4, max_tool_calls=2),
        run_id="completion-gate",
    ).run(
        system_prompt="system",
        user_prompt="user",
        completion_check=lambda: {
            "complete": calls == ["fixed"],
            "repair_required": calls != ["fixed"],
        },
    )

    assert result.status == "completed"
    assert result.final_text == "done"
    assert calls == ["fixed"]
    event_types = [event["event_type"] for event in audit.events]
    assert event_types.count("completion_checked") == 2
    assert "completion_rejected" in event_types


def test_runtime_fails_closed_when_completion_check_contract_is_invalid() -> None:
    result = AgentRuntime(
        provider=ScriptedProvider([ModelResponse(content="done")]),
        tools=_echo_registry(),
        run_id="invalid-completion-check",
    ).run(
        system_prompt="system",
        user_prompt="user",
        completion_check=lambda: None,  # type: ignore[return-value]
    )

    assert result.status == "failed"
    assert "completion_check must return" in (result.error or "")


def test_runtime_can_finish_from_deterministic_tool_state_without_final_turn() -> None:
    calls: list[str] = []
    result = AgentRuntime(
        provider=ScriptedProvider(
            [
                ModelResponse(
                    tool_calls=(
                        ToolCall("complete", "echo", {"value": "finished"}),
                    )
                )
            ]
        ),
        tools=_echo_registry(calls=calls),
        run_id="tool-state-completion",
    ).run(
        system_prompt="system",
        user_prompt="user",
        completion_check=lambda: {"complete": calls == ["finished"]},
        complete_on_tool_state=True,
    )

    assert result.status == "completed"
    assert result.iterations == 1
    assert result.final_text == "确定性完成门已确认所有必要工具状态。"
