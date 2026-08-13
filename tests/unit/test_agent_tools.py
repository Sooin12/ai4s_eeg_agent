from __future__ import annotations

import pytest

from bci_autodiscovery.agents.tools import ToolArgumentError, ToolDefinition, ToolRegistry


def test_tool_registry_rejects_unknown_arguments_before_handler() -> None:
    calls: list[int] = []
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="square",
            description="Square an integer.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        ),
        lambda value: calls.append(value) or value * value,
    )

    with pytest.raises(ToolArgumentError):
        registry.execute("square", {"value": 2, "unexpected": True})

    assert calls == []


def test_tool_registry_validates_basic_types() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="square",
            description="Square an integer.",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        ),
        lambda value: value * value,
    )

    assert registry.execute("square", {"value": 3}) == 9
    with pytest.raises(ToolArgumentError):
        registry.execute("square", {"value": "3"})
