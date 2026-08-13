"""Local tool registry with fail-closed argument and approval checks."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Literal


ApprovalMode = Literal["never", "always"]


class ToolError(RuntimeError):
    """Base class for tool protocol failures."""


class UnknownToolError(ToolError):
    pass


class ToolArgumentError(ToolError):
    pass


class ToolExecutionError(ToolError):
    pass


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    approval: ApprovalMode = "never"
    approval_reason: str | None = None
    decision_kind: str = "engineering"
    tags: tuple[str, ...] = ()

    def to_api_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(frozen=True)
class RegisteredTool:
    definition: ToolDefinition
    handler: Callable[..., Any] = field(repr=False)


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list, tuple),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}


def _validate_value(value: Any, schema: dict[str, Any], location: str) -> None:
    expected = schema.get("type")
    if expected:
        accepted = _JSON_TYPES.get(expected)
        if accepted is None:
            raise ToolArgumentError(f"Unsupported schema type at {location}: {expected}")
        if expected in {"integer", "number"} and isinstance(value, bool):
            raise ToolArgumentError(f"{location} must be {expected}, not boolean")
        if not isinstance(value, accepted):
            raise ToolArgumentError(
                f"{location} must be {expected}; observed {type(value).__name__}"
            )
    if "enum" in schema and value not in schema["enum"]:
        raise ToolArgumentError(f"{location} must be one of {schema['enum']!r}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        missing = sorted(required.difference(value))
        if missing:
            raise ToolArgumentError(f"{location} is missing required fields: {missing}")
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value).difference(properties))
            if unknown:
                raise ToolArgumentError(f"{location} contains unknown fields: {unknown}")
        for key, nested in properties.items():
            if key in value:
                _validate_value(value[key], nested, f"{location}.{key}")
    if isinstance(value, (list, tuple)) and "items" in schema:
        for index, item in enumerate(value):
            _validate_value(item, schema["items"], f"{location}[{index}]")


def validate_json_value(
    value: Any, schema: dict[str, Any], *, location: str = "value"
) -> None:
    """Public deterministic schema gate shared by tools and artifact validators."""

    _validate_value(value, schema, location)


class ToolRegistry:
    """Registry exposed to a model; handlers remain local and auditable."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, definition: ToolDefinition, handler: Callable[..., Any]) -> None:
        if definition.name in self._tools:
            raise ValueError(f"Tool already registered: {definition.name}")
        if not definition.name.replace("_", "").isalnum():
            raise ValueError(f"Tool name must be alphanumeric/underscore: {definition.name}")
        self._tools[definition.name] = RegisteredTool(definition, handler)

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise UnknownToolError(f"Unknown tool: {name}") from exc

    def definitions(self) -> list[dict[str, Any]]:
        return [self._tools[name].definition.to_api_dict() for name in sorted(self._tools)]

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        registered = self.get(name)
        if not isinstance(arguments, dict):
            raise ToolArgumentError("Tool arguments must be a JSON object")
        _validate_value(arguments, registered.definition.input_schema, "arguments")
        try:
            signature = inspect.signature(registered.handler)
            signature.bind(**arguments)
            result = registered.handler(**arguments)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"{type(exc).__name__}: {exc}") from exc
        return result

    def __len__(self) -> int:
        return len(self._tools)
