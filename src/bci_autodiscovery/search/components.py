"""Versioned BCI component registry contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ComponentRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ComponentRegistry:
    registry_id: str
    status: str
    components: tuple[dict[str, Any], ...]
    compatibility_rules: tuple[dict[str, Any], ...]
    source_path: Path

    @classmethod
    def load(cls, path: Path) -> "ComponentRegistry":
        source = Path(path).expanduser().resolve()
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ComponentRegistryError(f"Cannot load component registry {source}: {exc}") from exc
        if raw.get("schema_version") != "1.0":
            raise ComponentRegistryError("Unsupported component registry schema")
        components = raw.get("components") or []
        ids = [item.get("id") for item in components]
        if not components or any(not item for item in ids):
            raise ComponentRegistryError("Registry must contain named components")
        if len(ids) != len(set(ids)):
            raise ComponentRegistryError("Component IDs must be unique")
        required = {"id", "category", "family", "description", "maturity", "cost_tier", "requirements", "parameters"}
        for item in components:
            missing = sorted(required.difference(item))
            if missing:
                raise ComponentRegistryError(f"Component {item.get('id')} missing {missing}")
        known = set(ids)
        for rule in raw.get("compatibility_rules") or []:
            if rule.get("if_component") and rule["if_component"] not in known:
                raise ComponentRegistryError(f"Rule references unknown component: {rule}")
            for component_id in rule.get("requires_any") or []:
                if component_id not in known:
                    raise ComponentRegistryError(f"Rule references unknown requirement: {rule}")
        return cls(
            registry_id=str(raw["registry_id"]),
            status=str(raw["status"]),
            components=tuple(components),
            compatibility_rules=tuple(raw.get("compatibility_rules") or []),
            source_path=source,
        )
