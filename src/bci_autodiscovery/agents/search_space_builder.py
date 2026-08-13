"""Agent that maps an evidence-backed dataset profile to a legal broad space."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci_autodiscovery.search import build_search_space_draft

from .contracts import AgentRunResult
from .language import DEFAULT_LANGUAGE_INSTRUCTION
from .runtime import AgentRuntime
from .tools import ToolDefinition, ToolRegistry


SEARCH_SPACE_SYSTEM_PROMPT = f"""{DEFAULT_LANGUAGE_INSTRUCTION}
You are the Search-Space Builder Agent in an auditable
BCI research system. You must call the local build_search_space_draft tool before making
any claim. Preserve dataset hard constraints. Build a
broad conditional space, not a brute-force Cartesian product. Literature may propose new
directions beyond the local registry and frontier discovery must remain a required next
stage. Never exclude a compatible method because it is conventional, expensive, or not
yet implemented; record maturity and cost only as annotations. Do not assign session roles,
metrics, budgets, subject-specific choices, or confirmation access. An unimplemented method
cannot become executable. Do not
activate the resulting draft or access frozen confirmation sessions. The structured tool
artifact is authoritative; your final response is only a concise completion note."""


def create_search_space_builder_tools() -> ToolRegistry:
    registry = ToolRegistry()

    def handler(
        dataset_profile_path: str,
        component_registry_path: str,
    ) -> dict[str, Any]:
        return build_search_space_draft(
            dataset_profile_path=dataset_profile_path,
            component_registry_path=component_registry_path,
        )

    registry.register(
        ToolDefinition(
            name="build_search_space_draft",
            description=(
                "Build a non-executable, evidence-linked draft of the broad legal BCI "
                "pipeline search space from a dataset profile and component registry."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "dataset_profile_path": {"type": "string"},
                    "component_registry_path": {"type": "string"},
                },
                "required": [
                    "dataset_profile_path",
                    "component_registry_path",
                ],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="draft_only",
            tags=("read-only", "deterministic", "search-space-draft"),
        ),
        handler,
    )
    return registry


@dataclass
class SearchSpaceBuilderAgent:
    runtime: AgentRuntime

    def run(
        self,
        *,
        dataset_profile_path: Path,
        component_registry_path: Path,
    ) -> AgentRunResult:
        request = {
            "dataset_profile_path": str(Path(dataset_profile_path).expanduser().resolve()),
            "component_registry_path": str(
                Path(component_registry_path).expanduser().resolve()
            ),
        }
        return self.runtime.run(
            system_prompt=SEARCH_SPACE_SYSTEM_PROMPT,
            user_prompt=json.dumps(request, ensure_ascii=False, sort_keys=True),
        )
