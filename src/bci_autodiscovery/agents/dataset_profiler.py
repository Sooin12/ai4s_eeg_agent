"""First research agent: dataset-level understanding backed by local evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bci_autodiscovery.profiling import (
    DatasetAdapterRegistry,
    create_default_adapter_registry,
)

from .contracts import AgentRunResult
from .language import DEFAULT_LANGUAGE_INSTRUCTION
from .runtime import AgentRuntime
from .tools import ToolDefinition, ToolRegistry


DATASET_PROFILER_SYSTEM_PROMPT = f"""{DEFAULT_LANGUAGE_INSTRUCTION}
You are the Dataset Profiler Agent in an auditable
BCI research system. You must first call inspect_dataset, then call profile_dataset with
the selected adapter. If inspection reports ambiguity, a recognized container requiring
semantic mapping, or unsupported data, stop without
inventing an adapter. Treat tool output as evidence, never invent unavailable metadata, preserve all
limitations, and do not choose a search/confirmation session split. Raw EEG stays local.
After profiling, call inspect_dataset_profiler_status and provide only a concise completion
note once complete is true; the structured tool artifact is the authoritative output
consumed by downstream agents."""


def create_dataset_profiler_tools(
    adapter_registry: DatasetAdapterRegistry | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    adapters = adapter_registry or create_default_adapter_registry()
    inspected: dict[tuple[str, str], str] = {}
    inspection_result: dict[str, Any] | None = None
    profile_result: dict[str, Any] | None = None

    def inspect_dataset(dataset_root: str, validation_path: str) -> dict[str, Any]:
        nonlocal inspection_result
        result = adapters.inspect(
            dataset_root=Path(dataset_root), validation_path=Path(validation_path)
        )
        inspection_result = result
        selected = result.get("selected_adapter_id")
        if selected:
            key = (
                str(Path(dataset_root).expanduser().resolve()),
                str(Path(validation_path).expanduser().resolve()),
            )
            inspected[key] = str(selected)
        return result

    def profile_dataset(
        dataset_id: str,
        adapter_id: str,
        dataset_root: str,
        validation_path: str,
    ) -> dict[str, Any]:
        nonlocal profile_result
        key = (
            str(Path(dataset_root).expanduser().resolve()),
            str(Path(validation_path).expanduser().resolve()),
        )
        selected = inspected.get(key)
        if selected is None:
            raise ValueError("inspect_dataset must succeed before profile_dataset")
        if adapter_id != selected:
            raise ValueError(
                f"profile_dataset adapter {adapter_id} does not match inspected adapter {selected}"
            )
        profile_result = adapters.profile(
            adapter_id=adapter_id,
            dataset_id_hint=dataset_id,
            dataset_root=Path(dataset_root),
            validation_path=Path(validation_path),
        )
        return profile_result

    registry.register(
        ToolDefinition(
            name="inspect_dataset",
            description=(
                "Inspect local structure and validation evidence, rank installed dataset "
                "adapters, and select one without transmitting raw signal samples."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "dataset_root": {"type": "string"},
                    "validation_path": {"type": "string"},
                },
                "required": ["dataset_root", "validation_path"],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="read_only_dataset_inspection",
            tags=("read-only", "adapter-discovery", "no-raw-signal-egress"),
        ),
        inspect_dataset,
    )

    registry.register(
        ToolDefinition(
            name="profile_dataset",
            description=(
                "Use the inspected adapter to read a validated dataset locally and return a normalized, "
                "evidence-linked dataset profile. No raw signal samples leave the tool."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "adapter_id": {"type": "string", "enum": list(adapters.adapter_ids)},
                    "dataset_root": {"type": "string"},
                    "validation_path": {"type": "string"},
                },
                "required": ["dataset_id", "adapter_id", "dataset_root", "validation_path"],
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="read_only_dataset_understanding",
            tags=("read-only", "deterministic", "dataset-profile"),
        ),
        profile_dataset,
    )

    def profiler_status() -> dict[str, Any]:
        inspection_status = (
            inspection_result.get("status") if inspection_result is not None else None
        )
        selected_adapter_id = (
            inspection_result.get("selected_adapter_id")
            if inspection_result is not None
            else None
        )
        terminal_without_profile = inspection_status in {
            "adapter_ambiguous_requires_review",
            "recognized_format_requires_semantic_mapping",
            "unsupported_requires_new_adapter",
        }
        return {
            "inspection_completed": inspection_result is not None,
            "inspection_status": inspection_status,
            "selected_adapter_id": selected_adapter_id,
            "profile_completed": profile_result is not None,
            "profile_dataset_id": (
                (profile_result.get("dataset") or {}).get("id")
                if profile_result is not None
                else None
            ),
            "terminal_without_profile": terminal_without_profile,
            "complete": profile_result is not None or terminal_without_profile,
        }

    registry.register(
        ToolDefinition(
            name="inspect_dataset_profiler_status",
            description="Inspect whether a validated DatasetProfile was produced successfully.",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            approval="never",
            decision_kind="read_only_status",
            tags=("read-only", "completion-check", "dataset-profile"),
        ),
        profiler_status,
    )
    return registry


@dataclass
class DatasetProfilerAgent:
    runtime: AgentRuntime

    def run(
        self,
        *,
        dataset_id: str,
        dataset_root: Path,
        validation_path: Path,
    ) -> AgentRunResult:
        request = {
            "dataset_id": dataset_id,
            "dataset_root": str(Path(dataset_root).expanduser().resolve()),
            "validation_path": str(Path(validation_path).expanduser().resolve()),
        }

        def completion_check() -> dict[str, Any]:
            return self.runtime.tools.execute("inspect_dataset_profiler_status", {})

        return self.runtime.run(
            system_prompt=DATASET_PROFILER_SYSTEM_PROMPT,
            user_prompt=json.dumps(request, ensure_ascii=False, sort_keys=True),
            completion_check=completion_check,
            complete_on_tool_state=True,
        )
