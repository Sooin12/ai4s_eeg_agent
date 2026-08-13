from __future__ import annotations

import json
from pathlib import Path

from bci_autodiscovery.agents.contracts import ModelResponse, ToolCall
from bci_autodiscovery.agents.protocol_planner import (
    ProtocolPlannerAgent,
    create_protocol_planner_tools,
)
from bci_autodiscovery.agents.providers import ScriptedProvider
from bci_autodiscovery.agents.runtime import AgentRuntime


def _profile(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset": {"id": "unseen_fixture"},
                "paradigm": {"family": "fixture", "actions": [{"label": "a"}, {"label": "b"}]},
                "resting_state": {"present": False},
                "signal": {"channel_count": 2, "sampling_frequency_hz": 100, "modalities": ["EEG"]},
                "equipment": {},
                "events": {"common_analysis_window_s": [0, 1]},
                "sessions": {"session_indices": [10, 20, 30], "sessions_per_subject": 3},
                "volume": {"trials": 60},
                "quality": {},
                "constraints": {"allowed": [], "forbidden": [], "requires_human_decision": []},
                "evidence": [{"source": "fixture"}],
            }
        ),
        encoding="utf-8",
    )


def test_protocol_planner_reads_profile_and_records_nonactivated_proposal(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    _profile(profile_path)
    tools, context = create_protocol_planner_tools(dataset_profile_path=profile_path)
    proposal = {
        "schema_version": "1.0",
        "protocol_id": "fixture-proposal",
        "dataset_id": "unseen_fixture",
        "status": "proposed_requires_human_approval",
        "split_unit": "session",
        "data_roles": {
            "profiling_and_calibration": ["10"],
            "pipeline_search_and_lock": ["20"],
            "frozen_confirmation": ["30"],
        },
        "leakage_rules": {
            "confirmation_inaccessible_before_lock": True,
            "confirmation_cannot_select_pipeline": True,
            "confirmation_cannot_set_thresholds": True,
            "all_fitting_training_partition_only": True,
            "repeat_confirmation_access_requires_approval": True,
        },
        "rationale": ["fixture-generated rationale"],
        "quality_anomaly_policy": ["record but do not silently exclude"],
        "alternatives_considered": ["another fixture partition"],
        "risks_and_open_decisions": ["requires human approval"],
    }
    provider = ScriptedProvider(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall("read", "read_dataset_profile_contract", {}),
                )
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "record",
                        "record_protocol_proposal",
                        {"proposal": proposal},
                    ),
                )
            ),
            ModelResponse(content="协议草案已记录，等待用户审计。"),
        ]
    )
    result = ProtocolPlannerAgent(
        runtime=AgentRuntime(provider=provider, tools=tools, run_id="protocol-test"),
        context=context,
    ).run()
    recorded = result.latest_tool_result("record_protocol_proposal")
    assert result.status == "completed"
    assert recorded["activation_performed"] is False
    assert recorded["status"] == "proposed_requires_human_approval"
    assert recorded["source_profile"]["sha256"]
