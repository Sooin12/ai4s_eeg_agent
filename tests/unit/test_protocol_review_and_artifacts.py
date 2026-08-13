from __future__ import annotations

import json
from pathlib import Path

import pytest

from bci_autodiscovery.agents.contracts import ModelResponse, ToolCall
from bci_autodiscovery.agents.protocol_reviewer import (
    ProtocolReviewerAgent,
    create_protocol_reviewer_tools,
)
from bci_autodiscovery.agents.providers import ScriptedProvider
from bci_autodiscovery.agents.runtime import AgentRuntime
from bci_autodiscovery.workflow.protocol_artifacts import (
    ProtocolArtifactError,
    ProtocolArtifactRegistry,
    sha256_file,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _profile() -> dict:
    return {
        "schema_version": "1.0",
        "dataset": {"id": "review_fixture"},
        "paradigm": {"family": "fixture", "actions": [{"label": "a"}, {"label": "b"}]},
        "resting_state": {"present": False},
        "signal": {"channel_count": 2, "sampling_frequency_hz": 100, "modalities": ["EEG"]},
        "equipment": {},
        "events": {"common_analysis_window_s": [0, 1]},
        "sessions": {"session_indices": [1, 2, 3], "sessions_per_subject": 3},
        "volume": {"trials": 60},
        "quality": {},
        "constraints": {"allowed": [], "forbidden": [], "requires_human_decision": []},
        "evidence": [{"source": "fixture"}],
    }


def _proposal(protocol_id: str = "fixture-v1") -> dict:
    return {
        "schema_version": "1.0",
        "protocol_id": protocol_id,
        "dataset_id": "review_fixture",
        "status": "proposed_requires_human_approval",
        "split_unit": "session",
        "data_roles": {
            "profiling_and_calibration": ["1"],
            "pipeline_search_and_lock": ["2"],
            "frozen_confirmation": ["3"],
        },
        "leakage_rules": {
            "confirmation_inaccessible_before_lock": True,
            "confirmation_cannot_select_pipeline": True,
            "confirmation_cannot_set_thresholds": True,
            "all_fitting_training_partition_only": True,
            "repeat_confirmation_access_requires_approval": True,
        },
        "rationale": ["agent rationale"],
        "quality_anomaly_policy": ["agent policy"],
        "alternatives_considered": ["agent alternative"],
        "risks_and_open_decisions": ["human decision pending"],
    }


def test_reviewer_records_a_versioned_nonactivated_revision(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    proposal_path = tmp_path / "proposal.json"
    _write_json(profile_path, _profile())
    _write_json(proposal_path, _proposal())
    revised = _proposal("fixture-v2")
    revised["rationale"] = ["revised by agent from user feedback"]
    tools, context = create_protocol_reviewer_tools(
        dataset_profile_path=profile_path,
        proposal_path=proposal_path,
        user_feedback="请修改理由",
    )
    provider = ScriptedProvider(
        [
            ModelResponse(tool_calls=(ToolCall("read", "read_protocol_review_context", {}),)),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        "revise",
                        "record_protocol_revision",
                        {
                            "revised_proposal": revised,
                            "change_summary": ["修改理由"],
                            "feedback_resolution": ["已处理"],
                        },
                    ),
                )
            ),
            ModelResponse(content="修订版已记录，仍待人工批准。"),
        ]
    )
    result = ProtocolReviewerAgent(
        runtime=AgentRuntime(provider=provider, tools=tools, run_id="review-test"),
        context=context,
    ).run()
    revision = result.latest_tool_result("record_protocol_revision")
    assert revision["status"] == "proposed_requires_human_approval"
    assert revision["activation_performed"] is False
    assert revision["revision"]["user_feedback_verbatim"] == "请修改理由"
    assert revision["revision"]["parent_sha256"]


def test_registry_preserves_revisions_and_exposes_only_approved_protocol(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    _write_json(profile_path, _profile())
    proposal_path = tmp_path / "proposal.json"
    proposal = _proposal()
    proposal["source_profile"] = {
        "path": str(profile_path),
        "sha256": sha256_file(profile_path),
    }
    _write_json(proposal_path, proposal)
    registry = ProtocolArtifactRegistry(root=tmp_path / "registry", dataset_id="review_fixture")
    revision = registry.register_revision(source_path=proposal_path, kind="agent_proposal")
    with pytest.raises(ProtocolArtifactError, match="no human-approved protocol"):
        registry.resolve_current_approved()
    approved = registry.approve(
        proposal_path=proposal_path,
        approved_by="fixture-user",
        decision_note="fixture approval",
    )
    approved_path = registry.resolve_current_approved()
    loaded = json.loads(approved_path.read_text(encoding="utf-8"))
    assert revision["sha256"]
    assert approved["status"] == "approved"
    assert loaded["approval"]["approved_by"] == "fixture-user"
    assert loaded["session_roles"] == loaded["data_roles"]
    assert registry.to_dict()["current_approved"]["path"] == str(approved_path)
