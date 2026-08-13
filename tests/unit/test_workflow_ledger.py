from pathlib import Path
import json

import pytest

from bci_autodiscovery.workflow import WorkflowLedger, WorkflowTransitionError


def _artifact(path: Path, dataset_id: str, status: str) -> Path:
    path.write_text(
        json.dumps({"dataset_id": dataset_id, "status": status}),
        encoding="utf-8",
    )
    return path


def test_workflow_is_dataset_neutral_ordered_and_agent_gated(tmp_path: Path) -> None:
    ledger = WorkflowLedger(path=tmp_path / "state.json", dataset_id="unseen-dataset")
    inspection = _artifact(tmp_path / "inspection.json", "unseen-dataset", "validated")
    profile = _artifact(tmp_path / "profile.json", "unseen-dataset", "validated")
    canonical = _artifact(tmp_path / "canonical.json", "unseen-dataset", "validated")
    with pytest.raises(WorkflowTransitionError, match="Expected next stage"):
        ledger.record_stage(
            stage="dataset_profile",
            artifact_path=profile,
            artifact_type="DatasetProfile",
            artifact_status="validated",
            upstream_sha256=[],
        )
    first = ledger.record_stage(
        stage="dataset_inspection",
        artifact_path=inspection,
        artifact_type="DatasetInspection",
        artifact_status="validated",
        upstream_sha256=[],
    )
    second = ledger.record_stage(
        stage="dataset_profile",
        artifact_path=profile,
        artifact_type="DatasetProfile",
        artifact_status="validated",
        upstream_sha256=[first["sha256"]],
    )
    record = ledger.record_stage(
        stage="canonical_search_space",
        artifact_path=canonical,
        artifact_type="CanonicalSearchSpace",
        artifact_status="validated",
        upstream_sha256=[second["sha256"]],
    )
    assert record["human_itemized_approval_used"] is False
    assert ledger.next_stage() == "frontier_discovery"
    assert ledger.to_dict()["dataset_id"] == "unseen-dataset"


def test_legacy_approval_ledger_is_inspectable_but_read_only(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "legacy-dataset",
                "status": "in_progress",
                "stages": {},
            }
        ),
        encoding="utf-8",
    )
    artifact = _artifact(tmp_path / "inspection.json", "legacy-dataset", "validated")
    ledger = WorkflowLedger(path=path, dataset_id="legacy-dataset")

    assert ledger.to_dict()["legacy_read_only"] is True
    with pytest.raises(WorkflowTransitionError, match="historical read-only"):
        ledger.record_stage(
            stage="dataset_inspection",
            artifact_path=artifact,
            artifact_type="DatasetInspection",
            artifact_status="validated",
            upstream_sha256=[],
        )
