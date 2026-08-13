from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bci_autodiscovery.reporting import AgentOutputError, AgentOutputPublisher


def test_agent_output_publisher_creates_user_view_with_provenance(tmp_path: Path) -> None:
    source = tmp_path / "run" / "dataset_profile.json"
    source.parent.mkdir()
    source.write_text('{"dataset": {"id": "fixture_mi"}}\n', encoding="utf-8")

    record = AgentOutputPublisher(tmp_path / "agent_outputs").publish(
        dataset_id="fixture_mi",
        stage="dataset_understanding",
        artifact_name="dataset_profile",
        source_path=source,
        run_id="run-001",
    )

    published = Path(record["published_path"])
    assert published.read_bytes() == source.read_bytes()
    assert record["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    index = json.loads((published.parents[1] / "index.json").read_text(encoding="utf-8"))
    assert index["view_kind"] == "derived_user_view"
    assert index["artifacts"]["dataset_understanding/dataset_profile"][
        "run_id"
    ] == "run-001"
    for stage in range(1, 8):
        assert any(path.name.startswith(f"{stage:02d}_") for path in published.parents[1].iterdir())


def test_agent_output_publisher_rejects_path_traversal(tmp_path: Path) -> None:
    source = tmp_path / "artifact.json"
    source.write_text("{}", encoding="utf-8")
    with pytest.raises(AgentOutputError, match="Unsafe dataset_id"):
        AgentOutputPublisher(tmp_path / "outputs").publish(
            dataset_id="../escape",
            stage="reports",
            artifact_name="report",
            source_path=source,
            run_id="run-001",
        )
