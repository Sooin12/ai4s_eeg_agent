from __future__ import annotations

from pathlib import Path

from bci_autodiscovery.profiling import DatasetFormatCatalog
from bci_autodiscovery.profiling.adapters import create_default_adapter_registry


def test_catalog_recognizes_mainstream_containers_without_claiming_semantics(
    tmp_path: Path,
) -> None:
    (tmp_path / "recording.edf").write_bytes(b"fixture")
    (tmp_path / "epochs.mat").write_bytes(b"fixture")
    (tmp_path / "streams.xdf").write_bytes(b"fixture")

    candidates = DatasetFormatCatalog().inspect(tmp_path)
    by_id = {item["format_id"]: item for item in candidates}

    assert {"edf_edfplus", "mat_container", "xdf"} <= set(by_id)
    assert by_id["mat_container"]["interpretation_level"] == (
        "generic_container_requires_semantic_sidecar"
    )
    assert "format_detection_does_not_establish_experiment_semantics" in (
        by_id["xdf"]["limitations"]
    )


def test_brainvision_probe_checks_required_companions(tmp_path: Path) -> None:
    (tmp_path / "record.vhdr").write_text("header", encoding="utf-8")
    candidate = DatasetFormatCatalog().inspect(tmp_path)[0]
    assert candidate["format_id"] == "brainvision_core"
    assert candidate["companion_complete"] is False

    (tmp_path / "record.vmrk").write_text("markers", encoding="utf-8")
    (tmp_path / "record.eeg").write_bytes(b"signal")
    candidate = DatasetFormatCatalog().inspect(tmp_path)[0]
    assert candidate["companion_complete"] is True


def test_registry_reports_recognized_format_but_refuses_profile_selection(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "unknown_axes.mat").write_bytes(b"fixture")
    validation = tmp_path / "validation.json"
    validation.write_text("{}", encoding="utf-8")

    result = create_default_adapter_registry().inspect(
        dataset_root=dataset_root, validation_path=validation
    )

    assert result["status"] == "recognized_format_requires_semantic_mapping"
    assert result["selected_adapter_id"] is None
    assert result["semantic_gate"]["profile_ready"] is False
    assert "array_axes_from_file_extension" in result["semantic_gate"][
        "forbidden_inferences"
    ]
