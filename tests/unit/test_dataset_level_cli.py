from __future__ import annotations

from types import SimpleNamespace

from bci_autodiscovery.agents import dataset_level_cli


def test_raw_dataset_cli_uses_selected_real_provider_for_profiler(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}
    provider = SimpleNamespace(
        name="kimi",
        model="fixture-model",
        timeout_seconds=0.0,
        progress_callback=None,
        complete=lambda **_kwargs: None,
    )

    monkeypatch.setattr(
        dataset_level_cli.OpenAICompatibleProvider,
        "kimi",
        lambda **_kwargs: provider,
    )

    class FakeDatasetLevelAgent:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def run(self, **_kwargs):
            return SimpleNamespace(
                status="completed",
                cycles=1,
                artifacts={},
                error=None,
            )

    monkeypatch.setattr(dataset_level_cli, "DatasetLevelAgent", FakeDatasetLevelAgent)
    dataset_root = tmp_path / "dataset"
    validation = tmp_path / "validation.json"
    dataset_root.mkdir()
    validation.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"

    exit_code = dataset_level_cli.main(
        [
            "--dataset-root",
            str(dataset_root),
            "--validation",
            str(validation),
            "--provider",
            "kimi",
            "--model",
            "fixture-model",
            "--run-dir",
            str(run_dir),
            "--run-id",
            "real-profiler-wiring",
        ]
    )

    assert exit_code == 0
    assert captured["profiler_provider"] is provider
    assert provider.timeout_seconds == 240.0
