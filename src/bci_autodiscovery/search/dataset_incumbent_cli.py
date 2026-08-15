"""Freeze a subject-balanced dataset-wide incumbent from standard epoch contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bci_autodiscovery.pipelines import DeterministicPipelineExecutor, PipelineSpec
from bci_autodiscovery.profiling import MatEpochSource
from bci_autodiscovery.search.dataset_incumbent import build_dataset_incumbent
from bci_autodiscovery.workflow.autonomy import load_json_object, sha256_path
from bci_autodiscovery.workflow.dataset_contract import load_dataset_level_contract


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise ValueError(f"Refusing to overwrite incumbent artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--dataset-level-contract", type=Path, required=True)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", required=True)
    parser.add_argument("--search-session", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.dataset_root.expanduser().resolve()
    validation_path = args.validation.expanduser().resolve()
    contract_path = args.dataset_level_contract.expanduser().resolve()
    plan_path = args.candidate_plan.expanduser().resolve()
    contract = load_dataset_level_contract(contract_path)
    plan = load_json_object(plan_path)
    if plan.get("schema_version") != "1.0" or plan.get("status") != "frozen_before_execution":
        raise ValueError("Candidate plan must be frozen_before_execution schema 1.0")
    dataset_id = str(contract["dataset_id"])
    if plan.get("dataset_id") != dataset_id:
        raise ValueError("Candidate plan and DatasetLevelContract disagree")
    candidates = [PipelineSpec.from_dict(item) for item in plan.get("candidates") or []]
    maximum = int(plan.get("maximum_candidate_configurations", 0))
    if maximum < 1 or len(candidates) > maximum:
        raise ValueError("Candidate plan exceeds its frozen configuration budget")
    source = MatEpochSource(dataset_root=root, validation_path=validation_path)
    subjects = [str(item) for item in args.subjects]
    executors = {
        subject_id: DeterministicPipelineExecutor(
            sessions=[
                source.load_session(
                    subject_id=subject_id,
                    session_id=str(args.search_session),
                )
            ]
        )
        for subject_id in subjects
    }
    artifact = build_dataset_incumbent(
        dataset_id=dataset_id,
        executors=executors,
        candidates=candidates,
        primary_metric=str(plan.get("primary_metric", "balanced_accuracy")),
        stability_penalty=float(plan.get("stability_penalty", 0.25)),
        minimum_subjects=int(plan.get("minimum_subjects", 3)),
        minimum_candidates=int(plan.get("minimum_candidates", 2)),
        minimum_personalization_gain=float(
            plan.get("minimum_personalization_gain", 0.03)
        ),
        source_contracts={
            "dataset_level_contract": {
                "path": str(contract_path),
                "sha256": sha256_path(contract_path),
                "contract_id": contract["contract_id"],
            },
            "candidate_plan": {
                "path": str(plan_path),
                "sha256": sha256_path(plan_path),
                "frozen_before_execution": True,
            },
            "semantic_validation": {
                "path": str(validation_path),
                "sha256": sha256_path(validation_path),
            },
        },
    )
    _write_json_exclusive(args.output, artifact)
    print(f"status: {artifact['status']}")
    print(f"subjects: {len(artifact['cohort_subject_ids'])}")
    print(f"candidates: {len(artifact['search_trace'])}")
    print(f"selected_pipeline: {artifact['selected_pipeline']['pipeline_id']}")
    print(f"macro_mean: {artifact['selected_score_summary']['macro_mean']:.6f}")
    print(f"output: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
