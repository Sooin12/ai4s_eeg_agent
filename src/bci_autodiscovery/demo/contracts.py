"""Hash-bound synthetic authorities used only by the engineering demo."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bci_autodiscovery.workflow.autonomy import sha256_path


def write_json(path: Path, value: object, *, refuse_overwrite: bool = False) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable demo artifact: {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_demo_dataset_contract(root: Path, *, dataset_id: str) -> tuple[Path, Path]:
    """Create a valid DatasetLevelContract without claiming real-dataset evidence."""

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    evidence_source = root / "synthetic_generation_evidence.json"
    write_json(
        evidence_source,
        {
            "dataset_id": dataset_id,
            "status": "deterministic_engineering_fixture",
            "scientific_claim_authorized": False,
            "subject_phenotypes": ["mu_power", "beta_covariance", "individual_beta_power"],
        },
    )
    profile = {
        "schema_version": "1.0",
        "dataset": {"id": dataset_id},
        "paradigm": {
            "family": "motor_imagery",
            "actions": [{"label": "left"}, {"label": "right"}],
        },
        "resting_state": {"present": False},
        "signal": {
            "channel_count": 8,
            "sampling_frequency_hz": 128,
            "modalities": ["EEG"],
        },
        "equipment": {"status": "synthetic_engineering_fixture"},
        "events": {"common_analysis_window_s": [0.0, 1.0]},
        "sessions": {"session_indices": ["profile", "search", "confirm"], "sessions_per_subject": 3},
        "volume": {"subjects": 3, "sessions": 9, "trials": 576},
        "quality": {"status": "generated_and_verified_by_demo"},
        "constraints": {
            "allowed": ["deterministic engineering validation"],
            "forbidden": ["external scientific efficacy claim"],
            "requires_research_design_decision": [
                "session role assignment",
                "evaluation metric and threshold",
                "candidate execution budget",
            ],
            "external_authority_blockers": [],
        },
        "evidence": [
            {
                "source": str(evidence_source),
                "sha256": sha256_path(evidence_source),
                "claim": "hash-bound deterministic synthetic generation specification",
            }
        ],
    }
    profile_path = root / "dataset_profile.json"
    write_json(profile_path, profile)

    canonical_path = root / "canonical_search_space.json"
    registry_path = root / "component_registry.json"
    evidence_db_path = root / "literature_evidence.sqlite"
    canonical_space = {
        "component_count": 5,
        "dimensions": {
            "preprocessing": [{"component_id": "bandpass", "execution_status": "not_activated"}],
            "channel_selection": [
                {"component_id": "all_channels", "execution_status": "not_activated"},
                {"component_id": "profile_ranked_named_channels", "execution_status": "not_activated"},
            ],
            "features": [
                {"component_id": "log_bandpower", "execution_status": "not_activated"},
                {"component_id": "csp_log_variance", "execution_status": "not_activated"},
            ],
            "models": [{"component_id": "shrinkage_lda", "execution_status": "not_activated"}],
        },
    }
    write_json(canonical_path, {"dataset_id": dataset_id, "canonical_space": canonical_space})
    write_json(registry_path, {"registry_id": "demo-executable-components-v1"})
    evidence_db_path.write_bytes(b"synthetic-demo-dataset-level-evidence\n")
    provenance = {
        "dataset_profile": {"path": str(profile_path), "sha256": sha256_path(profile_path)},
        "canonical_search_space": {"path": str(canonical_path), "sha256": sha256_path(canonical_path)},
        "component_registry": {"path": str(registry_path), "sha256": sha256_path(registry_path)},
        "evidence_db": {"path": str(evidence_db_path), "sha256": sha256_path(evidence_db_path)},
    }
    stage_boundary = {
        "session_roles_assigned": False,
        "evaluation_metrics_selected": False,
        "experiment_budget_allocated": False,
        "subject_data_accessed": False,
        "confirmation_data_accessed": False,
        "execution_activation_performed": False,
    }
    draft = {
        "schema_version": "2.0",
        "contract_id": f"{dataset_id}-dataset-level-draft",
        "dataset_id": dataset_id,
        "status": "dataset_level_draft_awaiting_critic",
        "canonical_space": canonical_space,
        "compatibility_rules": [],
        "dataset_hard_constraints": ["engineering fixture; no external efficacy claim"],
        "deferred_to_downstream_agents": {
            "research_design_decisions": profile["constraints"]["requires_research_design_decision"],
            "component_decisions": ["complete individualized pipeline composition"],
            "external_authority_blockers": [],
        },
        "excluded_components": [],
        "frontier_space": {"directions": []},
        "provenance": provenance,
        "stage_boundary": stage_boundary,
    }
    draft_path = root / "dataset_level_draft.json"
    write_json(draft_path, draft)
    critique = {
        "schema_version": "1.0",
        "review_id": f"{dataset_id}-dataset-critic-pass",
        "dataset_id": dataset_id,
        "reviewed_draft_sha256": sha256_path(draft_path),
        "verdict": "pass",
        "findings": [],
        "required_revisions": [],
        "rationale": "The hash-bound engineering fixture stays inside the Dataset-Level stage boundary.",
        "source_draft": {"path": str(draft_path), "sha256": sha256_path(draft_path)},
    }
    critique_path = root / "dataset_critique.json"
    write_json(critique_path, critique)
    contract = deepcopy(draft)
    contract.update(
        {
            "contract_id": f"{dataset_id}-dataset-level-{sha256_path(draft_path)[:12]}",
            "status": "frozen_dataset_level_contract",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_critic": {
                "path": str(critique_path),
                "sha256": sha256_path(critique_path),
                "verdict": "pass",
            },
            "freeze_record": {
                "source_draft": {"path": str(draft_path), "sha256": sha256_path(draft_path)},
                "human_itemized_approval_used": False,
                "network_discovery_completed": True,
                "executable_activation_performed": False,
                "session_protocol_created": False,
                "subject_or_confirmation_data_accessed": False,
            },
        }
    )
    contract_path = root / "dataset_level_contract.json"
    write_json(contract_path, contract)

    envelope = {
        "schema_version": "2.0",
        "envelope_id": f"{dataset_id}-autonomy-v2",
        "status": "authorized",
        "objective": {
            "research_question": "Can the autonomous loop discover distinct pipelines for heterogeneous synthetic subjects?",
            "paradigm": "motor_imagery",
            "conclusion_scope": "internal_evidence_report",
        },
        "dataset": {
            "dataset_id": dataset_id,
            "dataset_level_contract_path": str(contract_path),
            "dataset_level_contract_sha256": sha256_path(contract_path),
        },
        "resource_budget": {
            "max_research_cycles": 6,
            "max_candidate_executions": 6,
            "max_compute_seconds": 7200,
            "max_api_tokens": 500000,
            "max_paid_cost": 12.0,
            "paid_cost_currency": "USD",
        },
        "permissions": {
            "allow_network_literature": True,
            "allow_logical_exclusions": True,
            "allow_first_confirmation_access": True,
        },
        "forbidden_actions": [
            "modify_or_delete_raw_data",
            "use_confirmation_for_search",
            "publish_external_claims",
        ],
        "confirmation_policy": {
            "max_access_count": 1,
            "requires_pipeline_lock": True,
            "reopen_search_after_confirmation": False,
        },
    }
    envelope_path = root / "autonomy_envelope.json"
    write_json(envelope_path, envelope)
    return contract_path, envelope_path
