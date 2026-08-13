from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bci_autodiscovery.workflow.autonomy import sha256_path


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def minimal_dataset_profile(
    dataset_id: str,
    *,
    session_indices: tuple[int, ...] = (1, 2, 3),
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "dataset": {"id": dataset_id},
        "paradigm": {
            "family": "motor_imagery",
            "actions": [{"label": "left"}, {"label": "right"}],
        },
        "resting_state": {"present": False},
        "signal": {
            "channel_count": 2,
            "sampling_frequency_hz": 100,
            "modalities": ["EEG"],
        },
        "equipment": {},
        "events": {"common_analysis_window_s": [0, 1]},
        "sessions": {
            "session_indices": list(session_indices),
            "sessions_per_subject": len(session_indices),
        },
        "volume": {"trials": 60},
        "quality": {},
        "constraints": {
            "allowed": [],
            "forbidden": [],
            "requires_research_design_decision": [
                "policy for incomplete trials",
                "search versus confirmation role split",
            ],
            "external_authority_blockers": [],
        },
        "evidence": [{"source": "placeholder", "claim": "fixture evidence"}],
    }


@dataclass(frozen=True)
class DatasetContractFixture:
    profile_path: Path
    draft_path: Path
    critique_path: Path
    contract_path: Path


def build_frozen_dataset_contract(
    root: Path,
    *,
    profile: dict[str, Any],
) -> DatasetContractFixture:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset_id = str(profile["dataset"]["id"])

    evidence_source = root / "profile-evidence.json"
    write_json(evidence_source, {"dataset_id": dataset_id, "status": "validated"})
    normalized_profile = deepcopy(profile)
    normalized_profile["constraints"].pop("requires_human_decision", None)
    normalized_profile["constraints"].setdefault(
        "requires_research_design_decision", []
    )
    normalized_profile["constraints"].setdefault("external_authority_blockers", [])
    normalized_profile["evidence"] = [
        {
            "source": str(evidence_source),
            "sha256": sha256_path(evidence_source),
            "claim": "hash-bound deterministic fixture evidence",
        }
    ]
    profile_path = root / "dataset_profile.json"
    write_json(profile_path, normalized_profile)

    canonical_path = root / "canonical_search_space.json"
    registry_path = root / "component_registry.json"
    evidence_db_path = root / "literature_evidence.sqlite"
    canonical_space = {
        "component_count": 3,
        "dimensions": {
            "preprocessing": [
                {"component_id": "bandpass_fixture", "execution_status": "not_activated"}
            ],
            "features": [
                {"component_id": "csp_fixture", "execution_status": "not_activated"}
            ],
            "models": [
                {"component_id": "lda_fixture", "execution_status": "not_activated"}
            ],
        },
    }
    write_json(
        canonical_path,
        {"dataset_id": dataset_id, "canonical_space": canonical_space},
    )
    write_json(registry_path, {"registry_id": "fixture-registry-v1"})
    evidence_db_path.write_bytes(b"deterministic-fixture-evidence-ledger\n")
    provenance = {
        "dataset_profile": {
            "path": str(profile_path),
            "sha256": sha256_path(profile_path),
        },
        "canonical_search_space": {
            "path": str(canonical_path),
            "sha256": sha256_path(canonical_path),
        },
        "component_registry": {
            "path": str(registry_path),
            "sha256": sha256_path(registry_path),
        },
        "evidence_db": {
            "path": str(evidence_db_path),
            "sha256": sha256_path(evidence_db_path),
        },
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
        "dataset_hard_constraints": [],
        "deferred_to_downstream_agents": {
            "research_design_decisions": [],
            "component_decisions": [],
            "external_authority_blockers": list(
                normalized_profile["constraints"].get(
                    "external_authority_blockers", []
                )
            ),
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
        "rationale": "The deterministic fixture contract is outcome-blind and in scope.",
        "source_draft": {
            "path": str(draft_path),
            "sha256": sha256_path(draft_path),
        },
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
                "source_draft": {
                    "path": str(draft_path),
                    "sha256": sha256_path(draft_path),
                },
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
    return DatasetContractFixture(
        profile_path=profile_path,
        draft_path=draft_path,
        critique_path=critique_path,
        contract_path=contract_path,
    )


def autonomy_envelope(
    contract_path: Path,
    *,
    dataset_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "envelope_id": f"{dataset_id}-autonomy-v2",
        "status": "authorized",
        "objective": {
            "research_question": "Find a reliable individualized pipeline efficiently.",
            "paradigm": "motor_imagery",
            "conclusion_scope": "internal_evidence_report",
        },
        "dataset": {
            "dataset_id": dataset_id,
            "dataset_level_contract_path": str(Path(contract_path).resolve()),
            "dataset_level_contract_sha256": sha256_path(contract_path),
        },
        "resource_budget": {
            "max_research_cycles": 12,
            "max_candidate_executions": 24,
            "max_compute_seconds": 3600,
            "max_api_tokens": 100000,
            "max_paid_cost": 10.0,
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
