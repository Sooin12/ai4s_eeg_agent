"""Deterministic freeze gate for critic-approved autonomous protocols."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bci_autodiscovery.agents.protocol_critic import validate_protocol_critique
from bci_autodiscovery.agents.research_protocol import (
    _load_authorized_research_context,
    validate_research_protocol_authority_bindings,
    validate_research_protocol_proposal,
)

from .autonomy import load_json_object, sha256_path
from .protocol_artifacts import atomic_json


class AutonomousProtocolFreezeError(ValueError):
    pass


def _assert_binding(
    binding: Any,
    *,
    expected_path: Path,
    expected_hash: str,
    field: str,
) -> None:
    if not isinstance(binding, dict):
        raise AutonomousProtocolFreezeError(f"Protocol lacks {field} binding")
    bound_path = Path(str(binding.get("path") or "")).expanduser().resolve()
    if bound_path != expected_path or binding.get("sha256") != expected_hash:
        raise AutonomousProtocolFreezeError(f"Protocol {field} binding does not match")


def freeze_autonomous_protocol(
    *,
    proposal_path: Path,
    critique_path: Path,
    dataset_level_contract_path: Path,
    autonomy_envelope_path: Path,
    expected_dataset_level_contract_sha256: str,
    expected_autonomy_envelope_sha256: str,
    expected_proposal_sha256: str,
    expected_critique_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze a protocol without human approval after independent, outcome-blind review."""

    proposal_path = Path(proposal_path).expanduser().resolve()
    critique_path = Path(critique_path).expanduser().resolve()
    (
        contract_path,
        contract,
        profile_path,
        profile,
        envelope_path,
        envelope,
    ) = _load_authorized_research_context(
        dataset_level_contract_path=dataset_level_contract_path,
        autonomy_envelope_path=autonomy_envelope_path,
    )
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise AutonomousProtocolFreezeError(
            f"Refusing to overwrite frozen protocol: {destination}"
        )

    proposal = load_json_object(proposal_path)
    critique = load_json_object(critique_path)
    contract_hash = sha256_path(contract_path)
    profile_hash = sha256_path(profile_path)
    envelope_hash = sha256_path(envelope_path)
    proposal_hash = sha256_path(proposal_path)
    critique_hash = sha256_path(critique_path)
    expected_hashes = {
        "DatasetLevelContract": (
            contract_hash,
            expected_dataset_level_contract_sha256,
        ),
        "AutonomyEnvelope": (envelope_hash, expected_autonomy_envelope_sha256),
        "proposal": (proposal_hash, expected_proposal_sha256),
        "critique": (critique_hash, expected_critique_sha256),
    }
    changed = sorted(
        name
        for name, (observed, expected) in expected_hashes.items()
        if observed != expected
    )
    if changed:
        raise AutonomousProtocolFreezeError(
            f"Protocol freeze inputs changed after stage completion: {changed}"
        )

    validate_research_protocol_proposal(
        proposal,
        dataset_contract=contract,
        profile=profile,
        envelope=envelope,
    )
    validate_research_protocol_authority_bindings(
        proposal,
        dataset_contract_path=contract_path,
        autonomy_envelope_path=envelope_path,
    )
    validate_protocol_critique(
        critique,
        proposal=proposal,
        proposal_sha256=proposal_hash,
        deterministic_validation_passed=True,
    )
    if critique.get("verdict") != "pass":
        raise AutonomousProtocolFreezeError(
            f"Protocol critic verdict is {critique.get('verdict')!r}, not pass"
        )
    _assert_binding(
        proposal.get("dataset_level_contract"),
        expected_path=contract_path,
        expected_hash=contract_hash,
        field="dataset_level_contract",
    )
    _assert_binding(
        critique.get("source_proposal"),
        expected_path=proposal_path,
        expected_hash=proposal_hash,
        field="source_proposal",
    )

    frozen = json.loads(json.dumps(proposal))
    frozen["status"] = "frozen_autonomous"
    frozen["activation_state"] = {
        "protocol_frozen": True,
        "session_role_contract_activated": True,
        "raw_data_accessed": False,
        "confirmation_accessed": False,
        "pipeline_execution_started": False,
    }
    frozen["session_roles"] = frozen["data_roles"]
    frozen["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    frozen["autonomous_freeze"] = {
        "mechanism": "planner_critic_deterministic_validator",
        "human_itemized_approval_used": False,
        "proposal": {
            "path": str(proposal_path),
            "sha256": expected_proposal_sha256,
        },
        "critique": {
            "path": str(critique_path),
            "sha256": expected_critique_sha256,
            "review_id": critique["review_id"],
            "verdict": critique["verdict"],
        },
        "dataset_level_contract": {
            "path": str(contract_path),
            "sha256": expected_dataset_level_contract_sha256,
            "contract_id": contract["contract_id"],
        },
        "dataset_profile": {
            "path": str(profile_path),
            "sha256": profile_hash,
            "binding_source": "DatasetLevelContract.provenance.dataset_profile",
        },
        "autonomy_envelope": {
            "path": str(envelope_path),
            "sha256": expected_autonomy_envelope_sha256,
            "envelope_id": envelope["envelope_id"],
        },
    }
    atomic_json(destination, frozen, refuse_overwrite=True)
    return frozen


def validate_frozen_autonomous_protocol(
    *,
    frozen_protocol_path: Path,
    proposal_path: Path,
    critique_path: Path,
    dataset_level_contract_path: Path,
    autonomy_envelope_path: Path,
) -> dict[str, Any]:
    """Validate an orphaned freeze artifact before a recovery run adopts it."""

    frozen_path = Path(frozen_protocol_path).expanduser().resolve()
    proposal_path = Path(proposal_path).expanduser().resolve()
    critique_path = Path(critique_path).expanduser().resolve()
    if not frozen_path.is_file():
        raise AutonomousProtocolFreezeError(
            f"Frozen protocol does not exist: {frozen_path}"
        )
    (
        contract_path,
        contract,
        _profile_path,
        profile,
        envelope_path,
        envelope,
    ) = _load_authorized_research_context(
        dataset_level_contract_path=dataset_level_contract_path,
        autonomy_envelope_path=autonomy_envelope_path,
    )
    proposal = load_json_object(proposal_path)
    critique = load_json_object(critique_path)
    frozen = load_json_object(frozen_path)
    validate_research_protocol_proposal(
        proposal,
        dataset_contract=contract,
        profile=profile,
        envelope=envelope,
    )
    validate_research_protocol_authority_bindings(
        proposal,
        dataset_contract_path=contract_path,
        autonomy_envelope_path=envelope_path,
    )
    validate_protocol_critique(
        critique,
        proposal=proposal,
        proposal_sha256=sha256_path(proposal_path),
        deterministic_validation_passed=True,
    )
    if critique.get("verdict") != "pass":
        raise AutonomousProtocolFreezeError("Recovered freeze lacks a passing Critic")
    if frozen.get("status") != "frozen_autonomous":
        raise AutonomousProtocolFreezeError("Recovered frozen protocol has invalid status")
    if frozen.get("protocol_id") != proposal.get("protocol_id"):
        raise AutonomousProtocolFreezeError("Recovered freeze belongs to another protocol")
    if frozen.get("data_roles") != proposal.get("data_roles"):
        raise AutonomousProtocolFreezeError("Recovered freeze changed the frozen data roles")
    expected_activation = {
        "protocol_frozen": True,
        "session_role_contract_activated": True,
        "raw_data_accessed": False,
        "confirmation_accessed": False,
        "pipeline_execution_started": False,
    }
    if frozen.get("activation_state") != expected_activation:
        raise AutonomousProtocolFreezeError("Recovered freeze crossed an execution boundary")
    record = frozen.get("autonomous_freeze") or {}
    expected_bindings = {
        "proposal": (proposal_path, sha256_path(proposal_path)),
        "critique": (critique_path, sha256_path(critique_path)),
        "dataset_level_contract": (contract_path, sha256_path(contract_path)),
        "autonomy_envelope": (envelope_path, sha256_path(envelope_path)),
    }
    for field, (expected_path, expected_sha) in expected_bindings.items():
        binding = record.get(field) or {}
        observed_path = Path(str(binding.get("path") or "")).expanduser().resolve()
        if observed_path != expected_path or binding.get("sha256") != expected_sha:
            raise AutonomousProtocolFreezeError(
                f"Recovered freeze has invalid {field} authority binding"
            )
    return frozen
