from __future__ import annotations

import json
from pathlib import Path

import pytest

from bci_autodiscovery.agents.research_protocol import (
    create_research_protocol_planner_tools,
)
from bci_autodiscovery.workflow.autonomy import (
    AutonomyEnvelopeError,
    load_autonomy_envelope,
)
from bci_autodiscovery.workflow.dataset_contract import (
    DatasetLevelContractError,
    load_dataset_level_contract,
)
from tests.fixtures.contracts import (
    autonomy_envelope,
    build_frozen_dataset_contract,
    minimal_dataset_profile,
    write_json,
)


def _contracts(tmp_path: Path) -> tuple[Path, Path]:
    dataset_id = "contract-boundary-fixture"
    fixture = build_frozen_dataset_contract(
        tmp_path / "dataset-level",
        profile=minimal_dataset_profile(dataset_id),
    )
    envelope_path = tmp_path / "autonomy.json"
    write_json(
        envelope_path,
        autonomy_envelope(fixture.contract_path, dataset_id=dataset_id),
    )
    return fixture.contract_path, envelope_path


def test_only_exact_authorized_frozen_dataset_contract_enters_planner(
    tmp_path: Path,
) -> None:
    contract_path, envelope_path = _contracts(tmp_path)
    contract = load_dataset_level_contract(contract_path)
    envelope = load_autonomy_envelope(
        envelope_path,
        expected_dataset_id=contract["dataset_id"],
        expected_dataset_contract_path=contract_path,
    )

    tools, context = create_research_protocol_planner_tools(
        dataset_level_contract_path=contract_path,
        autonomy_envelope_path=envelope_path,
    )
    exposed = tools.execute("read_autonomous_research_context", {})

    assert envelope["dataset"]["dataset_level_contract_sha256"]
    assert context["dataset_level_contract_sha256"] == envelope["dataset"][
        "dataset_level_contract_sha256"
    ]
    assert exposed["dataset_level_contract"]["status"] == (
        "frozen_dataset_level_contract"
    )
    assert exposed["dataset_profile"]["dataset"]["id"] == contract["dataset_id"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "dataset_level_draft_awaiting_critic", "status"),
        ("schema_version", "1.0", "schema_version"),
    ],
)
def test_dataset_contract_rejects_nonfrozen_or_wrong_schema(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    contract_path, _ = _contracts(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract[field] = value
    write_json(contract_path, contract)

    with pytest.raises(DatasetLevelContractError, match=message):
        load_dataset_level_contract(contract_path)


def test_dataset_contract_rejects_critic_stage_or_provenance_tamper(
    tmp_path: Path,
) -> None:
    contract_path, _ = _contracts(tmp_path)
    original = json.loads(contract_path.read_text(encoding="utf-8"))

    critic_tamper = json.loads(json.dumps(original))
    critic_tamper["dataset_critic"]["verdict"] = "revise"
    write_json(contract_path, critic_tamper)
    with pytest.raises(DatasetLevelContractError, match="Critic.*pass"):
        load_dataset_level_contract(contract_path)

    stage_tamper = json.loads(json.dumps(original))
    stage_tamper["stage_boundary"]["session_roles_assigned"] = True
    write_json(contract_path, stage_tamper)
    with pytest.raises(DatasetLevelContractError, match="stage_boundary"):
        load_dataset_level_contract(contract_path)

    write_json(contract_path, original)
    profile_path = Path(original["provenance"]["dataset_profile"]["path"])
    profile_path.write_text(profile_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(DatasetLevelContractError, match="provenance.*integrity"):
        load_dataset_level_contract(contract_path)


def test_envelope_rejects_contract_path_hash_and_dataset_mismatch(
    tmp_path: Path,
) -> None:
    contract_path, envelope_path = _contracts(tmp_path)
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))

    envelope["dataset"]["dataset_level_contract_sha256"] = "0" * 64
    write_json(envelope_path, envelope)
    with pytest.raises(AutonomyEnvelopeError, match="integrity"):
        load_autonomy_envelope(envelope_path)

    envelope = autonomy_envelope(
        contract_path,
        dataset_id="another-dataset",
    )
    write_json(envelope_path, envelope)
    with pytest.raises(AutonomyEnvelopeError, match="another dataset"):
        load_autonomy_envelope(envelope_path)

    envelope = autonomy_envelope(contract_path, dataset_id="contract-boundary-fixture")
    write_json(envelope_path, envelope)
    other_path = tmp_path / "other-contract.json"
    other_path.write_bytes(contract_path.read_bytes())
    with pytest.raises(AutonomyEnvelopeError, match="different DatasetLevelContract path"):
        load_autonomy_envelope(
            envelope_path,
            expected_dataset_contract_path=other_path,
        )


def test_legacy_profile_only_envelope_cannot_enter_formal_research_design(
    tmp_path: Path,
) -> None:
    contract_path, envelope_path = _contracts(tmp_path)
    contract = load_dataset_level_contract(contract_path)
    profile_reference = contract["provenance"]["dataset_profile"]
    legacy = json.loads(envelope_path.read_text(encoding="utf-8"))
    legacy["schema_version"] = "1.0"
    legacy["dataset"] = {
        "dataset_id": contract["dataset_id"],
        "profile_path": profile_reference["path"],
        "profile_sha256": profile_reference["sha256"],
    }
    write_json(envelope_path, legacy)

    with pytest.raises(AutonomyEnvelopeError, match="schema_version"):
        load_autonomy_envelope(envelope_path)
