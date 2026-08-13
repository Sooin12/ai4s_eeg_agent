"""Fail-closed finalization of independently reviewed internal evidence reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bci_autodiscovery.agents.scientific_critic import (
    ScientificCriticError,
    _load_report_sources,
    validate_scientific_critique,
)
from bci_autodiscovery.agents.evidence_reporter import validate_evidence_report
from bci_autodiscovery.workflow.autonomy import load_json_object, sha256_path


class EvidenceFinalizationError(RuntimeError):
    pass


def finalize_internal_evidence_report(
    *,
    evidence_report_path: Path,
    scientific_critique_path: Path,
    final_report_path: Path,
) -> dict[str, Any]:
    report_path = Path(evidence_report_path).expanduser().resolve()
    critique_path = Path(scientific_critique_path).expanduser().resolve()
    output_path = Path(final_report_path).expanduser().resolve()
    if output_path.exists():
        raise EvidenceFinalizationError("Final internal evidence report already exists")
    report = load_json_object(report_path)
    critique = load_json_object(critique_path)
    report_hash = sha256_path(report_path)
    try:
        loaded, decision = _load_report_sources(report)
        validate_evidence_report(
            report,
            protocol=loaded["frozen_protocol"],
            pipeline_lock=loaded["pipeline_lock"],
            confirmation_result=loaded["confirmation_result"],
            deterministic_decision=decision,
        )
        if report.get("deterministic_decision") != decision:
            raise EvidenceFinalizationError("Embedded deterministic decision is stale")
        validate_scientific_critique(
            critique,
            report=report,
            report_sha256=report_hash,
            deterministic_validation_passed=True,
        )
    except (ScientificCriticError, KeyError, TypeError, ValueError) as exc:
        raise EvidenceFinalizationError(f"Scientific finalization gate failed: {exc}") from exc
    if critique.get("verdict") != "pass":
        raise EvidenceFinalizationError("Scientific Critic did not pass the evidence report")
    source = critique.get("source_report") or {}
    if (
        Path(str(source.get("path"))).expanduser().resolve() != report_path
        or source.get("sha256") != report_hash
    ):
        raise EvidenceFinalizationError("Scientific critique is not bound to exact report file")
    final = json.loads(json.dumps(report))
    final["status"] = "finalized_internal_evidence_report"
    final["scientific_critique"] = {
        "path": str(critique_path),
        "sha256": sha256_path(critique_path),
        "verdict": "pass",
    }
    final["external_claim_authorized"] = False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(final, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise EvidenceFinalizationError("Final internal evidence report already exists") from exc
    return final
