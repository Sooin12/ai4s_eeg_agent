"""One-shot, fail-closed access to a frozen confirmation role."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bci_autodiscovery.agents.pipeline_lock_critic import (
    PipelineLockCriticError,
    validate_pipeline_lock,
    validate_pipeline_lock_critique,
)
from bci_autodiscovery.pipelines import DeterministicPipelineExecutor
from bci_autodiscovery.profiling.subject_measurements import EpochSession
from bci_autodiscovery.workflow.autonomy import (
    load_autonomy_envelope,
    load_json_object,
    sha256_path,
)


class ConfirmationAccessError(RuntimeError):
    """Raised when frozen confirmation cannot be opened under the frozen contracts."""


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    """Create an audit artifact exactly once without overwriting prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise ConfirmationAccessError(f"Audit artifact already exists: {path}") from exc


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OneShotConfirmationController:
    """Fit before access, consume the access token, then evaluate without refitting."""

    def __init__(
        self,
        *,
        search_executor: DeterministicPipelineExecutor,
        confirmation_loader: Callable[[], Sequence[EpochSession]],
        pipeline_lock_path: Path,
        lock_critique_path: Path,
        frozen_protocol_path: Path,
        autonomy_envelope_path: Path,
        access_record_path: Path,
        confirmation_result_path: Path,
    ) -> None:
        self.search_executor = search_executor
        self.confirmation_loader = confirmation_loader
        self.pipeline_lock_path = Path(pipeline_lock_path).expanduser().resolve()
        self.lock_critique_path = Path(lock_critique_path).expanduser().resolve()
        self.frozen_protocol_path = Path(frozen_protocol_path).expanduser().resolve()
        self.autonomy_envelope_path = Path(autonomy_envelope_path).expanduser().resolve()
        self.access_record_path = Path(access_record_path).expanduser().resolve()
        self.confirmation_result_path = Path(confirmation_result_path).expanduser().resolve()

    def confirm(self) -> dict[str, Any]:
        """Consume the single access only after all outcome-blind gates have passed."""

        if self.access_record_path.exists() or self.confirmation_result_path.exists():
            raise ConfirmationAccessError(
                "Frozen confirmation has already been accessed or produced a terminal result"
            )

        lock = load_json_object(self.pipeline_lock_path)
        critique = load_json_object(self.lock_critique_path)
        protocol = load_json_object(self.frozen_protocol_path)
        if protocol.get("status") != "frozen_autonomous":
            raise ConfirmationAccessError("Confirmation requires a frozen autonomous protocol")
        dataset_id = str(protocol.get("dataset_id") or "")
        envelope = load_autonomy_envelope(
            self.autonomy_envelope_path,
            expected_dataset_id=dataset_id,
        )
        if not envelope["permissions"].get("allow_first_confirmation_access"):
            raise ConfirmationAccessError(
                "AutonomyEnvelope does not authorize first confirmation access"
            )

        lock_hash = sha256_path(self.pipeline_lock_path)
        critique_hash = sha256_path(self.lock_critique_path)
        try:
            validate_pipeline_lock(lock, protocol=protocol, envelope=envelope)
            validate_pipeline_lock_critique(
                critique,
                lock=lock,
                lock_sha256=lock_hash,
                deterministic_validation_passed=True,
            )
        except (PipelineLockCriticError, KeyError, TypeError, ValueError) as exc:
            raise ConfirmationAccessError(f"Pipeline lock gate failed: {exc}") from exc
        if critique.get("verdict") != "pass":
            raise ConfirmationAccessError("Pipeline lock critic did not pass the lock")
        source_lock = critique.get("source_lock") or {}
        try:
            reviewed_path = Path(str(source_lock.get("path"))).expanduser().resolve()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ConfirmationAccessError("Critique has an invalid source lock path") from exc
        if reviewed_path != self.pipeline_lock_path or source_lock.get("sha256") != lock_hash:
            raise ConfirmationAccessError("Critique is not bound to the exact pipeline lock file")
        if critique.get("confirmation_results_available") is not False:
            raise ConfirmationAccessError("Pipeline lock review was not outcome blind")

        search_ids = [session.session_id for session in self.search_executor.sessions]
        expected_search_ids = [
            str(item) for item in protocol["data_roles"]["pipeline_search_and_lock"]
        ]
        if len(search_ids) != len(set(search_ids)) or set(search_ids) != set(expected_search_ids):
            raise ConfirmationAccessError("Search executor sessions differ from the frozen role")
        if self.search_executor.subject_id != lock.get("subject_id"):
            raise ConfirmationAccessError("Search executor and pipeline lock subjects differ")

        # All learned state is fitted before the confirmation access token is consumed.
        fitted = self.search_executor.fit(lock["selected_pipeline"])
        if fitted.spec != lock["selected_pipeline"]:
            raise ConfirmationAccessError("Fitted model specification differs from pipeline lock")

        access_record = {
            "schema_version": "1.0",
            "access_id": f"confirmation-access-{lock['lock_id']}",
            "status": "consumed",
            "access_count": 1,
            "consumed_at_utc": _utc_now(),
            "dataset_id": dataset_id,
            "subject_id": lock["subject_id"],
            "protocol_id": protocol["protocol_id"],
            "lock_id": lock["lock_id"],
            "model_sha256": fitted.model_sha256,
            "expected_confirmation_session_ids": [
                str(item) for item in protocol["data_roles"]["frozen_confirmation"]
            ],
            "source_contracts": {
                "pipeline_lock": {
                    "path": str(self.pipeline_lock_path),
                    "sha256": lock_hash,
                },
                "pipeline_lock_critique": {
                    "path": str(self.lock_critique_path),
                    "sha256": critique_hash,
                },
                "frozen_protocol": {
                    "path": str(self.frozen_protocol_path),
                    "sha256": sha256_path(self.frozen_protocol_path),
                },
                "autonomy_envelope": {
                    "path": str(self.autonomy_envelope_path),
                    "sha256": sha256_path(self.autonomy_envelope_path),
                },
            },
            "confirmation_outcomes_observed_before_record": False,
            "search_reopen_allowed": False,
            "retry_allowed_after_failure": False,
        }
        _write_json_exclusive(self.access_record_path, access_record)

        try:
            confirmation_sessions = tuple(self.confirmation_loader())
            expected_confirmation_ids = access_record["expected_confirmation_session_ids"]
            actual_confirmation_ids = [session.session_id for session in confirmation_sessions]
            if (
                not confirmation_sessions
                or len(actual_confirmation_ids) != len(set(actual_confirmation_ids))
                or set(actual_confirmation_ids) != set(expected_confirmation_ids)
            ):
                raise ConfirmationAccessError(
                    "Loaded confirmation sessions differ from the frozen confirmation role"
                )
            evaluation = self.search_executor.evaluate_fitted(
                fitted,
                sessions=confirmation_sessions,
                data_role="frozen_confirmation",
            )
            primary_metric = str(protocol["evaluation"]["primary_metric"])
            if primary_metric not in evaluation["metrics"]:
                raise ConfirmationAccessError(
                    f"Confirmation lacks frozen primary metric {primary_metric!r}"
                )
            search_score = float(lock["selected_search_score"])
            confirmation_score = float(evaluation["metrics"][primary_metric])
            result = {
                "schema_version": "1.0",
                "confirmation_id": f"confirmation-{lock['lock_id']}",
                "status": "completed_one_shot",
                "dataset_id": dataset_id,
                "subject_id": lock["subject_id"],
                "protocol_id": protocol["protocol_id"],
                "lock_id": lock["lock_id"],
                "primary_metric": primary_metric,
                "search_score": search_score,
                "confirmation_score": confirmation_score,
                "confirmation_minus_search": confirmation_score - search_score,
                "evaluation": evaluation,
                "fitted_pipeline": fitted.to_dict(),
                "access_record": {
                    "path": str(self.access_record_path),
                    "sha256": sha256_path(self.access_record_path),
                },
                "selection_or_refit_after_confirmation": False,
                "search_reopened": False,
                "retry_allowed": False,
            }
            _write_json_exclusive(self.confirmation_result_path, result)
            return result
        except Exception as exc:
            failure = {
                "schema_version": "1.0",
                "confirmation_id": f"confirmation-{lock['lock_id']}",
                "status": "confirmation_failed_after_access_consumed",
                "dataset_id": dataset_id,
                "subject_id": lock["subject_id"],
                "protocol_id": protocol["protocol_id"],
                "lock_id": lock["lock_id"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "access_record": {
                    "path": str(self.access_record_path),
                    "sha256": sha256_path(self.access_record_path),
                },
                "search_reopened": False,
                "retry_allowed": False,
            }
            _write_json_exclusive(self.confirmation_result_path, failure)
            raise

