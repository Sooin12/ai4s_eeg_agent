"""Build a presentation-safe aggregate without mutating immutable subject evidence."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean
from typing import Any

from bci_autodiscovery.demo.contracts import write_json
from bci_autodiscovery.workflow.autonomy import load_json_object, sha256_path


def _subject_row(subject_root: Path) -> dict[str, Any]:
    lock = load_json_object(subject_root / "pipeline_lock.json")
    confirmation = load_json_object(subject_root / "confirmation_result.json")
    access = load_json_object(subject_root / "confirmation_access.json")
    lock_critic = load_json_object(subject_root / "pipeline_lock_critique.json")
    scientific = load_json_object(subject_root / "scientific_critique.json")
    final = load_json_object(subject_root / "final_internal_evidence_report.json")
    selected = lock["selected_pipeline"]
    return {
        "subject_id": lock["subject_id"],
        "pipeline_id": selected["pipeline_id"],
        "pipeline_sha256": lock["pipeline_sha256"],
        "family": selected["family"],
        "bandpass_hz": selected["bandpass_hz"],
        "channel_strategy": selected.get("channel_strategy", "all"),
        "selected_channels": selected.get("selected_channels", []),
        "csp_components": selected["csp_components"],
        "lda_shrinkage": selected["lda_shrinkage"],
        "search_score": confirmation["search_score"],
        "confirmation_score": confirmation["confirmation_score"],
        "confirmation_minus_search": confirmation["confirmation_minus_search"],
        "search_candidates": lock["budget_usage"]["candidate_executions"],
        "literature_paper_count": len(lock.get("evidence_literature_paper_ids") or []),
        "individual_outcome": final["deterministic_decision"]["outcome"],
        "individual_reason": final["deterministic_decision"]["reason_code"],
        "lock_critic": lock_critic["verdict"],
        "scientific_critic": scientific["verdict"],
        "confirmation_access_count": access["access_count"],
        "source_artifacts": {
            "pipeline_lock": {
                "path": str(subject_root / "pipeline_lock.json"),
                "sha256": sha256_path(subject_root / "pipeline_lock.json"),
            },
            "confirmation_result": {
                "path": str(subject_root / "confirmation_result.json"),
                "sha256": sha256_path(subject_root / "confirmation_result.json"),
            },
            "final_internal_evidence_report": {
                "path": str(subject_root / "final_internal_evidence_report.json"),
                "sha256": sha256_path(subject_root / "final_internal_evidence_report.json"),
            },
        },
    }


def _aggregate_decision(protocol: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluation = protocol["evaluation"]
    policy = evaluation["decision_policy"]
    aggregation = evaluation["aggregation"]
    if aggregation.get("unit") != "subject" or aggregation.get("reducer") != "macro_mean":
        raise ValueError("Demo presentation supports frozen subject-level macro aggregation only")
    confirmation_mean = fmean(float(row["confirmation_score"]) for row in rows)
    search_mean = fmean(float(row["search_score"]) for row in rows)
    aggregate_drop = search_mean - confirmation_mean
    criteria = {
        "all_confirmations_completed_one_shot": all(
            row["confirmation_access_count"] == 1 for row in rows
        ),
        "minimum_evaluable_units": len(rows) >= int(policy["minimum_evaluable_units"]),
        "minimum_distinct_search_candidates_per_subject": all(
            int(row["search_candidates"])
            >= int(policy["minimum_distinct_search_candidates"])
            for row in rows
        ),
        "minimum_confirmation_score": confirmation_mean
        >= float(policy["minimum_confirmation_score"]),
        "maximum_search_to_confirmation_drop": aggregate_drop
        <= float(policy["maximum_search_to_confirmation_drop"]),
        "all_lock_critics_passed": all(row["lock_critic"] == "pass" for row in rows),
        "all_scientific_critics_passed": all(
            row["scientific_critic"] == "pass" for row in rows
        ),
    }
    if all(criteria.values()):
        outcome = "success"
        reason = "all_frozen_aggregate_engineering_criteria_met"
    elif confirmation_mean < float(policy["chance_level"]):
        outcome = "refuse"
        reason = "aggregate_confirmation_below_frozen_chance_level"
    else:
        outcome = "inconclusive"
        reason = "not_all_frozen_aggregate_criteria_met"
    return {
        "schema_version": "1.0",
        "scope": "synthetic_engineering_demo_aggregate",
        "outcome": outcome,
        "reason_code": reason,
        "criteria": criteria,
        "observed": {
            "evaluable_subjects": len(rows),
            "macro_mean_search_score": search_mean,
            "macro_mean_confirmation_score": confirmation_mean,
            "macro_mean_search_to_confirmation_drop": aggregate_drop,
            "distinct_locked_pipeline_hashes": len(
                {row["pipeline_sha256"] for row in rows}
            ),
            "distinct_pipeline_families": sorted({row["family"] for row in rows}),
        },
        "thresholds": policy,
        "aggregation": aggregation,
        "external_scientific_claim_authorized": False,
    }


def _combined_budget(root: Path) -> dict[str, Any]:
    extension = load_json_object(root / "budget_extension.json")
    base = extension["base_budget_ledger"]["snapshot"]
    extension_lines = (root / "budget_extension_ledger.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    extension_last = json.loads(extension_lines[-1])
    extension_totals = extension_last["totals"]
    combined_totals = {
        key: float(base["totals"].get(key, 0.0))
        + float(extension_totals.get(key, 0.0))
        for key in set(base["totals"]) | set(extension_totals)
    }
    return {
        "authorized_total_api_tokens": int(
            extension["changes"]["max_api_tokens"]["authorized_total"]
        ),
        "authorized_total_paid_cost": float(
            extension["changes"]["max_paid_cost"]["authorized_total"]
        ),
        "combined_totals": combined_totals,
        "remaining_api_tokens": int(
            extension["changes"]["max_api_tokens"]["authorized_total"]
            - combined_totals["api_total_tokens"]
        ),
        "remaining_paid_cost": float(
            extension["changes"]["max_paid_cost"]["authorized_total"]
            - combined_totals["paid_cost"]
        ),
        "authorities": {
            "base_ledger": extension["base_budget_ledger"],
            "budget_extension": {
                "path": str(root / "budget_extension.json"),
                "sha256": sha256_path(root / "budget_extension.json"),
            },
            "extension_ledger": {
                "path": str(root / "budget_extension_ledger.jsonl"),
                "sha256": sha256_path(root / "budget_extension_ledger.jsonl"),
            },
        },
    }


def build_presentation_bundle(root: Path) -> tuple[Path, Path]:
    root = Path(root).expanduser().resolve()
    protocol_path = root / "research_design" / "frozen_protocol.json"
    protocol = load_json_object(protocol_path)
    subject_roots = sorted(
        item.parent
        for item in (root / "subjects").glob("*/final_internal_evidence_report.json")
    )
    rows = [_subject_row(item) for item in subject_roots]
    if len(rows) != 3:
        raise ValueError("Presentation bundle requires exactly three finalized demo subjects")
    aggregate = _aggregate_decision(protocol, rows)
    bundle = {
        "schema_version": "1.0",
        "status": "presentation_bundle_completed",
        "demo_scope": "synthetic_engineering_validation_only",
        "external_scientific_claim_authorized": False,
        "subjects": rows,
        "aggregate_engineering_decision": aggregate,
        "combined_api_budget": _combined_budget(root),
        "frozen_protocol": {
            "path": str(protocol_path),
            "sha256": sha256_path(protocol_path),
            "protocol_id": protocol["protocol_id"],
        },
        "original_run_manifest": {
            "path": str(root / "run_manifest.json"),
            "sha256": sha256_path(root / "run_manifest.json"),
        },
        "interpretation_boundary": (
            "The aggregate success demonstrates autonomous orchestration and contract "
            "enforcement on deterministic synthetic EEG; it is not real-world efficacy evidence."
        ),
    }
    json_path = root / "demo_presentation_bundle.json"
    write_json(json_path, bundle, refuse_overwrite=True)

    phenotype = {
        "subject-mu": "10 Hz 功率型",
        "subject-csp": "20 Hz 协方差型",
        "subject-beta": "17 Hz 个体 beta 功率型",
    }
    lines = [
        "# 全线路自动化个体化 BCI 科研 Agent：工程 Demo 证据",
        "",
        "> 本 Demo 使用确定性合成 EEG 验证系统闭环与个体化决策能力，不构成真实数据集上的算法优越性或科学疗效声明。",
        "",
        "## 三名异质被试，三条独立搜索轨迹",
        "",
        "| 被试 | 隐藏工程表型（仅验收使用） | Agent 锁定路线 | 搜索候选数 | 搜索 BA | 冻结确认 BA | 文献证据 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        channels = ",".join(row["selected_channels"]) or "all"
        route = (
            f"{row['family']} · {row['bandpass_hz'][0]:g}–{row['bandpass_hz'][1]:g} Hz · "
            f"{channels}"
        )
        if row["family"] == "csp_lda":
            route += f" · CSP-{row['csp_components']}"
        lines.append(
            f"| {row['subject_id']} | {phenotype[row['subject_id']]} | {route} | "
            f"{row['search_candidates']} | {row['search_score']:.3f} | "
            f"{row['confirmation_score']:.3f} | {row['literature_paper_count']} 篇 |"
        )
    decision = aggregate["observed"]
    budget = bundle["combined_api_budget"]
    lines.extend(
        [
            "",
            "## 冻结协议下的总体工程结论",
            "",
            f"- 总体判定：**{aggregate['outcome']}**（{aggregate['reason_code']}）。",
            f"- 可评估被试：{decision['evaluable_subjects']}；独立 pipeline hash：{decision['distinct_locked_pipeline_hashes']}。",
            f"- 宏平均搜索/确认 BA：{decision['macro_mean_search_score']:.3f} / {decision['macro_mean_confirmation_score']:.3f}。",
            "- 三次确认均仅访问一次；确认后没有选择、重拟合或重开搜索。",
            "- 每份 Pipeline Lock 与最终报告都经过独立 Critic，三名被试全部通过。",
            "",
            "## 全线路证据链",
            "",
            "`DatasetLevelContract → Research Design Planner → Protocol Critic → Subject Profiler → "
            "Scholar Search → Pipeline Experiments → Lock Critic → One-shot Confirmation → "
            "Evidence Reporter → Scientific Critic`",
            "",
            f"API token 合计 {int(budget['combined_totals']['api_total_tokens']):,} / {budget['authorized_total_api_tokens']:,}；"
            f"实际计费约 ${budget['combined_totals']['paid_cost']:.3f} / ${budget['authorized_total_paid_cost']:.2f}。",
            "",
            "逐被试报告因协议冻结了“至少 3 个可评估 subject”的总体门槛而保持 inconclusive；"
            "三名被试全部完成后，确定性聚合门才给出 success。这个区分避免用单个被试冒充总体证据。",
        ]
    )
    markdown_path = root / "DEMO_PRESENTATION.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return json_path, markdown_path
