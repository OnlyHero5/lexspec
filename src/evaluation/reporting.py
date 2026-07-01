"""
评估报告：摘要 CSV 生成与格式化比较表。

为双轨评估（任务指标 + 语言学指标 + 显著性检验）
生成可读控制台输出与 CSV 摘要。
"""

from __future__ import annotations

import csv
from typing import Any, Dict, List

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _primary_phenomenon(phenomena: Dict[str, bool]) -> str:
    """确定子句的主现象标签。

    返回分层分析用的最具体现象标签。
    若多种现象并存，按优先级返回首个
    （passive > conditional > relative > long_dist > negation）。
    """
    order = ["passive", "conditional", "relative_clause",
             "long_distance", "negation"]
    for phen in order:
        if phenomena.get(phen, False):
            return phen
    return "other"


def _write_summary_csv(
    task_metrics: Dict[str, Dict],
    linguistic_metrics: Dict[str, Dict],
    significance: Dict[str, Any],
    output_path: str,
) -> None:
    """写入各实验关键指标的摘要 CSV。

    每行一个实验，列含总体 F1、各字段 F1、语言学指标与显著性指示。

    参数:
        task_metrics:       实验名 -> 任务指标字典。
        linguistic_metrics:  实验名 -> 语言学指标字典。
        significance:        显著性检验结果。
        output_path:         输出 CSV 路径。
    """
    rows: List[Dict[str, Any]] = []

    # 确定存在的实验集合。
    exp_names = sorted(
        set(task_metrics.keys())
        | set(linguistic_metrics.keys())
    )

    for name in exp_names:
        row: Dict[str, Any] = {"experiment": name}

        # 任务指标。
        tm = task_metrics.get(name, {})
        row["overall_f1"] = tm.get("overall_f1", None)
        row["subject_text_f1"] = tm.get("subject_text_f1", None)
        row["subject_role_acc"] = tm.get("subject_role_acc", None)
        row["predicate_f1"] = tm.get("predicate_f1", None)
        row["object_f1"] = tm.get("object_f1", None)
        row["condition_f1"] = tm.get("condition_f1", None)

        # 语言学指标。
        lm = linguistic_metrics.get(name, {})
        ling_summary = lm.get("summary", {})
        quality = ling_summary.get("linguistic_quality_indicators", {})
        passive = ling_summary.get("passive_voice_handling", {})
        validator = ling_summary.get("validator_performance", {})

        row["dependency_legality"] = quality.get("dependency_legality", None)
        row["condition_iou"] = quality.get("condition_boundary_iou", None)
        row["passive_recovery_acc"] = passive.get("recovery_accuracy", None)
        row["false_agent_rate"] = passive.get("false_agent_rate", None)
        row["correction_valid_rate"] = validator.get("valid_rate", None)
        row["correction_success_rate"] = validator.get("correction_success_rate", None)

        rows.append(row)

    if not rows:
        logger.warning("No data to write to summary CSV.")
        return

    # 写入 CSV。
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Summary CSV written to: %s", output_path)


def _print_comparison_table(
    task_metrics: Dict[str, Dict],
    linguistic_metrics: Dict[str, Dict],
    significance: Dict[str, Any],
) -> None:
    """向 stdout 打印格式化比较表。"""
    print("\n" + "=" * 72)
    print("LEXSPEC EXPERIMENT COMPARISON TABLE")
    print("=" * 72)

    # ---- 任务指标 ----
    print("\n--- Task Metrics (Weighted Triplet F1) ---")
    header = f"{'Experiment':<18s} {'Overall F1':>10s} {'Subj':>8s} {'Role':>8s} {'Pred':>8s} {'Obj':>8s} {'Cond':>8s}"
    print(header)
    print("-" * 68)
    for name, tm in task_metrics.items():
        if not tm:
            print(f"{name:<18s} {'--':>10s} {'--':>8s} {'--':>8s} {'--':>8s} {'--':>8s} {'--':>8s}")
            continue
        print(
            f"{name:<18s} "
            f"{tm.get('overall_f1', 0):>10.4f} "
            f"{tm.get('subject_text_f1', 0):>8.4f} "
            f"{tm.get('subject_role_acc', 0):>8.4f} "
            f"{tm.get('predicate_f1', 0):>8.4f} "
            f"{tm.get('object_f1', 0):>8.4f} "
            f"{tm.get('condition_f1', 0):>8.4f}"
        )

    # ---- 语言学指标 ----
    print("\n--- Linguistic Metrics ---")
    ling_header = f"{'Experiment':<18s} {'DepLegality':>12s} {'CondIoU':>10s} {'PassiveAcc':>10s} {'FalseAgent':>10s}"
    print(ling_header)
    print("-" * 60)
    for name, lm in linguistic_metrics.items():
        if not lm:
            print(f"{name:<18s} {'--':>12s} {'--':>10s} {'--':>10s} {'--':>10s}")
            continue
        passive = lm.get("passive_recovery", {})
        print(
            f"{name:<18s} "
            f"{lm.get('dependency_path_legality', 0):>12.4f} "
            f"{lm.get('condition_iou', 0):>10.4f} "
            f"{passive.get('recovery_accuracy', 0):>10.4f} "
            f"{passive.get('false_agent_rate', 0):>10.4f}"
        )

    # ---- 修正指标 ----
    print("\n--- Correction Analysis ---")
    corr_header = f"{'Experiment':<18s} {'ValidRate':>10s} {'CorrRate':>10s} {'ReflexRate':>10s} {'SuccessRate':>12s}"
    print(corr_header)
    print("-" * 60)
    for name, lm in linguistic_metrics.items():
        ca = lm.get("correction_analysis", {}) if lm else {}
        if not ca:
            print(f"{name:<18s} {'--':>10s} {'--':>10s} {'--':>10s} {'--':>12s}")
            continue
        print(
            f"{name:<18s} "
            f"{ca.get('valid_rate', 0):>10.4f} "
            f"{ca.get('corrected_rate', 0):>10.4f} "
            f"{ca.get('reflexion_rate', 0):>10.4f} "
            f"{ca.get('correction_success_rate', 0):>12.4f}"
        )

    # ---- 显著性 ----
    print("\n--- Significance Tests (Bootstrap, 10k resamples) ---")
    sig_comp = significance.get("comparisons", {})
    sig_matrix = significance.get("summary", {}).get("significance_matrix", {})
    sig_header = f"{'Comparison':<30s} {'MeanDiff':>10s} {'95% CI':>22s} {'p-value':>8s} {'Sig?':>6s}"
    print(sig_header)
    print("-" * 76)
    for comp_key, comp_data in sig_comp.items():
        bs = comp_data.get("bootstrap", {})
        if isinstance(bs, dict) and "mean_diff" in bs:
            print(
                f"{comp_key:<30s} "
                f"{bs['mean_diff']:>10.4f} "
                f"[{bs['ci_95_lower']:.4f}, {bs['ci_95_upper']:.4f}]  "
                f"{bs.get('p_value', 1):>8.4f} "
                f"{'YES' if bs.get('significant_at_0.05', False) else 'no':>6s}"
            )

    # --- 分层显著性 ---
    strat = significance.get("stratified", {})
    if strat:
        print("\n--- Stratified Significance (Baseline vs Ours-Reflexion by Phenomenon) ---")
        print(f"{'Phenomenon':<20s} {'N':>5s} {'MeanDiff':>10s} {'p-value':>8s} {'Sig?':>6s}")
        print("-" * 55)
        for phen, result in strat.items():
            if isinstance(result, dict) and "mean_diff" in result:
                print(
                    f"{phen:<20s} "
                    f"{result['subset_size']:>5d} "
                    f"{result['mean_diff']:>10.4f} "
                    f"{result['p_value']:>8.4f} "
                    f"{'YES' if result.get('significant_at_0.05', False) else 'no':>6s}"
                )

    print("\n" + "=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)
