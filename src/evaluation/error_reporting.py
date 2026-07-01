"""
错误报告：单实验错误分析与跨实验比较。

提供单实验错误分析函数，并打印全部实验的比较表。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.extraction.schema import LegalTriplet, DependencyTree, ErrorCase
from src.evaluation.data_loading import load_predictions_as_triplets
from src.evaluation.error_analyzer import (
    classify_errors, save_error_cases,
)
from src.evaluation.error_summary import (
    error_distribution_report, generate_error_summary,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def analyze_experiment_errors(
    experiment_name: str,
    predictions: List[Dict],
    gold: List[LegalTriplet],
    trees: List[DependencyTree],
    output_dir: str,
    constraints_path: str = "configs/constraints.yaml",
) -> Dict[str, Any]:
    """对单个实验运行完整错误分析。

    步骤:
      1. 将预测转换为 LegalTriplet 对象。
      2. 运行 ``classify_errors()`` 生成 ErrorCase 对象。
      3. 计算错误分布统计。
      4. 将分类错误案例保存为 JSONL 文件。
      5. 返回供比较表使用的摘要字典。

    参数:
        experiment_name:  例如 "baseline"、"ours_dep"、"ours_reflexion"。
        predictions:      该实验的预测字典列表。
        gold:             金标准 LegalTriplet 列表。
        trees:            DependencyTree 对象列表（等长）。
        output_dir:       错误案例文件输出目录。

    返回:
        含错误分布统计与摘要的字典。
    """
    logger.info("Analyzing errors for experiment: %s", experiment_name)

    # 转换预测。
    pred_triplets = load_predictions_as_triplets(predictions)

    # 对齐长度。
    n = min(len(pred_triplets), len(gold), len(trees))
    if n == 0:
        logger.warning("No data for %s -- skipping error analysis", experiment_name)
        return {
            "experiment": experiment_name,
            "total_samples": 0,
            "error_count": 0,
            "error_rate": 0.0,
            "primary_distribution": {},
            "secondary_distribution": {},
        }
    pred_triplets = pred_triplets[:n]
    gold = gold[:n]
    trees_used = trees[:n]

    # 分类错误。
    error_cases: List[ErrorCase] = classify_errors(
        predictions=pred_triplets,
        gold=gold,
        trees=trees_used,
        constraints_path=constraints_path,
    )

    # 计算分布。
    dist = error_distribution_report(error_cases)

    # 保存分类错误案例。
    exp_error_dir = Path(output_dir) / experiment_name
    save_error_cases(error_cases, str(exp_error_dir))

    # 生成并打印摘要。
    summary_text = generate_error_summary(error_cases)
    print(f"\n{summary_text}")

    # 构建返回字典。
    return {
        "experiment": experiment_name,
        "total_samples": n,
        "error_count": len(error_cases),
        "error_rate": len(error_cases) / n if n > 0 else 0.0,
        "primary_distribution": dist.get("primary_distribution", {}),
        "secondary_distribution": dist.get("secondary_distribution", {}),
        "cross_tabulation": dist.get("cross_tabulation", {}),
        "most_common_patterns": dist.get("most_common_patterns", []),
    }


def print_error_comparison_table(
    results: Dict[str, Dict[str, Any]],
) -> None:
    """打印跨实验错误统计比较表。

    并排展示错误数、错误率及主类别分布，便于快速比较。

    参数:
        results:  实验名 -> 错误分析摘要字典 的映射。
    """
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS COMPARISON -- ALL EXPERIMENTS")
    print("=" * 80)

    # 各实验总体统计。
    print(f"\n{'Experiment':<20s} {'Samples':>8s} {'Errors':>8s} {'ErrorRate':>10s}")
    print("-" * 46)
    for name in ["baseline", "ours_dep", "ours_reflexion"]:
        r = results.get(name, {})
        if not r:
            continue
        print(
            f"{name:<20s} "
            f"{r.get('total_samples', 0):>8d} "
            f"{r.get('error_count', 0):>8d} "
            f"{r.get('error_rate', 0):>9.1%}"
        )

    # 主类别分布。
    print("\n--- Primary Category Distribution (Linguistic Phenomenon) ---")
    all_categories = sorted(set(
        cat for r in results.values()
        for cat in r.get("primary_distribution", {}).keys()
    ))
    header = f"{'Experiment':<20s}"
    for cat in all_categories:
        header += f" {cat:>18s}"
    print(header)
    print("-" * (20 + 20 * len(all_categories)))
    for name in ["baseline", "ours_dep", "ours_reflexion"]:
        r = results.get(name, {})
        if not r:
            continue
        line = f"{name:<20s}"
        for cat in all_categories:
            cnt = r.get("primary_distribution", {}).get(cat, 0)
            line += f" {cnt:>18d}"
        print(line)

    # 次类别分布。
    print("\n--- Secondary Category Distribution (Field Error Type) ---")
    all_fields = sorted(set(
        f for r in results.values()
        for f in r.get("secondary_distribution", {}).keys()
    ))
    header = f"{'Experiment':<20s}"
    for f in all_fields:
        header += f" {f:>18s}"
    print(header)
    print("-" * (20 + 20 * len(all_fields)))
    for name in ["baseline", "ours_dep", "ours_reflexion"]:
        r = results.get(name, {})
        if not r:
            continue
        line = f"{name:<20s}"
        for f in all_fields:
            cnt = r.get("secondary_distribution", {}).get(f, 0)
            line += f" {cnt:>18d}"
        print(line)

    print("=" * 80)
