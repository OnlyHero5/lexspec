"""
评估编排器：全部实验的双轨评估。

编排三条评估轨道：
  - 轨道 1：任务指标（加权三元组 F1）
  - 轨道 2：语言学指标（依存合法性、被动恢复等）
  - 轨道 3：显著性检验（配对 bootstrap、Wilcoxon、分层）
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.extraction.schema import (
    LegalTriplet, DependencyTree, ValidationResult,
)
from src.evaluation.data_loading import (
    load_predictions, load_gold_triplets, load_predictions_as_triplets,
    parse_trees_for_testset,
)
from src.evaluation.reporting import (
    _primary_phenomenon, _write_summary_csv, _print_comparison_table,
)
from src.evaluation.normalization import load_party_aliases
from src.evaluation.triplet_f1 import (
    compute_triplet_f1,
)
from src.evaluation.field_f1 import (
    compute_per_sample_f1,
)
from src.evaluation.linguistic_metrics import (
    compute_all_linguistic_metrics,
)
from src.evaluation.significance import (
    run_all_comparisons, stratified_significance,
)
from src.utils.progress import progress_bar
from src.utils.logging import get_logger
from src.utils.io import read_jsonl, write_json

logger = get_logger(__name__)


def run_evaluation(
    predictions_dir: str,
    gold_path: str,
    testset_path: str,
    output_dir: str,
    config_path: str = "configs/model.yaml",
    constraints_path: str = "configs/constraints.yaml",
) -> None:
    """编排全部实验的双轨评估。

    执行三条评估轨道，并将结果保存到 ``output_dir/metrics/``。

    参数:
        predictions_dir:  含 ``baseline.jsonl``、``ours_dep.jsonl``、
            ``ours_reflexion.jsonl`` 的预测结果目录。
        gold_path:        金标三元组 JSONL 路径。
        testset_path:     测试集 JSONL 路径（含 ``text`` 字段，用于 UD 解析）。
        output_dir:       评估结果输出根目录；指标写入 ``output_dir/metrics/``。
        config_path:      ``model.yaml`` 路径，用于初始化 Stanza 流水线。
        constraints_path: ``constraints.yaml`` 路径，用于 F1 权重与当事方别名。

    返回:
        无（``None``）。结果以 JSON/CSV 文件写入 ``output_dir/metrics/``，
        并在日志中打印跨实验对比表。
    """
    pred_dir = Path(predictions_dir)

    # --- 加载预测文件 ---
    logger.info("Loading predictions...")
    baseline_preds = load_predictions(str(pred_dir / "baseline.jsonl"))
    ours_dep_preds = load_predictions(str(pred_dir / "ours_dep.jsonl"))
    ours_reflex_preds = load_predictions(str(pred_dir / "ours_reflexion.jsonl"))

    # --- 加载金标三元组 ---
    gold_triplets = load_gold_triplets(gold_path)

    # --- 加载测试集（用于解析树） ---
    testset = read_jsonl(testset_path) if Path(testset_path).exists() else []

    # --- 为测试集解析 UD 树（语言学指标所需） ---
    trees: List[DependencyTree] = []
    if testset:
        metrics_dir = Path(output_dir) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        trees = parse_trees_for_testset(
            testset_path=testset_path,
            config_path=config_path,
            progress_path=str(metrics_dir / "parse_test_clauses.progress"),
        )

    # --- 实验名与预测数据映射 ---
    experiments: Dict[str, List[Dict]] = {
        "baseline": baseline_preds,
        "ours_dep": ours_dep_preds,
        "ours_reflexion": ours_reflex_preds,
    }

    # --- 转换预测三元组 ---
    exp_triplets: Dict[str, List[LegalTriplet]] = {}
    for name, preds in experiments.items():
        exp_triplets[name] = load_predictions_as_triplets(preds)

    # --- 轨道 1：任务指标（三元组 F1） ---
    task_metrics: Dict[str, Dict] = {}
    per_sample_scores: Dict[str, List[float]] = {}

    has_gold = len(gold_triplets) > 0
    if has_gold:
        logger.info("Computing task metrics (weighted triplet F1)...")
        party_aliases = load_party_aliases(constraints_path)
        f1_weights_path = constraints_path
        for name in progress_bar(
            sorted(exp_triplets.keys()),
            desc="Task metrics F1",
            unit="experiment",
        ):
            triplets = exp_triplets[name]
            if not triplets:
                logger.warning("No predictions for %s -- skipping task metrics", name)
                task_metrics[name] = {}
                per_sample_scores[name] = []
                continue

            # 确保与金标等长。
            n = min(len(triplets), len(gold_triplets))
            if len(triplets) != len(gold_triplets):
                logger.warning(
                    "%s predictions (%d) != gold (%d) -- truncating to %d",
                    name, len(triplets), len(gold_triplets), n,
                )

            f1_metrics = compute_triplet_f1(
                predictions=triplets[:n],
                gold=gold_triplets[:n],
                party_aliases=party_aliases,
                constraints_path=f1_weights_path,
            )
            task_metrics[name] = f1_metrics

            # 逐样本 F1，供显著性检验。
            ps_scores = compute_per_sample_f1(
                predictions=triplets[:n],
                gold=gold_triplets[:n],
                party_aliases=party_aliases,
                constraints_path=constraints_path,
            )
            per_sample_scores[name] = ps_scores
    else:
        logger.warning(
            "No gold triplets available -- task metrics will be empty."
        )
        for name in experiments:
            task_metrics[name] = {}
            per_sample_scores[name] = []

    # --- 轨道 2：语言学指标 ---
    linguistic_metrics: Dict[str, Dict] = {}
    if trees and has_gold and len(trees) == len(gold_triplets):
        logger.info("Computing linguistic metrics...")

        for name in progress_bar(
            ["baseline", "ours_dep", "ours_reflexion"],
            desc="Linguistic metrics",
            unit="experiment",
        ):
            triplets = exp_triplets.get(name, [])
            if not triplets:
                linguistic_metrics[name] = {}
                continue

            n = min(len(triplets), len(gold_triplets), len(trees))

            # 加载验证结果以计算修正率（ours_dep/ours_reflex）。
            val_results: Optional[List[ValidationResult]] = None
            if name in ("ours_dep", "ours_reflexion"):
                val_path = pred_dir / f"{name}_validations.jsonl"
                if val_path.exists():
                    try:
                        from src.utils.io import load_pydantic_list
                        val_results = load_pydantic_list(str(val_path), ValidationResult)
                        val_results = val_results[:n]
                    except Exception as exc:
                        logger.debug("Could not load validation results: %s", exc)

            try:
                ling = compute_all_linguistic_metrics(
                    predictions=triplets[:n],
                    gold=gold_triplets[:n],
                    trees=trees[:n],
                    validation_results=val_results,
                )
                linguistic_metrics[name] = ling
            except Exception as exc:
                logger.error(
                    "Linguistic metrics computation failed for %s: %s", name, exc,
                )
                linguistic_metrics[name] = {"error": str(exc)}
    else:
        logger.warning(
            "UD trees not available or mismatched lengths -- "
            "linguistic metrics will be empty."
        )
        for name in experiments:
            linguistic_metrics[name] = {}

    # --- 轨道 3：显著性检验 ---
    significance_results: Dict[str, Any] = {}
    if len(per_sample_scores) >= 2 and all(
        len(scores) > 0 for scores in per_sample_scores.values()
    ):
        logger.info("Running significance tests...")

        # 成对比较。
        try:
            sig = run_all_comparisons(
                experiment_results={
                    k: v for k, v in per_sample_scores.items() if v
                },
                n_resamples=10000,
                random_seed=42,
            )
            significance_results = sig
        except Exception as exc:
            logger.error("Significance testing failed: %s", exc)
            significance_results = {"error": str(exc)}

        # 按现象分层显著性（若测试集含现象标签）。
        if testset and all("phenomena" in c for c in testset):
            stratified_results: Dict[str, Any] = {}
            phen_names = ["passive", "conditional", "relative_clause",
                          "long_distance", "negation"]
            # 为简洁仅比较 baseline vs ours_reflexion。
            if ("baseline" in per_sample_scores
                    and "ours_reflexion" in per_sample_scores
                    and len(per_sample_scores["baseline"]) == len(testset)):
                labels = [
                    _primary_phenomenon(c.get("phenomena", {}))
                    for c in testset[:len(per_sample_scores["baseline"])]
                ]
                for phen in progress_bar(
                    phen_names, desc="Stratified significance", unit="phen",
                ):
                    try:
                        strat = stratified_significance(
                            scores_a=per_sample_scores["baseline"],
                            scores_b=per_sample_scores["ours_reflexion"],
                            phenomenon_labels=labels,
                            phenomenon_name=phen,
                            n_resamples=10000,
                            random_seed=42,
                        )
                        stratified_results[phen] = strat
                    except Exception as exc:
                        logger.debug(
                            "Stratified significance failed for '%s': %s", phen, exc,
                        )
                        stratified_results[phen] = {"error": str(exc)}
            significance_results["stratified"] = stratified_results
    else:
        logger.warning(
            "Insufficient per-sample scores for significance testing."
        )

    # --- 保存全部结果 ---
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # 保存任务指标。
    write_json(str(metrics_dir / "task_metrics.json"), task_metrics)

    # 保存语言学指标。
    write_json(str(metrics_dir / "linguistic_metrics.json"), linguistic_metrics)

    # 保存显著性结果。
    write_json(str(metrics_dir / "significance.json"), significance_results)

    # 写入摘要 CSV。
    _write_summary_csv(task_metrics, linguistic_metrics, significance_results,
                       str(metrics_dir / "summary.csv"))

    # --- 打印综合比较表 ---
    _print_comparison_table(task_metrics, linguistic_metrics, significance_results)
