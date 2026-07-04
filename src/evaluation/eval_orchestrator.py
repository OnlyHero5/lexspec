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
    load_predictions, load_gold_records, load_predictions_as_triplets,
    parse_trees_for_clauses, align_experiment_predictions,
    load_validations_aligned,
)
from src.evaluation.alignment import (
    ClauseAlignmentError,
    align_to_gold_order,
    records_to_triplets,
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
from src.utils.constraints import get_corpus_sampling_config, load_constraints_config

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

    # --- 加载金标（clause_id 为对齐基准）---
    gold_records = load_gold_records(gold_path)
    if not gold_records:
        logger.warning("No gold records — metrics will be empty.")
    gold_triplets = records_to_triplets(gold_records)

    # --- 加载并对齐各实验预测 ---
    logger.info("Loading and aligning predictions to gold clause_ids...")
    experiments: Dict[str, List[Dict]] = {}
    for name, filename in [
        ("baseline", "baseline.jsonl"),
        ("ours_dep", "ours_dep.jsonl"),
        ("ours_reflexion", "ours_reflexion.jsonl"),
    ]:
        pred_path = pred_dir / filename
        if not gold_records:
            experiments[name] = load_predictions(str(pred_path))
            continue
        if not pred_path.exists():
            logger.warning("Predictions file missing for %s — skipping", name)
            experiments[name] = []
            continue
        try:
            aligned_preds, _ = align_experiment_predictions(
                gold_path, str(pred_path), strict=True,
            )
            experiments[name] = aligned_preds
        except Exception as exc:
            raise RuntimeError(
                f"Failed to align '{name}' predictions to gold at '{gold_path}': {exc}"
            ) from exc

    # --- 加载测试集并按金标顺序对齐（语言学指标 / 分层检验）---
    testset: List[Dict] = []
    if gold_records and Path(testset_path).exists():
        raw_testset = read_jsonl(testset_path)
        try:
            testset = align_to_gold_order(
                gold_records, raw_testset, other_label="testset", strict=True,
            )
        except ClauseAlignmentError as exc:
            raise RuntimeError(
                f"Testset '{testset_path}' is not aligned with gold '{gold_path}': {exc}"
            ) from exc
    elif Path(testset_path).exists():
        testset = read_jsonl(testset_path)

    # --- 为对齐后的测试集解析 UD 树 ---
    trees: List[DependencyTree] = []
    if testset and gold_records and len(testset) == len(gold_records):
        metrics_dir = Path(output_dir) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        trees = parse_trees_for_clauses(
            testset,
            config_path=config_path,
            progress_path=str(metrics_dir / "parse_test_clauses.progress"),
        )

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

            if len(triplets) != len(gold_triplets):
                raise ValueError(
                    f"{name}: aligned predictions ({len(triplets)}) != "
                    f"gold ({len(gold_triplets)}) after clause_id join"
                )

            f1_metrics = compute_triplet_f1(
                predictions=triplets,
                gold=gold_triplets,
                party_aliases=party_aliases,
                constraints_path=f1_weights_path,
            )
            task_metrics[name] = f1_metrics

            ps_scores = compute_per_sample_f1(
                predictions=triplets,
                gold=gold_triplets,
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

            if len(triplets) != len(gold_triplets) or len(triplets) != len(trees):
                logger.warning(
                    "%s: length mismatch (pred=%d gold=%d trees=%d) — skipping",
                    name, len(triplets), len(gold_triplets), len(trees),
                )
                linguistic_metrics[name] = {}
                continue

            n = len(triplets)

            # 加载验证结果以计算修正率（ours_dep/ours_reflex）。
            val_results: Optional[List[ValidationResult]] = None
            if name in ("ours_dep", "ours_reflexion"):
                val_path = pred_dir / f"{name}_validations.jsonl"
                if val_path.exists():
                    try:
                        val_results = load_validations_aligned(
                            gold_path, str(val_path), strict=True,
                        )
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

        sampling_cfg = get_corpus_sampling_config(
            load_constraints_config(constraints_path), constraints_path,
        )
        bootstrap_seed = int(sampling_cfg["random_seed"])

        # 成对比较。
        try:
            sig = run_all_comparisons(
                experiment_results={
                    k: v for k, v in per_sample_scores.items() if v
                },
                n_resamples=10000,
                random_seed=bootstrap_seed,
            )
            significance_results = sig
        except Exception as exc:
            logger.error("Significance testing failed: %s", exc)
            significance_results = {"error": str(exc)}

        # 按现象分层显著性（三组消融成对比较）。
        if testset and all("phenomena" in c for c in testset):
            stratified_results: Dict[str, Any] = {}
            phen_names = ["passive", "conditional", "relative_clause",
                          "long_distance", "negation"]
            labels = [
                _primary_phenomenon(c.get("phenomena", {}))
                for c in testset[: len(gold_records)]
            ]
            stratified_pairs = [
                ("baseline", "ours_dep"),
                ("baseline", "ours_reflexion"),
                ("ours_dep", "ours_reflexion"),
            ]
            for exp_a, exp_b in stratified_pairs:
                if (
                    exp_a not in per_sample_scores
                    or exp_b not in per_sample_scores
                    or len(per_sample_scores[exp_a]) != len(labels)
                ):
                    continue
                pair_key = f"{exp_a}_vs_{exp_b}"
                stratified_results[pair_key] = {}
                for phen in progress_bar(
                    phen_names,
                    desc=f"Stratified {pair_key}",
                    unit="phen",
                ):
                    try:
                        strat = stratified_significance(
                            scores_a=per_sample_scores[exp_a],
                            scores_b=per_sample_scores[exp_b],
                            phenomenon_labels=labels,
                            phenomenon_name=phen,
                            n_resamples=10000,
                            random_seed=bootstrap_seed,
                        )
                        stratified_results[pair_key][phen] = strat
                    except Exception as exc:
                        logger.debug(
                            "Stratified significance failed for '%s' %s: %s",
                            pair_key, phen, exc,
                        )
                        stratified_results[pair_key][phen] = {"error": str(exc)}
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
