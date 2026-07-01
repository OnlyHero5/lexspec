#!/usr/bin/env python3
"""
LexSpec 实验 07: 错误分析 —— 语言学错误分类
============================================

实验流水线第 7 步（依据规范第 8.2 节）。

本脚本对三种实验变体进行全面的语言学错误分析。对每个实验：

  1. 将预测三元组与金标准三元组进行对比。
  2. 使用两级分类体系对错误进行分类：
     - **一级类别**（语言学现象）：
       passive_voice、conditional_boundary、relative_clause、
       long_distance_dependency、negation_exception、other
     - **二级类别**（字段错误类型）：
       subject、role、predicate、object、condition_omission、
       condition_overextension
  3. 生成中英双语语言学解释，引用具体 UD 依存关系。
  4. 输出含交叉表的误差分布报告。
  5. 将分类后的错误案例保存至 ``outputs/error_cases/``。

用法
----
    python experiments/step_07_analyze_errors.py \\
        --config configs/model.yaml \\
        --output-dir outputs/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Any

import yaml

# --- 将项目根目录加入 sys.path ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.extraction.schema import (
    LegalTriplet, DependencyTree, ValidationResult, ErrorCase,
    ErrorCategory, FieldErrorType,
)
from src.evaluation.data_loading import (
    load_predictions, load_gold_triplets, parse_trees_for_testset,
)
from src.evaluation.error_reporting import (
    analyze_experiment_errors, print_error_comparison_table,
)
from src.utils.logging import setup_logging, get_logger
from src.utils.io import read_jsonl, write_jsonl, write_json, load_pydantic_list
from src.utils.progress import progress_bar

logger = get_logger(__name__)


def main() -> None:
    """主入口：对所有实验运行错误分析。"""
    arg_parser = argparse.ArgumentParser(
        description="LexSpec 实验 07: 语言学错误分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arg_parser.add_argument(
        "--config",
        type=str,
        default="configs/model.yaml",
        help="模型配置 YAML 文件路径",
    )
    arg_parser.add_argument(
        "--predictions-dir",
        type=str,
        default="outputs/predictions",
        help="预测 JSONL 文件所在目录",
    )
    arg_parser.add_argument(
        "--gold",
        type=str,
        default="data/processed/gold_triplets.jsonl",
        help="金标准三元组 JSONL 文件路径",
    )
    arg_parser.add_argument(
        "--testset",
        type=str,
        default="data/processed/lexspec_100.jsonl",
        help="测试集 JSONL 文件路径（用于 UD 解析）",
    )
    arg_parser.add_argument(
        "--constraints",
        type=str,
        default="configs/constraints.yaml",
        help="约束配置 YAML 文件路径",
    )
    arg_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="输出文件目录",
    )
    args = arg_parser.parse_args()

    # --- 初始化日志 ---
    import logging as _logging
    setup_logging(log_dir=str(Path(args.output_dir) / "logs"), level=_logging.INFO)

    # --- 加载金标准三元组 ---
    gold_triplets = load_gold_triplets(args.gold)
    if not gold_triplets:
        logger.error(
            "Cannot run error analysis without gold triplets. "
            "Generate gold labels first via the annotation pipeline."
        )
        print(
            "\nWARNING: Gold triplets not found at '%s'.\n"
            "Error analysis requires gold-standard labels to compare against.\n"
            "Run the annotation pipeline (src/annotation/) to generate gold triplets,\n"
            "or set --gold to point at an existing gold file.\n"
            % args.gold
        )
        sys.exit(0)  # 非致命失败 —— 优雅退出。

    error_dir = Path(args.output_dir) / "error_cases"
    error_dir.mkdir(parents=True, exist_ok=True)

    # --- 解析 UD 树 ---
    trees = parse_trees_for_testset(
        testset_path=args.testset,
        config_path=args.config,
        progress_path=str(error_dir / "parse_test_clauses.progress"),
    )

    # --- 加载各实验的预测结果 ---
    pred_dir = Path(args.predictions_dir)
    experiments = {
        "baseline": load_predictions(str(pred_dir / "baseline.jsonl")),
        "ours_dep": load_predictions(str(pred_dir / "ours_dep.jsonl")),
        "ours_reflexion": load_predictions(str(pred_dir / "ours_reflexion.jsonl")),
    }

    all_results: Dict[str, Dict[str, Any]] = {}

    for exp_name, preds in progress_bar(
        experiments.items(), desc="Error analysis", unit="experiment",
    ):
        if not preds:
            logger.warning("No predictions for %s -- skipping error analysis", exp_name)
            all_results[exp_name] = {
                "experiment": exp_name,
                "total_samples": 0,
                "error_count": 0,
                "error_rate": 0.0,
                "primary_distribution": {},
                "secondary_distribution": {},
            }
            continue

        result = analyze_experiment_errors(
            experiment_name=exp_name,
            predictions=preds,
            gold=gold_triplets,
            trees=trees,
            output_dir=str(error_dir),
            constraints_path=args.constraints,
        )
        all_results[exp_name] = result

    # --- 打印跨实验对比表 ---
    print_error_comparison_table(all_results)

    # --- 保存汇总摘要 ---
    aggregate_summary = {
        "experiments": all_results,
        "total_errors_by_experiment": {
            k: v.get("error_count", 0) for k, v in all_results.items()
        },
    }
    summary_path = error_dir / "error_summary.json"
    write_json(str(summary_path), aggregate_summary)
    logger.info("Aggregate error summary saved to: %s", summary_path)

    print("\nError analysis files saved to: %s" % error_dir)
    for exp_name in ["baseline", "ours_dep", "ours_reflexion"]:
        exp_dir = error_dir / exp_name
        if exp_dir.exists():
            jsonl_files = sorted(exp_dir.glob("*.jsonl"))
            if jsonl_files:
                print(f"  {exp_name}/:")
                for f in jsonl_files:
                    print(f"    {f.name}")


if __name__ == "__main__":
    main()
