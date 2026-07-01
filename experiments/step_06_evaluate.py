#!/usr/bin/env python3
"""
LexSpec 实验 06: 双轨评估与显著性检验
=====================================

实验流水线第 6 步（依据规范第 8.2 节）。

本脚本对三种实验变体（Baseline、Ours-Dep、Ours-Reflexion）进行全面评估，计算：

  **轨道 1 —— 任务指标（加权三元组 F1）**
    - 总体加权 F1（5 个分量权重：subject_text 0.35、
      subject_role 0.10、predicate 0.20、object 0.20、condition 0.15）
    - 各字段精确率、召回率、F1
    - 逐样本 F1 分数（用于显著性检验）

  **轨道 2 —— 语言学指标**
    - 依存路径合法率
    - 被动语态恢复准确率
    - 条件边界 IoU
    - 语言学修正率（Ours-Dep 与 Ours-Reflexion）

  **轨道 3 —— 显著性检验**
    - 配对 bootstrap（10,000 次重采样）进行两两比较
    - Wilcoxon 符号秩检验（补充）
    - 按语言学现象分层的显著性分析

用法
----
    python experiments/step_06_evaluate.py \\
        --config configs/model.yaml \\
        --output-dir outputs/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# --- 将项目根目录加入 sys.path ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.eval_orchestrator import run_evaluation
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    """主入口：运行双轨评估。"""
    arg_parser = argparse.ArgumentParser(
        description="LexSpec 实验 06: 双轨评估与显著性检验",
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
        help="测试集 JSONL 文件路径（用于 UD 树解析）",
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

    logger.info("Starting dual-track evaluation...")
    t_start = time.perf_counter()

    run_evaluation(
        predictions_dir=args.predictions_dir,
        gold_path=args.gold,
        testset_path=args.testset,
        output_dir=args.output_dir,
        config_path=args.config,
        constraints_path=args.constraints,
    )

    t_elapsed = time.perf_counter() - t_start
    logger.info("Evaluation complete in %.2f seconds", t_elapsed)


if __name__ == "__main__":
    main()
