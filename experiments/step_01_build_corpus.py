#!/usr/bin/env python3
"""
LexSpec 实验 01: 从 CUAD 数据构建评估语料库
============================================

从 CUAD v1 合同数据构建 LexSpec 评估语料库。支持两种模式：

  - 分层抽样模式（默认 100 条条款）：按语言学现象配额均衡抽样，确保被动语态、
    条件从句、关系从句、长距离依存和否定各拥有足够样本以供统计比较。
  - 全量模式（--all）：使用全部 510 份合同中的全部有效条款。

用法::

    python experiments/step_01_build_corpus.py
    python experiments/step_01_build_corpus.py --all
    python experiments/step_01_build_corpus.py --source spans --all \\
        --output data/processed/lexspec_spans.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.corpus.phenomena_detector import is_boilerplate_clause
from src.corpus.cuad_loader import load_cuad_data, load_cuad_spans, load_cuad_qa_spans
from src.corpus.clause_processor import split_into_clauses, build_clause_records
from src.corpus.selection import select_all_clauses, select_balanced_testset
from src.utils.logging import setup_logging, get_logger
from src.utils.io import write_jsonl
from src.utils.config import 加载模型配置, 构建Stanza解析器
from src.utils.constraints import (
    get_corpus_sampling_config,
    get_phenomenon_quotas,
    get_validation_thresholds,
    load_constraints_config,
)

logger = get_logger(__name__)


# ======================================================================
# 主入口
# ======================================================================


def main() -> None:
    """主入口：从 CUAD 数据构建 LexSpec 评估语料库。"""
    parser = argparse.ArgumentParser(
        description="LexSpec 实验 01: 从 CUAD 数据构建评估语料库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "全量模式：使用全部 510 份合同中的全部有效条款，不进行分层抽样。"
            "自动设置 --selection-mode all --max-contexts 0 --target-count 0，"
            "默认输出至 data/processed/lexspec_corpus.jsonl"
        ),
    )
    parser.add_argument(
        "--source",
        choices=["sentences", "spans", "qa_spans"],
        default="sentences",
        help=(
            "条款来源：sentences=Stanza 对完整合同进行句子切分（数量最多）；"
            "spans=master_clauses.csv 专家标注片段（约 5k）；"
            "qa_spans=CUAD JSON 答案片段（约 13k）"
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=["stratified", "all"],
        default="stratified",
        help="stratified=均衡分层抽样；all=全部有效且已解析的条款",
    )
    parser.add_argument(
        "--constraints",
        type=str,
        default="configs/constraints.yaml",
        help="约束配置文件路径（抽样配额、阈值）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model.yaml",
        help="模型配置文件路径（Stanza 设置）",
    )
    parser.add_argument(
        "--cuad-path",
        type=str,
        default="data/raw/CUAD_v1/CUAD_v1.json",
        help="CUAD v1 JSON 文件路径",
    )
    parser.add_argument(
        "--master-clauses-path",
        type=str,
        default="data/raw/CUAD_v1/master_clauses.csv",
        help="CUAD master_clauses.csv 路径（与 --source spans 配合使用）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/lexspec_100.jsonl",
        help="所选测试集的输出路径",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="覆盖 constraints YAML 中的 corpus_sampling.target_count_default",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=0,
        help="最多处理的合同段落数（0 = 全部 510 份；默认全部）",
    )
    parser.add_argument(
        "--max-clauses",
        type=int,
        default=0,
        help="最多处理的候选条款数（0 = 全部；用于 smoke test，在解析前截断）",
    )
    args = parser.parse_args()

    # --all 快捷模式。
    if args.all:
        args.selection_mode = "all"
        args.max_contexts = 0
        args.target_count = 0
        if args.output == "data/processed/lexspec_100.jsonl":
            args.output = "data/processed/lexspec_corpus.jsonl"

    # ---- 初始化 ----
    import logging as _logging
    setup_logging(log_dir="outputs", level=_logging.INFO)

    constraints = load_constraints_config(args.constraints)
    sampling_cfg = get_corpus_sampling_config(constraints, args.constraints)
    validation_cfg = get_validation_thresholds(constraints, args.constraints)
    long_distance_mdd = validation_cfg["long_distance_mdd"]

    default_target = sampling_cfg["target_count_default"]
    if args.target_count is None:
        args.target_count = default_target

    logger.info("Loading model config: %s", args.config)
    model_config = 加载模型配置(args.config)
    nlp_parser = 构建Stanza解析器(model_config, args.config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = str(output_path.with_suffix(".progress"))

    # ---- 加载 CUAD 条款 ----
    if args.source == "spans":
        clauses = load_cuad_spans(args.master_clauses_path)
        source_label = "cuad_v1_spans"
    elif args.source == "qa_spans":
        clauses = load_cuad_qa_spans(args.cuad_path)
        source_label = "cuad_v1_qa"
    else:
        contexts = load_cuad_data(args.cuad_path)
        max_ctx = args.max_contexts if args.max_contexts > 0 else None
        clauses = split_into_clauses(
            nlp_parser, contexts, max_contexts=max_ctx, progress_path=progress_path,
        )
        source_label = "cuad_v1_sentences"

    if args.max_clauses and args.max_clauses > 0:
        clauses = clauses[: args.max_clauses]
        logger.info("Smoke test: truncated to first %d candidate clauses", len(clauses))

    # ---- 条款选择 ----
    if args.selection_mode == "all":
        records, _ = build_clause_records(
            nlp_parser, clauses, source_label,
            long_distance_mdd=long_distance_mdd,
            progress_path=progress_path,
        )
        selected = select_all_clauses(records)
    else:
        target = args.target_count if args.target_count > 0 else default_target
        phenomenon_quotas = get_phenomenon_quotas(constraints, target, args.constraints)
        selected = select_balanced_testset(
            parser=nlp_parser,
            clauses=clauses,
            target_count=target,
            phenomenon_quotas=phenomenon_quotas,
            random_seed=sampling_cfg["random_seed"],
            long_distance_mdd=long_distance_mdd,
            source_label=source_label,
            progress_path=progress_path,
        )

    # ---- 保存输出 ----
    write_jsonl(str(output_path), selected)
    logger.info("Saved %d clauses to: %s", len(selected), output_path)

    # ---- 打印选择统计 ----
    print("\n" + "=" * 60)
    print("LexSpec Corpus Construction Complete")
    print("=" * 60)
    print(f"Source:          {args.source} ({source_label})")
    print(f"Selection mode:  {args.selection_mode}")
    print(f"Total clauses:   {len(selected)}")
    print()

    phen_names = [
        "passive", "conditional", "relative_clause",
        "long_distance", "negation",
    ]
    print(f"{'Phenomenon':<25s} {'Count':>6s}  {'Pct':>10s}")
    print("-" * 43)
    for phen in phen_names:
        count = sum(1 for r in selected if r["phenomena"][phen])
        pct = count / len(selected) * 100 if selected else 0
        print(f"  {phen:<23s} {count:>6d}  {pct:>9.1f}%")
    print()

    # 全量模式下显示已过滤的定义条款数量。
    if args.selection_mode == "all":
        defs_filtered = sum(
            1 for r in records if r["phenomena"].get("is_definition", False)
        )
        if defs_filtered:
            print(f"  (Filtered {defs_filtered} definition clauses)")
    print()

    # 多现象重叠统计（排除 is_definition）。
    print("Multi-phenomenon clause distribution:")
    multi = Counter(
        sum(1 for k, v in r["phenomena"].items() if k != "is_definition" and v)
        for r in selected
    )
    for k in sorted(multi):
        print(f"  {k} phenomena: {multi[k]} clauses")
    print("=" * 60)


if __name__ == "__main__":
    main()
