#!/usr/bin/env python3
"""
LexSpec 步骤 04: Ours-Dep —— 大语言模型 + UD 约束校验
=======================================================

在基线大语言模型抽取之后，使用 UD 依存句法约束校验器对抽取结果进行
验证和自动修正。当校验器判定状态为:

  - VALID:              直接使用原始大语言模型三元组
  - CORRECTED:          使用校验器自动修正后的三元组
  - REFLEXION_REQUIRED: 使用原始大语言模型三元组（Reflexion 修正延后到步骤 05）

本实验测试 UD 约束**单独**（不经过迭代大语言模型重生成）能否提升抽取质量。

流水线（逐条款）:
  1. 复用 Baseline 初抽三元组（baseline.jsonl，保证消融单变量）
  2. Stanza UD 依存解析 (StanzaParser.parse)
  3. 约束校验 (ConstraintValidator.validate)
  4. 根据校验状态选择最终三元组
  5. 记录校验结果供后续错误分析

输出:
  ``outputs/predictions/ours_dep.jsonl``           —— 最终三元组
  ``outputs/predictions/ours_dep_validations.jsonl`` —— 校验详情

用法::

    python experiments/step_04_extract_dep.py \\
        --config configs/model.yaml \\
        --testset data/processed/lexspec_100.jsonl \\
        --output-dir outputs/
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.extraction.schema import (
    LegalTriplet, Subject, Action, Condition, LegalRole,
    ValidationStatus,
)
from src.linguistic.condition_extractor import ConditionExtractor
from src.linguistic.polarity_detector import PolarityDetector
from src.linguistic.validator import ConstraintValidator
from src.utils.config import 加载模型配置, 构建Stanza解析器
from src.evaluation.data_loading import (
    require_baseline_triplet_map,
    skipped_validation_record,
    validation_result_record,
    save_validation_records,
)
from src.utils.logging import setup_logging, get_logger
from src.utils.io import read_jsonl, write_jsonl
from src.utils.progress import progress_bar

logger = get_logger(__name__)


def _空结果(clause_id: str, clause_text: str) -> Dict:
    """为无法处理的条款构建空结果记录。"""
    return {
        "clause_id": clause_id, "text": clause_text,
        "triplet": LegalTriplet(
            subject=Subject(text="", role=LegalRole.OTHER),
            action=Action(predicate="", object=""),
            condition=Condition(),
        ).model_dump(mode="json"),
        "validation_status": "SKIPPED", "used_correction": False,
        "success": False,
        "stanza_parse_ok": False, "total_time_s": 0.0,
    }


def 运行Dep流水线(
    test_clauses: List[Dict],
    config_path: str = "configs/model.yaml",
    prompts_path: str = "configs/prompts.yaml",
    constraints_path: str = "configs/constraints.yaml",
    progress_path: str | None = None,
    baseline_predictions_path: str | None = None,
) -> tuple[List[Dict], List[Dict]]:
    """运行完整 Ours-Dep 流水线: Baseline 初抽 + UD 校验。

    若提供 ``baseline_predictions_path``，则复用 Baseline 初抽三元组，
    不再重复调用 LLM，以保证消融实验单变量控制。

    返回:
        (最终结果字典列表, 带 clause_id 的校验记录列表) 元组。
    """
    config = 加载模型配置(config_path)
    parser = 构建Stanza解析器(config)

    if not baseline_predictions_path:
        raise ValueError(
            "Ours-Dep 消融实验必须提供 baseline_predictions_path 以复用 step_03 初抽。"
        )
    baseline_map = require_baseline_triplet_map(
        baseline_predictions_path, test_clauses,
    )
    cond_extractor = ConditionExtractor(constraints_path)
    polarity_det = PolarityDetector(constraints_path)
    validator = ConstraintValidator(
        parser=parser, condition_extractor=cond_extractor,
        polarity_detector=polarity_det, constraints_path=constraints_path,
    )

    final_results: List[Dict] = []
    validation_records: List[Dict] = []

    logger.info("开始 Ours-Dep 流水线，共 %d 条条款...", len(test_clauses))

    for clause_record in progress_bar(
        test_clauses,
        desc="Ours-Dep 流水线",
        unit="clause",
        progress_path=progress_path,
    ):
        clause_id = clause_record.get("clause_id", "?")
        clause_text = clause_record.get("text", "")

        if not clause_text:
            logger.warning("条款 %s 文本为空 —— 跳过", clause_id)
            final_results.append(_空结果(clause_id, ""))
            validation_records.append(skipped_validation_record(clause_id))
            continue

        t_start = time.perf_counter()

        # ---- 步骤 1: 复用 baseline 初抽（消融单变量控制）----
        llm_triplet: LegalTriplet = baseline_map[clause_id]

        # ---- 步骤 2: Stanza UD 依存解析 ----
        dep_tree = parser.parse(clause_text)

        # ---- 步骤 3: UD 约束校验 ----
        val_result = validator.validate(
            triplet=llm_triplet, text=clause_text, tree=dep_tree,
        )
        validation_records.append(validation_result_record(clause_id, val_result))

        # ---- 步骤 4: 选择最终三元组 ----
        if (
            val_result.status == ValidationStatus.CORRECTED
            and val_result.corrected_prediction is not None
        ):
            final_triplet = val_result.corrected_prediction
            used_correction = True
        else:
            final_triplet = llm_triplet
            used_correction = False

        t_elapsed = time.perf_counter() - t_start
        final_results.append({
            "clause_id": clause_id, "text": clause_text,
            "triplet": final_triplet.model_dump(mode="json"),
            "validation_status": val_result.status.value,
            "used_correction": used_correction, "stanza_parse_ok": True,
            "total_time_s": round(t_elapsed, 4),
        })

    logger.info(
        "Ours-Dep 流水线完成: %d 条",
        len(final_results),
    )
    return final_results, validation_records


def main() -> None:
    """主入口: 运行 Ours-Dep 流水线。"""
    arg_parser = argparse.ArgumentParser(
        description="LexSpec 步骤 04: Ours-Dep —— 大语言模型 + UD 约束校验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arg_parser.add_argument("--config", default="configs/model.yaml", help="模型配置文件路径")
    arg_parser.add_argument("--testset", default="data/processed/lexspec_100.jsonl", help="测试集 JSONL 路径")
    arg_parser.add_argument("--output-dir", default="outputs", help="输出目录")
    arg_parser.add_argument(
        "--baseline-predictions",
        default=None,
        help="Baseline 预测 JSONL；默认使用 <output-dir>/predictions/baseline.jsonl",
    )
    args = arg_parser.parse_args()

    import logging as _logging
    setup_logging(log_dir=str(Path(args.output_dir) / "logs"), level=_logging.INFO)

    testset_path = Path(args.testset)
    if not testset_path.exists():
        logger.error("测试集未找到 '%s'。请先运行 step_01_build_corpus.py。", testset_path)
        sys.exit(1)

    test_clauses = read_jsonl(str(testset_path))
    logger.info("已加载 %d 条测试条款: %s", len(test_clauses), testset_path)

    output_dir = Path(args.output_dir) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = args.baseline_predictions or str(output_dir / "baseline.jsonl")

    final_results, validation_records = 运行Dep流水线(
        test_clauses=test_clauses,
        config_path=args.config,
        progress_path=str(output_dir / "ours_dep.progress"),
        baseline_predictions_path=baseline_path,
    )

    predictions_path = output_dir / "ours_dep.jsonl"
    write_jsonl(str(predictions_path), final_results)
    logger.info("已保存 %d 条 Ours-Dep 预测至: %s", len(final_results), predictions_path)

    validations_path = output_dir / "ours_dep_validations.jsonl"
    save_validation_records(str(validations_path), validation_records)
    logger.info("已保存 %d 条校验结果至: %s", len(validation_records), validations_path)

    status_counter = Counter(r["validation_status"] for r in final_results)
    correction_count = sum(1 for r in final_results if r["used_correction"])
    total = len(final_results)

    print("\n" + "=" * 60)
    print("Ours-Dep（大语言模型 + UD 校验）流水线完成")
    print("=" * 60)
    print(f"  总条款数:               {total:>5d}")
    for status in ["VALID", "CORRECTED", "REFLEXION_REQUIRED"]:
        cnt = status_counter.get(status, 0)
        print(f"  {status:<20s}       {cnt:>5d}  ({cnt / max(total, 1) * 100:5.1f}%)")
    print(f"  已应用修正:             {correction_count:>5d}")
    print(f"  预测输出:               {predictions_path}")
    print(f"  校验输出:               {validations_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
