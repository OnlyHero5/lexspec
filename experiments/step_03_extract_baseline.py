#!/usr/bin/env python3
"""
LexSpec 步骤 03: 基线 —— 纯大语言模型抽取（无约束）
=====================================================

对测试集中每条合同条款，使用实验模型（Qwen3.5 9B）进行零样本法律三元组
抽取，不施加任何语言学约束或修正。此实验建立性能下界，作为后续
Dep 约束和 Reflexion 修正实验的对比基准。

流水线（逐条款）:
  1. 加载条款文本
  2. 调用 LegalTripletExtractor.extract() 进行纯大语言模型抽取
  3. 记录抽取结果、耗时和成功/失败状态

输出:
  ``outputs/predictions/baseline.jsonl`` —— 每条条款一个记录。

用法::

    python experiments/step_03_extract_baseline.py \\
        --config configs/model.yaml \\
        --testset data/processed/lexspec_100.jsonl \\
        --output-dir outputs/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.extraction.extractor import LegalTripletExtractor
from src.extraction.schema import (
    LegalTriplet, Subject, Action, Condition, LegalRole, ConditionType,
)
from src.utils.config import 加载模型配置, 构建实验客户端
from src.utils.logging import setup_logging, get_logger
from src.utils.io import read_jsonl, write_jsonl

logger = get_logger(__name__)


def 运行基线抽取(
    test_clauses: List[Dict],
    config_path: str = "configs/model.yaml",
    prompts_path: str = "configs/prompts.yaml",
) -> List[Dict]:
    """对测试集中全部条款运行基线大语言模型抽取。

    成功标准: 抽取出的三元组同时包含非空的 subject.text 和 action.predicate。

    参数:
        test_clauses: 测试集条款字典列表（每项含 clause_id 和 text）。
        config_path: 模型配置文件路径。
        prompts_path: 提示词配置文件路径。

    返回:
        结果字典列表，每条款一项，包含 clause_id / text / triplet /
        extraction_time_s / success。
    """
    config = 加载模型配置(config_path)
    client = 构建实验客户端(config)
    extractor = LegalTripletExtractor(client=client, prompts_path=prompts_path)

    results: List[Dict] = []
    logger.info("开始基线抽取，共 %d 条条款...", len(test_clauses))

    for clause_record in tqdm(test_clauses, desc="基线抽取", unit="clause"):
        clause_id = clause_record.get("clause_id", "?")
        clause_text = clause_record.get("text", "")

        if not clause_text:
            logger.warning("条款 %s 文本为空 —— 跳过", clause_id)
            results.append({
                "clause_id": clause_id, "text": "",
                "triplet": LegalTriplet(
                    subject=Subject(text="", role=LegalRole.OTHER),
                    action=Action(predicate="", object=""),
                    condition=Condition(),
                ).model_dump(mode="json"),
                "extraction_time_s": 0.0, "success": False,
            })
            continue

        t_start = time.perf_counter()
        triplet: LegalTriplet = extractor.extract(clause_text)
        t_elapsed = time.perf_counter() - t_start

        success = bool(
            triplet.subject.text.strip()
            and triplet.action.predicate.strip()
        )

        results.append({
            "clause_id": clause_id, "text": clause_text,
            "triplet": triplet.model_dump(mode="json"),
            "extraction_time_s": round(t_elapsed, 4), "success": success,
        })

        if not success:
            logger.debug("%s: 抽取返回空或不完整的三元组", clause_id)

    logger.info(
        "基线抽取完成: %d/%d 成功",
        sum(1 for r in results if r["success"]), len(results),
    )
    return results


def main() -> None:
    """主入口: 运行基线大语言模型抽取。"""
    arg_parser = argparse.ArgumentParser(
        description="LexSpec 步骤 03: 基线大语言模型抽取（无约束）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arg_parser.add_argument("--config", default="configs/model.yaml", help="模型配置文件路径")
    arg_parser.add_argument("--testset", default="data/processed/lexspec_100.jsonl", help="测试集 JSONL 路径")
    arg_parser.add_argument("--output-dir", default="outputs", help="输出目录")
    args = arg_parser.parse_args()

    import logging as _logging
    setup_logging(log_dir=str(Path(args.output_dir) / "logs"), level=_logging.INFO)

    testset_path = Path(args.testset)
    if not testset_path.exists():
        logger.error("测试集未找到 '%s'。请先运行 step_01_build_corpus.py。", testset_path)
        sys.exit(1)

    test_clauses = read_jsonl(str(testset_path))
    logger.info("已加载 %d 条测试条款: %s", len(test_clauses), testset_path)

    results = 运行基线抽取(test_clauses=test_clauses, config_path=args.config)

    output_dir = Path(args.output_dir) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "baseline.jsonl"
    write_jsonl(str(output_path), results)
    logger.info("已保存 %d 条基线预测至: %s", len(results), output_path)

    total = len(results)
    successful = sum(1 for r in results if r["success"])
    success_rate = successful / total * 100 if total > 0 else 0
    times = [r["extraction_time_s"] for r in results if r["success"]]
    avg_time = sum(times) / len(times) if times else 0.0

    print("\n" + "=" * 60)
    print("基线（纯大语言模型）抽取完成")
    print("=" * 60)
    print(f"  总条款数:               {total:>5d}")
    print(f"  成功抽取:               {successful:>5d}  ({success_rate:5.1f}%)")
    print(f"  失败抽取:               {total - successful:>5d}  ({100 - success_rate:5.1f}%)")
    print(f"  平均抽取耗时:           {avg_time:>8.3f}s")
    print(f"  输出:                   {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
