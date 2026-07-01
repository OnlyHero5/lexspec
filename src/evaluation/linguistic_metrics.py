"""
语言学指标统一入口。

聚合四项语言学专用评估指标：
  1. 依存路径合法率（dep_path_metrics.py）
  2. 被动语态恢复准确率（passive_metrics.py）
  3. 条件边界 IoU（condition_metrics.py）
  4. 语言学修正率（condition_metrics.py）

这些指标作诊断工具：帮助识别导致抽取失败的语言学现象，指导针对性改进。
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

from src.extraction.schema import (
    LegalTriplet, DependencyTree, ValidationResult,
)
from src.evaluation.dep_path_metrics import compute_dependency_path_legality
from src.evaluation.passive_metrics import compute_passive_recovery_accuracy
from src.evaluation.condition_metrics import compute_condition_iou, compute_correction_rate
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_all_linguistic_metrics(
    predictions: List[LegalTriplet],
    gold: List[LegalTriplet],
    trees: List[DependencyTree],
    validation_results: Optional[List[ValidationResult]] = None,
) -> Dict[str, Any]:
    """一次调用计算全部四项语言学指标。

    语言学评估维度的主入口。
    返回可直接序列化为 JSON 写入评估报告的综合字典。

    参数:
        predictions: 预测 LegalTriplet 列表。
        gold: 金标准 LegalTriplet 列表。
        trees: UD 解析得到的 DependencyTree 列表。
        validation_results: 可选 ValidationResult 列表。
                            为 None 时省略修正率指标。

    返回:
        顶层键包括：
        - dependency_path_legality: float
        - passive_recovery: Dict[str, float]（被动语态指标）
        - condition_iou: float
        - correction_analysis: Dict[str, float] 或 None
        - summary: Dict，报告用聚合亮点

    异常:
        ValueError: 核心输入（predictions、gold、trees）长度不一致。
    """
    n = len(predictions)
    if n != len(gold) or n != len(trees):
        raise ValueError(
            f"Core inputs must have the same length. "
            f"Got predictions={len(predictions)}, gold={len(gold)}, trees={len(trees)}."
        )

    # 指标 1：依存路径合法性。
    legality = compute_dependency_path_legality(predictions, trees)

    # 指标 2：被动语态恢复准确率。
    passive_metrics = compute_passive_recovery_accuracy(predictions, trees, gold)

    # 指标 3：条件边界 IoU。
    condition_iou = compute_condition_iou(predictions, trees)

    # 指标 4：修正率（仅当提供验证结果时）。
    correction_metrics = None
    if validation_results is not None:
        correction_metrics = compute_correction_rate(validation_results)

    # 构建评估报告摘要。
    summary = {
        "linguistic_quality_indicators": {
            "dependency_legality": legality,
            "condition_boundary_iou": condition_iou,
        },
        "passive_voice_handling": {
            "passive_count": passive_metrics["passive_count"],
            "recovery_accuracy": passive_metrics["recovery_accuracy"],
            "false_agent_rate": passive_metrics["false_agent_rate"],
        },
    }
    if correction_metrics is not None:
        summary["validator_performance"] = {
            "valid_rate": correction_metrics["valid_rate"],
            "correction_success_rate": correction_metrics["correction_success_rate"],
            "reflexion_required_rate": correction_metrics["reflexion_rate"],
        }

    return {
        "dependency_path_legality": legality,
        "passive_recovery": passive_metrics,
        "condition_iou": condition_iou,
        "correction_analysis": correction_metrics,
        "summary": summary,
    }
