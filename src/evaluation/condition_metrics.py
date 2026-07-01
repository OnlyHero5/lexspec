"""
条件边界 IoU 与语言学修正率指标。

  - 条件边界 IoU：预测条件片段与 UD 推导条件片段间的词元级交并比。
  - 语言学修正率：UD 约束验证器成功修正 LLM 抽取错误的频次统计。
"""

from __future__ import annotations

from typing import List, Dict, Set

from src.extraction.schema import (
    LegalTriplet, DependencyTree, ValidationResult,
)
from src.evaluation.text_normalizer import normalize
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_condition_iou(
    predictions: List[LegalTriplet],
    trees: List[DependencyTree],
) -> float:
    """计算预测条件片段与 UD 条件片段间的平均词元级 IoU。

    对每个预测与 UD 树均含条件的子句（树中有 advcl 且
    prediction.condition.text 非空），计算：

        IoU = |prediction_tokens ∩ condition_subtree_tokens| /
              |prediction_tokens ∪ condition_subtree_tokens|

    衡量系统识别条件从句边界的准确度。IoU = 1.0 表示预测片段
    与 UD 条件片段完全一致。

    两侧均无条件的子句排除（显然一致，对 IoU 无信息量）。
    仅一侧有条件时该子句 IoU = 0.0。

    参数:
        predictions: 预测 LegalTriplet 列表。
        trees: DependencyTree 列表（等长）。

    返回:
        至少一侧有条件的全部子句对的平均 IoU。[0, 1] 浮点数。

    异常:
        ValueError: 输入列表长度不同。
    """
    if len(predictions) != len(trees):
        raise ValueError(
            f"predictions and trees must have the same length. "
            f"Got {len(predictions)} and {len(trees)}."
        )

    ious: List[float] = []
    skipped_no_condition = 0
    skipped_one_side = 0

    for pred, tree in zip(predictions, trees):
        if tree.token_count == 0:
            continue

        # 检查 UD 树是否含条件从句（advcl）。
        advcl_tokens = tree.find_tokens_by_deprel("advcl")
        has_tree_condition = len(advcl_tokens) > 0
        has_pred_condition = bool(pred.condition.text and pred.condition.text.strip())

        if not has_tree_condition and not has_pred_condition:
            # 两侧均无条件 — 跳过（对 IoU 无信息量）。
            skipped_no_condition += 1
            continue

        # 获取预测条件词元。
        pred_text = normalize(pred.condition.text, remove_articles=True)
        pred_token_set = set(pred_text.split()) if pred_text else set()

        # 获取树条件子树词元（全部 advcl 片段）。
        tree_condition_set: Set[str] = set()
        for advcl_token in advcl_tokens:
            # 取该 advcl 中心的完整子树。
            subtree_tokens = tree.get_subtree_tokens(advcl_token.index)
            for st in subtree_tokens:
                # 用词形基形式匹配（归一化后）。
                tree_condition_set.add(st.lemma.lower())

        if not pred_token_set or not tree_condition_set:
            # 一侧无词元 — IoU = 0。
            ious.append(0.0)
            skipped_one_side += 1
            continue

        # 计算 IoU。
        intersection = pred_token_set & tree_condition_set
        union = pred_token_set | tree_condition_set

        iou = len(intersection) / len(union) if union else 0.0
        ious.append(iou)

    mean_iou = sum(ious) / len(ious) if ious else 0.0

    logger.info(
        "Condition Boundary IoU: %.4f (n=%d, skipped_no_cond=%d, skipped_one_side=%d)",
        mean_iou, len(ious), skipped_no_condition, skipped_one_side,
    )
    return mean_iou


def compute_correction_rate(
    validation_results: List[ValidationResult],
) -> Dict[str, float]:
    """统计 UD 约束验证器成功修正 LLM 抽取错误的频次。

    验证器对每个预测产生三种状态之一：
      - VALID:              预测与 UD 证据一致；无需修正。
      - CORRECTED:          轻微字段错误已自动修复。
      - REFLEXION_REQUIRED: 结构性错误，需 LLM 重新抽取。

    报告这些状态的分布及修正成功率：验证器可自动修正（CORRECTED）
    与需 LLM 重抽（REFLEXION）的比例。

    参数:
        validation_results: UD 约束验证器产生的 ValidationResult 列表。

    返回:
        字典，键包括：
        - total_validated: int — 验证结果总数。
        - valid_count: int — VALID 状态数。
        - valid_rate: float — VALID / total。
        - corrected_count: int — CORRECTED 状态数。
        - corrected_rate: float — CORRECTED / total。
        - reflexion_count: int — REFLEXION_REQUIRED 状态数。
        - reflexion_rate: float — REFLEXION_REQUIRED / total。
        - correction_success_rate: float — CORRECTED / (CORRECTED + REFLEXION)。
          表示无需 LLM 重抽即可自动修正的错误比例。
    """
    total = len(validation_results)
    if total == 0:
        return {
            "total_validated": 0,
            "valid_count": 0,
            "valid_rate": 0.0,
            "corrected_count": 0,
            "corrected_rate": 0.0,
            "reflexion_count": 0,
            "reflexion_rate": 0.0,
            "correction_success_rate": 0.0,
        }

    from src.extraction.schema import ValidationStatus

    valid_count = sum(1 for r in validation_results if r.status == ValidationStatus.VALID)
    corrected_count = sum(1 for r in validation_results if r.status == ValidationStatus.CORRECTED)
    reflexion_count = sum(1 for r in validation_results if r.status == ValidationStatus.REFLEXION_REQUIRED)

    valid_rate = valid_count / total
    corrected_rate = corrected_count / total
    reflexion_rate = reflexion_count / total

    # 修正成功率：在错误（corrected + reflexion）中自动修正的比例。
    error_total = corrected_count + reflexion_count
    correction_success_rate = corrected_count / error_total if error_total > 0 else 0.0

    logger.info(
        "Correction rate: total=%d, VALID=%d(%.2f), CORRECTED=%d(%.2f), "
        "REFLEXION=%d(%.2f), success=%.4f",
        total, valid_count, valid_rate, corrected_count, corrected_rate,
        reflexion_count, reflexion_rate, correction_success_rate,
    )

    return {
        "total_validated": total,
        "valid_count": valid_count,
        "valid_rate": valid_rate,
        "corrected_count": corrected_count,
        "corrected_rate": corrected_rate,
        "reflexion_count": reflexion_count,
        "reflexion_rate": reflexion_rate,
        "correction_success_rate": correction_success_rate,
    }
