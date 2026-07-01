"""
字段级错误识别的检测辅助函数。

逐字段比较预测与金标，并确定次级（字段级）错误类别。
"""

from __future__ import annotations

from typing import List

from src.extraction.schema import (
    LegalTriplet, FieldErrorType,
)
from src.evaluation.text_normalizer import normalize
from src.evaluation.field_f1 import token_f1


def detect_field_errors(
    prediction: LegalTriplet,
    gold: LegalTriplet,
) -> List[str]:
    """逐字段比较预测与金标，返回错误标签列表。

    错误标签为不匹配字段的路径：
      - "subject.text": 主语文本词元 F1 < 1.0
      - "subject.role": 角色不同
      - "action.predicate": 谓词词元 F1 < 1.0
      - "action.object": 宾语词元 F1 < 1.0
      - "condition.text": 条件词元 F1 < 1.0
      - "condition.omission": 预测无条件而金标有条件
      - "condition.overextension": 预测有条件而金标无，或
        预测片段明显更大

    参数:
        prediction: 系统预测。
        gold: 金标准。

    返回:
        表示错误的字段路径字符串列表。完全匹配时为空列表。
    """
    errors: List[str] = []

    # 主语文本：词元级 F1 比较。
    pred_st = normalize(prediction.subject.text)
    gold_st = normalize(gold.subject.text)
    _, _, st_f1 = token_f1(pred_st, gold_st)
    if st_f1 < 1.0:
        errors.append("subject.text")

    # 主语角色：枚举精确匹配。
    if prediction.subject.role != gold.subject.role:
        errors.append("subject.role")

    # 谓词：词元级 F1。
    pred_pr = normalize(prediction.action.predicate)
    gold_pr = normalize(gold.action.predicate)
    _, _, pr_f1 = token_f1(pred_pr, gold_pr)
    if pr_f1 < 1.0:
        errors.append("action.predicate")

    # 宾语：词元级 F1。
    pred_ob = normalize(prediction.action.object)
    gold_ob = normalize(gold.action.object)
    _, _, ob_f1 = token_f1(pred_ob, gold_ob)
    if ob_f1 < 1.0:
        errors.append("action.object")

    # 条件：词元级 F1 + 遗漏/过度扩展检测。
    pred_co = normalize(prediction.condition.text)
    gold_co = normalize(gold.condition.text)

    has_pred_cond = bool(pred_co.strip())
    has_gold_cond = bool(gold_co.strip())

    if not has_pred_cond and has_gold_cond:
        # 遗漏：预测漏掉金标中存在的条件。
        errors.append("condition.omission")
    elif has_pred_cond and not has_gold_cond:
        # 过度扩展：预测幻觉出条件。
        errors.append("condition.overextension")
    elif has_pred_cond and has_gold_cond:
        _, _, co_f1 = token_f1(pred_co, gold_co)
        if co_f1 < 1.0:
            # 两侧均有条件但内容不同。
            # 判断是否主要为边界问题。
            pred_tokens = set(pred_co.split())
            gold_tokens = set(gold_co.split())
            overlap = pred_tokens & gold_tokens
            # 若预测明显更大（2 倍以上）且重叠较高，
            # 归类为过度扩展；否则为一般条件错误。
            if len(pred_tokens) > 2 * len(gold_tokens) and len(overlap) >= len(gold_tokens) * 0.5:
                errors.append("condition.overextension")
            elif co_f1 < 0.5:
                errors.append("condition.text")

    return errors


def determine_secondary_category(field_errors: List[str]) -> FieldErrorType:
    """将检测到的字段错误映射为次级（字段级）错误类型。

    多字段同时出错时的分类优先级：
      1. 主语错误（对法律分析最关键）
      2. 条件错误（边界问题）
      3. 谓词/宾语错误

    参数:
        field_errors: detect_field_errors() 返回的错误字段路径列表。

    返回:
        表示主要受影响字段的单个 FieldErrorType 枚举值。
    """
    # 优先检查主语相关错误（最高优先级）。
    if any(e.startswith("subject") for e in field_errors):
        if "subject.role" in field_errors and "subject.text" not in field_errors:
            return FieldErrorType.ROLE
        return FieldErrorType.SUBJECT

    # 检查条件相关错误。
    if "condition.omission" in field_errors:
        return FieldErrorType.CONDITION_OMISSION
    if "condition.overextension" in field_errors:
        return FieldErrorType.CONDITION_OVEREXTENSION
    if any(e.startswith("condition") for e in field_errors):
        return FieldErrorType.CONDITION_OMISSION  # 默认条件错误类型。

    # 检查谓词/宾语错误。
    if "action.predicate" in field_errors:
        return FieldErrorType.PREDICATE
    if "action.object" in field_errors:
        return FieldErrorType.OBJECT

    # 回退：有错误但未匹配（不应发生），
    # 默认 SUBJECT 作为最关键字段。
    return FieldErrorType.SUBJECT
