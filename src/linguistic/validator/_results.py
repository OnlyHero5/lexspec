"""
结果构建工具：LinguisticEvidence、反馈与修正应用。

这些函数从累积的校验状态构建 ValidationResult 子对象。
"""

from __future__ import annotations

from typing import Optional, List

from src.extraction.schema import (
    DependencyTree,
    Token,
    LegalTriplet,
    Subject,
    Action,
    Condition,
    LegalRole,
    ConditionType,
    LinguisticEvidence,
    FieldCorrection,
    ConditionSpan,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_linguistic_evidence(
    tree: DependencyTree,
    predicate_idx: int,
    ud_subject: Optional[Token],
    ud_object: Optional[Token],
    condition_span: Optional[ConditionSpan],
    is_passive_detected: bool,
    modality_aux: str,
    polarity: str,
    legal_role: LegalRole,
) -> LinguisticEvidence:
    """从校验状态构建 LinguisticEvidence 对象。"""
    predicate_token = tree.get_token(predicate_idx)
    max_argument_distance = _max_argument_distance(tree, predicate_idx)

    evidence = LinguisticEvidence(
        predicate=predicate_token.lemma if predicate_token else "",
        predicate_index=predicate_idx,
        ud_subject=ud_subject.text if ud_subject else "",
        ud_object=ud_object.text if ud_object else "",
        condition_span=condition_span.text if condition_span else "",
        condition_type=(
            condition_span.condition_type if condition_span
            else ConditionType.NONE
        ),
        passive_detected=is_passive_detected,
        modality_aux=modality_aux,
        polarity=polarity,
        legal_role=legal_role,
        max_argument_distance=max_argument_distance,
    )

    return evidence


def _max_argument_distance(tree: DependencyTree, predicate_idx: int) -> int:
    """计算谓词到主语/宾语论元的最大依存距离。"""
    if predicate_idx <= 0:
        return 0
    max_dist = 0
    for deprel in ("nsubj", "nsubj:pass", "obj"):
        for child in tree.get_children(predicate_idx, deprel=deprel):
            dist = tree.get_dependency_distance(child.index, predicate_idx)
            max_dist = max(max_dist, dist)
    return max_dist


def build_feedback(feedback_parts: List[str]) -> str:
    """组装人类可读的反馈字符串。

    用于两种场景：
      1. Reflexion：反馈包含在发回大语言模型的 Reflexion 提示中，
         说明错误及如何重新分析从句。
      2. 错误分析：反馈记录在错误案例记录中，
         供下游诊断报告使用。

    反馈使用自然语言并引用具体 UD 关系，
    帮助大语言模型理解应关注的句法模式。

    参数：
        feedback_parts: 各校验步骤产生的单条反馈字符串列表。

    返回：
        拼接后的反馈字符串，无问题时为空字符串。
    """
    if not feedback_parts:
        return ""

    # 为每条反馈编号以便阅读。
    numbered = [
        f"{i}. {part}" for i, part in enumerate(feedback_parts, 1)
    ]

    preamble = (
        "The UD syntactic analysis identified the following issues with "
        "the extracted triplet:"
    )

    return preamble + "\n" + "\n".join(numbered)


def apply_corrections(
    triplet: LegalTriplet,
    corrections: List[FieldCorrection],
) -> LegalTriplet:
    """应用字段修正列表以产出修正后的三元组。

    遍历全部字段修正并更新三元组对应字段。
    修正后的三元组为新的 LegalTriplet 对象 —— 原始对象从不修改。

    处理的字段路径：
      - "subject.text" -> triplet.subject.text
      - "subject.role" -> triplet.subject.role（由字符串转换）
      - "action.object" -> triplet.action.object
      - "action.predicate" -> triplet.action.predicate
      - "condition.text" -> triplet.condition.text
      - "condition.type" -> triplet.condition.type（由字符串转换）

    参数：
        triplet: 原始大语言模型三元组（不修改）。
        corrections: FieldCorrection 对象列表。

    返回：
        应用修正后的新 LegalTriplet。
    """
    # 从原始三元组副本开始。
    # 按需构建新的 Subject、Action、Condition 对象。
    new_subject_text = triplet.subject.text
    new_subject_role = triplet.subject.role
    new_predicate = triplet.action.predicate
    new_object = triplet.action.object
    new_condition_text = triplet.condition.text
    new_condition_type = triplet.condition.type

    for correction in corrections:
        field = correction.field
        corrected_value = correction.corrected

        if field == "subject.text":
            new_subject_text = corrected_value
            logger.debug(
                "Applied correction: subject.text '%s' -> '%s'",
                correction.original, corrected_value,
            )
        elif field == "subject.role":
            # 将字符串角色转回 LegalRole 枚举。
            try:
                new_subject_role = LegalRole(corrected_value)
                logger.debug(
                    "Applied correction: subject.role '%s' -> '%s'",
                    correction.original, corrected_value,
                )
            except ValueError:
                logger.warning(
                    "Invalid role correction value '%s' — keeping original",
                    corrected_value,
                )
        elif field == "action.object":
            new_object = corrected_value
            logger.debug(
                "Applied correction: action.object '%s' -> '%s'",
                correction.original, corrected_value,
            )
        elif field == "action.predicate":
            new_predicate = corrected_value
            logger.debug(
                "Applied correction: action.predicate '%s' -> '%s'",
                correction.original, corrected_value,
            )
        elif field == "condition.text":
            new_condition_text = corrected_value
            logger.debug(
                "Applied correction: condition.text '%s' -> '%s'",
                correction.original[:40], corrected_value[:40],
            )
        elif field == "condition.type":
            try:
                new_condition_type = ConditionType(corrected_value)
                logger.debug(
                    "Applied correction: condition.type '%s' -> '%s'",
                    correction.original, corrected_value,
                )
            except ValueError:
                logger.warning(
                    "Invalid condition type correction '%s' — keeping original",
                    corrected_value,
                )
        else:
            logger.warning("Unknown correction field: '%s'", field)

    return LegalTriplet(
        subject=Subject(text=new_subject_text, role=new_subject_role),
        action=Action(predicate=new_predicate, object=new_object),
        condition=Condition(text=new_condition_text, type=new_condition_type),
    )
