"""
标注共识用的字段级辅助函数
==========================
在标注共识过程中，从 LegalTriplet 的六个细粒度字段
提取、解析与比较所用的工具。

导出：
  - FIELD_SPEC:                       6 个比较字段的定义
  - _extract_field_values:            从 LegalTriplet 提取字段值
  - _parse_role:                      将角色值解析为 LegalRole 枚举
  - _parse_condition_type:            将条件类型解析为 ConditionType 枚举
  - _classify_disagreement_phenomenon: 将分歧映射到语言现象类别
  - _triplets_equal:                  判断两三元组在字段层面是否相同
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple

from src.extraction.schema import (
    LegalTriplet,
    LegalRole,
    ConditionType,
)
from src.annotation.normalization import normalize_text
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 最细粒度比较的字段（6 个）。
# 每项为 (field_name, parent_attribute, child_attribute) 元组。
# 驱动逐字段比较循环。
# ---------------------------------------------------------------------------

FIELD_SPEC: List[Tuple[str, str, str]] = [
    ("subject.text",       "subject",     "text"),
    ("subject.role",       "subject",     "role"),
    ("action.predicate",   "action",      "predicate"),
    ("action.object",      "action",      "object"),
    ("condition.text",     "condition",   "text"),
    ("condition.type",     "condition",   "type"),
]


def _extract_field_values(triplet: LegalTriplet) -> Dict[str, Any]:
    """从 LegalTriplet 提取 6 个独立字段值。

    参数：
        triplet: LegalTriplet 实例。

    返回：
        字段名（如 "subject.text"）到取值的 dict。
    """
    return {
        "subject.text":     triplet.subject.text,
        "subject.role":     triplet.subject.role.value
                            if isinstance(triplet.subject.role, LegalRole)
                            else str(triplet.subject.role),
        "action.predicate": triplet.action.predicate,
        "action.object":    triplet.action.object,
        "condition.text":   triplet.condition.text,
        "condition.type":   triplet.condition.type.value
                            if isinstance(triplet.condition.type, ConditionType)
                            else str(triplet.condition.type),
    }


def _parse_role(value) -> LegalRole:
    """将角色值（字符串或 LegalRole 枚举）解析为 LegalRole。

    参数：
        value: 如 "obligor" 的字符串或 LegalRole 枚举实例。

    返回：
        LegalRole 枚举值。无法识别时默认为 LegalRole.OTHER。
    """
    if isinstance(value, LegalRole):
        return value
    try:
        return LegalRole(str(value))
    except (ValueError, TypeError):
        logger.debug("Unrecognized role value '%s' -- defaulting to OTHER", value)
        return LegalRole.OTHER


def _parse_condition_type(value) -> ConditionType:
    """将条件类型值解析为 ConditionType 枚举。

    参数：
        value: 如 "trigger" 的字符串或 ConditionType 枚举实例。

    返回：
        ConditionType 枚举值。无法识别时默认为 ConditionType.NONE。
    """
    if isinstance(value, ConditionType):
        return value
    try:
        return ConditionType(str(value))
    except (ValueError, TypeError):
        logger.debug(
            "Unrecognized condition type '%s' -- defaulting to NONE", value
        )
        return ConditionType.NONE


def _classify_disagreement_phenomenon(
    field: str,
    qwen_anno: LegalTriplet,
    gemma_anno: LegalTriplet,
) -> str:
    """将字段级分歧归类为语言现象。

    根据分歧字段与上下文映射为有意义的
    现象标签，用于报告与诊断。

    参数：
        field: 不一致的字段名（如 "subject.role"）。
        qwen_anno: Qwen 标注（供上下文）。
        gemma_anno: Gemma 标注（供上下文）。

    返回：
        现象标签字符串（如 "role_mismatch"、"passive_voice"、
        "condition_boundary"、"object_identification"）。
    """
    if field == "subject.role":
        # 角色分歧常源于情态解读
        # 或被动/主动语态混淆。
        return "role_assignment"

    elif field == "subject.text":
        # 主语文本分歧可能表示对行为者
        # 的不同解读（如被动语态施事混淆）。
        return "subject_identification"

    elif field == "action.predicate":
        # 谓词分歧表示对哪一动词为
        # 主要法律谓词的不同解读。
        return "predicate_selection"

    elif field == "action.object":
        # 宾语分歧常源于长距离依存
        # 或辖域歧义。
        return "object_identification"

    elif field in ("condition.text", "condition.type"):
        # 条件分歧表示边界检测问题
        # 或条件类型分类差异。
        return "condition_detection"

    else:
        return "other"


def _triplets_equal(a: LegalTriplet, b: LegalTriplet) -> bool:
    """判断两个 LegalTriplet 在字段层面是否相同。

    使用与 field_level_consensus 相同的规范化判断相等，
    因此忽略表面形式差异（大小写、冠词）。

    参数：
        a: 第一个三元组。
        b: 第二个三元组。

    返回：
        规范化后 6 个字段均一致为 True，否则 False。
    """
    a_vals = _extract_field_values(a)
    b_vals = _extract_field_values(b)

    for field_name, _, _ in FIELD_SPEC:
        a_val = str(a_vals[field_name])
        b_val = str(b_vals[field_name])

        is_text_field = field_name in (
            "subject.text", "action.predicate", "action.object", "condition.text"
        )

        if is_text_field:
            if normalize_text(a_val) != normalize_text(b_val):
                return False
        else:
            if a_val != b_val:
                return False

    return True
