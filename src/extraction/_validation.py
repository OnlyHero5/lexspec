"""
三元组校验与规范化 —— Pydantic 模型校验 + 枚举值规范化
======================================================

将 LLM 返回的解析后字典转换为经过校验的 ``LegalTriplet`` 实例，
包含枚举值模糊匹配和字段回退逻辑。
"""

from __future__ import annotations

from typing import Dict, Any

from src.extraction.schema import (
    LegalTriplet,
    Subject,
    Action,
    Condition,
    LegalRole,
    ConditionType,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def validate_and_normalize_triplet(
    parsed: Dict[str, Any],
    clause: str,
) -> LegalTriplet:
    """对照 LegalTriplet 模式校验解析后的字典并规范化。

    在大语言模型返回值与严格的 ``LegalTriplet`` Pydantic 模型之间架起桥梁：

    1. 强制转换枚举字符串值（如 ``"OBLIGOR"`` → ``LegalRole.OBLIGOR``）
    2. 用合理默认值填充缺失的可选字段
    3. 运行 Pydantic 校验（``LegalTriplet.model_validate``）
    4. 规范化输出（修剪空白、清理条件字段）

    参数:
        parsed:  大语言模型响应的解析后 JSON 字典。
        clause:  原始条款文本（用于日志上下文）。

    返回:
        经过校验的 ``LegalTriplet`` 实例。

    异常:
        ValueError:  若强制转换后仍无法对照 ``LegalTriplet`` 模式校验。
    """
    # --- 步骤 1：提取并强制转换嵌套字典 ---
    # 大语言模型输出预期含顶层键 "subject"、"action"、"condition"，
    # 各含嵌套字典。若键完全缺失则提供空字典（Pydantic 模型将捕获缺失字段）。
    subject_raw = parsed.get("subject", {})
    action_raw = parsed.get("action", {})
    condition_raw = parsed.get("condition", {})

    # 处理 subject/action/condition 为字符串的情况（大语言模型返回纯字符串而非对象的边缘情况）。
    if isinstance(subject_raw, str):
        # 大语言模型可能仅返回当事方名称字符串。
        # 转换为最小字典以避免校验崩溃。
        logger.debug("subject was a string — converting to dict: %r", subject_raw)
        subject_raw = {"text": subject_raw, "role": "other"}
    if isinstance(action_raw, str):
        logger.debug("action was a string — converting to dict: %r", action_raw)
        action_raw = {"predicate": action_raw, "object": ""}
    if isinstance(condition_raw, str):
        logger.debug("condition was a string — converting to dict: %r", condition_raw)
        condition_raw = {"text": condition_raw, "type": "none"}

    # 确保为字典（而非列表或其他类型）。
    if not isinstance(subject_raw, dict):
        subject_raw = {}
    if not isinstance(action_raw, dict):
        action_raw = {}
    if not isinstance(condition_raw, dict):
        condition_raw = {}

    # --- 步骤 2：规范化枚举字符串值 ---
    # 大语言模型可能以不同大小写返回角色值。
    # 映射为 LegalRole 期望的规范小写形式。
    role_str = subject_raw.get("role", "other")
    role_enum = coerce_legal_role(role_str)

    # 条件类型规范化——同理。
    cond_type_str = condition_raw.get("type", "none")
    cond_type_enum = coerce_condition_type(cond_type_str)

    # --- 步骤 3：构建候选字典 ---
    candidate = {
        "subject": {
            "text": str(subject_raw.get("text", "")).strip(),
            "role": role_enum,
        },
        "action": {
            "predicate": str(action_raw.get("predicate", "")).strip(),
            "object": str(action_raw.get("object", "")).strip(),
        },
        "condition": {
            "text": str(condition_raw.get("text", "")).strip(),
            "type": cond_type_enum,
        },
    }

    # --- 步骤 4：Pydantic 校验 ---
    # 捕获剩余的模式违规（错误类型、缺失必填字段等），
    # 若数据无法强制转换则抛出 ValidationError。
    triplet = LegalTriplet.model_validate(candidate)

    # --- 步骤 5：校验后规范化 ---
    # Pydantic 不强制执行的额外清理：
    # - 若条件文本为空或仅含空白，将类型设为 NONE
    #   （空条件不应具有非 NONE 类型）。
    # - 若条件类型为 NONE 但存在条件文本，保留文本
    #   （类型赋值有误但文本可能仍有价值）。
    normalized_condition = triplet.condition
    if normalized_condition.text.strip() == "":
        # 无有意义条件文本——为一致性将类型重置为 NONE。
        if normalized_condition.type != ConditionType.NONE:
            logger.debug(
                "Resetting condition type from %s to NONE (empty text)",
                normalized_condition.type.value,
            )
        normalized_condition = Condition(text="", type=ConditionType.NONE)

    # 使用规范化后的条件构建最终三元组。
    # LegalTriplet 为冻结模型，需创建新实例。
    triplet = LegalTriplet(
        subject=triplet.subject,
        action=triplet.action,
        condition=normalized_condition,
    )

    return triplet


# -------------------------------------------------------------------------
# 枚举强制转换辅助函数
# -------------------------------------------------------------------------


def coerce_legal_role(raw: str) -> LegalRole:
    """将原始字符串强制转换为 ``LegalRole`` 枚举值。

    处理大语言模型输出的常见变体：大小写差异、下划线与空格、部分匹配。
    若字符串无法匹配则回退到 ``LegalRole.OTHER``。

    参数:
        raw:  大语言模型返回的原始角色字符串（如 ``"OBLIGOR"``、
              ``"right_holder"``、``"Right Holder"``）。

    返回:
        匹配的 ``LegalRole`` 枚举值。
    """
    if not raw:
        return LegalRole.OTHER

    normalized = raw.strip().lower().replace(" ", "_").replace("-", "_")

    # 按值直接查找。
    try:
        return LegalRole(normalized)
    except ValueError:
        pass

    # 模糊匹配：检查原始字符串是否包含已知角色关键词。
    fuzzy_map = {
        "obligor": LegalRole.OBLIGOR,
        "oblig": LegalRole.OBLIGOR,
        "right": LegalRole.RIGHT_HOLDER,
        "holder": LegalRole.RIGHT_HOLDER,
        "prohibit": LegalRole.PROHIBITED_PARTY,
        "indemnif": LegalRole.INDEMNIFYING_PARTY,
    }
    for keyword, role in fuzzy_map.items():
        if keyword in normalized:
            logger.debug(
                "Fuzzy-matched role '%s' -> %s (keyword: '%s')",
                raw,
                role.value,
                keyword,
            )
            return role

    logger.debug("Could not coerce role '%s' — falling back to OTHER", raw)
    return LegalRole.OTHER


def coerce_condition_type(raw: str) -> ConditionType:
    """将原始字符串强制转换为 ``ConditionType`` 枚举值。

    处理大小写差异、下划线与空格及大语言模型输出的常见变体。
    若字符串无法匹配则回退到 ``ConditionType.NONE``。

    参数:
        raw:  原始条件类型字符串（如 ``"TEMPORAL"``、
              ``"Trigger"``、``"exception"``、``"none"``）。

    返回:
        匹配的 ``ConditionType`` 枚举值。
    """
    if not raw:
        return ConditionType.NONE

    normalized = raw.strip().lower().replace(" ", "_").replace("-", "_")

    # 按值直接查找。
    try:
        return ConditionType(normalized)
    except ValueError:
        pass

    # 对大语言模型常见变体进行模糊匹配。
    fuzzy_map = {
        "temporal": ConditionType.TEMPORAL,
        "time": ConditionType.TEMPORAL,
        "trigger": ConditionType.TRIGGER,
        "event": ConditionType.TRIGGER,
        "conditional": ConditionType.TRIGGER,
        "except": ConditionType.EXCEPTION,
        "exception": ConditionType.EXCEPTION,
        "carve": ConditionType.EXCEPTION,
        "none": ConditionType.NONE,
        "null": ConditionType.NONE,
        "empty": ConditionType.NONE,
        "no": ConditionType.NONE,
    }
    for keyword, ctype in fuzzy_map.items():
        if keyword in normalized:
            logger.debug(
                "Fuzzy-matched condition type '%s' -> %s (keyword: '%s')",
                raw,
                ctype.value,
                keyword,
            )
            return ctype

    logger.debug(
        "Could not coerce condition type '%s' — falling back to NONE",
        raw,
    )
    return ConditionType.NONE


# -------------------------------------------------------------------------
# 回退构建
# -------------------------------------------------------------------------


def build_fallback_triplet(
    clause: str,
    error: str,
) -> LegalTriplet:
    """抽取完全失败时构建最小回退三元组。

    构造所有字段均为空/默认值的 ``LegalTriplet``。
    使下游处理（批量抽取、评估、错误分析）在大语言模型对某条款失败时仍能继续。

    回退三元组在日志中保留原始条款文本，但不嵌入三元组
    （``schema.py`` 中的 ``LegalTriplet`` 模型无 ``clause`` 字段）。
    原始条款与错误上下文以 WARNING 级别记录，供事后调试。

    参数:
        clause:  抽取失败的原始条款文本。
        error:   供日志记录的可读错误描述。

    返回:
        主体、动作、条件均为空（各字段为合理默认值）的 ``LegalTriplet``。
    """
    logger.warning(
        "Using fallback triplet for clause (len=%d): %s | Error: %s",
        len(clause),
        clause[:100] + ("..." if len(clause) > 100 else ""),
        error,
    )

    return LegalTriplet(
        subject=Subject(
            text="",
            role=LegalRole.OTHER,
        ),
        action=Action(
            predicate="",
            object="",
        ),
        condition=Condition(
            text="",
            type=ConditionType.NONE,
        ),
    )
