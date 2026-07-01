"""
校验步骤 5：条件校验。

使用 IoU 重叠将大语言模型抽取的条件与 UD 条件跨度比对。
"""

from __future__ import annotations

from typing import Optional, List

from src.extraction.schema import (
    LegalTriplet,
    DependencyTree,
    FieldCorrection,
    ConditionType,
    ConditionSpan,
)
from src.linguistic.condition_extractor import ConditionExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)


def step5_validate_condition(
    triplet: LegalTriplet,
    tree: DependencyTree,
    predicate_idx: int,
    condition_extractor: ConditionExtractor,
    condition_overlap: float,
    corrections: List[FieldCorrection],
    feedback_parts: List[str],
) -> Optional[ConditionSpan]:
    """步骤 5：校验条件字段。

    条件校验是最复杂的步骤，因涉及跨度边界比对，
    而非仅词元匹配。主要问题：

    1. 条件遗漏：UD 找到条件从句但大语言模型
       将 condition.type 标为 NONE。大语言模型遗漏条件。

    2. 条件过度抽取：大语言模型标有条件但 UD 未找到。
       大语言模型幻觉条件或将主句内容当作条件。

    3. 条件边界错误：UD 与大语言模型均找到条件
       但跨度差异显著（IoU 低于阈值）。
       大语言模型可能包含主句词元或截断条件从句。

    4. 条件类型错误：双方均找到条件但大语言模型
       分类错误（如 TEMPORAL 与 TRIGGER）。

    策略：
      - 通过 ConditionExtractor 提取 UD 条件跨度。
      - 使用词元重叠（IoU）与大语言模型条件文本比对。
      - IoU >= 阈值：接受（轻微边界差异可接受）。
      - IoU < 阈值：添加修正或触发 Reflexion。

    参数：
        triplet: 大语言模型抽取的三元组。
        tree: 依存树。
        predicate_idx: 谓词的 1 基索引。
        condition_extractor: ConditionExtractor 实例。
        condition_overlap: 条件跨度匹配的最小 IoU 阈值。
        corrections: 追加 FieldCorrection 的列表。
        feedback_parts: 追加反馈字符串的列表。

    返回：
        UD ConditionSpan，未检测到条件时返回 None。
    """
    llm_has_condition = (
        triplet.condition.text
        and triplet.condition.text.strip()
        and triplet.condition.type != ConditionType.NONE
    )

    # 提取该谓词的全部 UD 条件跨度。
    ud_spans = condition_extractor.extract_all(tree, predicate_idx)

    # 情况 1：大语言模型与 UD 均无条件。
    if not ud_spans and not llm_has_condition:
        return None

    # 情况 2：UD 找到条件但大语言模型遗漏（遗漏）。
    if ud_spans and not llm_has_condition:
        primary_span = ud_spans[0]
        feedback_parts.append(
            f"Condition omitted: UD parse identifies a "
            f"{primary_span.condition_type.value} condition clause: "
            f"'{primary_span.text}'. The LLM marked this clause as "
            f"having no condition."
        )
        corrections.append(FieldCorrection(
            field="condition.text",
            original="",
            corrected=primary_span.text,
            reason=(
                f"UD parse identifies a {primary_span.condition_type.value} "
                f"condition clause via advcl relation with mark='{primary_span.mark_text}': "
                f"'{primary_span.text}'. The LLM omitted this condition entirely."
            ),
        ))
        corrections.append(FieldCorrection(
            field="condition.type",
            original=ConditionType.NONE.value,
            corrected=primary_span.condition_type.value,
            reason=(
                f"Condition type derived from mark word "
                f"'{primary_span.mark_text}' -> {primary_span.condition_type.value}."
            ),
        ))
        # 返回 UD 跨度以便填充证据对象。
        # 因有 UD 证据，状态将为 CORRECTED。
        return primary_span

    # 情况 3：大语言模型有条件但 UD 未找到（过度抽取）。
    if not ud_spans and llm_has_condition:
        feedback_parts.append(
            f"Condition over-extraction: the LLM extracted condition "
            f"'{triplet.condition.text}' but no condition clause was "
            f"found in the UD parse. The LLM may have extracted main "
            f"clause content as a condition."
        )
        # 检查大语言模型条件文本是否与主句重叠。
        # 有助于区分幻觉与抽取 UD 未标为 advcl 的时间状语。
        correction_applied = False
        for token_idx in range(1, tree.token_count + 1):
            token = tree.get_token(token_idx)
            if token is None:
                continue
            if token.text.lower() in triplet.condition.text.lower():
                # 条件文本包含主句词元。
                # 可能为过度抽取 —— 移除条件。
                correction_applied = True
                break

        if correction_applied:
            corrections.append(FieldCorrection(
                field="condition.text",
                original=triplet.condition.text,
                corrected="",
                reason=(
                    "No condition clause found in UD parse. The LLM-extracted "
                    "condition text includes main clause tokens, suggesting "
                    "over-extraction. Condition removed."
                ),
            ))
            corrections.append(FieldCorrection(
                field="condition.type",
                original=triplet.condition.type.value,
                corrected=ConditionType.NONE.value,
                reason="No condition clause present in UD parse.",
            ))
        # 若无主句重叠，大语言模型可能抽取了
        # UD 解析失败的真实条件。无法置信修正 —— 属解析错误。
        # 仍返回 None 表示「未找到 UD 条件」。
        return None

    # 情况 4：大语言模型与 UD 均有条件 —— 比对跨度。
    if ud_spans and llm_has_condition:
        primary_span = ud_spans[0]

        # 计算大语言模型条件文本与 UD 跨度的重叠。
        overlap = ConditionExtractor.compute_condition_overlap(
            triplet.condition.text, primary_span, tree
        )

        if overlap >= condition_overlap:
            # 重叠足够。检查条件类型是否匹配。
            if triplet.condition.type != primary_span.condition_type:
                # 类型不匹配 —— 修正类型。
                corrections.append(FieldCorrection(
                    field="condition.type",
                    original=triplet.condition.type.value,
                    corrected=primary_span.condition_type.value,
                    reason=(
                        f"Condition type derived from mark word "
                        f"'{primary_span.mark_text}' should be "
                        f"{primary_span.condition_type.value}, not "
                        f"{triplet.condition.type.value}."
                    ),
                ))
            # 条件跨度可接受。
            return primary_span

        # 重叠低于阈值 —— 边界错误。
        feedback_parts.append(
            f"Condition boundary error: LLM condition span "
            f"'{triplet.condition.text}' has low overlap (IoU={overlap:.2f}) "
            f"with UD condition span '{primary_span.text}'. "
            f"The LLM may have truncated or extended the condition clause."
        )
        corrections.append(FieldCorrection(
            field="condition.text",
            original=triplet.condition.text,
            corrected=primary_span.text,
            reason=(
                f"Condition boundary mismatch (IoU={overlap:.2f} < "
                f"threshold={condition_overlap}). UD parse identifies "
                f"the condition via advcl relation: '{primary_span.text}'."
            ),
        ))
        if triplet.condition.type != primary_span.condition_type:
            corrections.append(FieldCorrection(
                field="condition.type",
                original=triplet.condition.type.value,
                corrected=primary_span.condition_type.value,
                reason=(
                    f"Condition type corrected to "
                    f"{primary_span.condition_type.value} based on mark word "
                    f"'{primary_span.mark_text}'."
                ),
            ))
        return primary_span

    return None
