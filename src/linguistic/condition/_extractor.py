"""
ConditionExtractor 的主提取接口。
"""

from __future__ import annotations

from typing import Optional, List

from src.extraction.schema import (
    DependencyTree,
    ConditionSpan,
    ConditionType,
)
from src.linguistic.ud_features import find_advcl_with_mark
from src.utils.logging import get_logger

logger = get_logger(__name__)


def extract(
    self,
    tree: DependencyTree,
    predicate_idx: int,
) -> Optional[ConditionSpan]:
    """提取谓词的主条件从句。

    若存在多个条件从句，返回第一个
    （通常为句法上最突出者 —— 表层顺序上最接近主谓词）。

    法律合同中，每个主谓词对应多个条件从句不常见但可能
    （如 "If X and upon Y, Seller shall Z"）。
    使用 extract_all() 获取全部条件。

    参数：
        tree: 依存树。
        predicate_idx: 主谓词的 1 基索引。

    返回：
        含文本、类型与标记信息的 ConditionSpan，或
        未检测到条件从句时返回 None。

    抛出：
        ValueError: predicate_idx 在树中未找到时。
    """
    spans = extract_all(self, tree, predicate_idx)
    if spans:
        # 返回第一个（最突出的）条件。
        return spans[0]
    return None


def extract_all(
    self,
    tree: DependencyTree,
    predicate_idx: int,
) -> List[ConditionSpan]:
    """提取全部条件从句（含多个条件）。

    同一谓词上存在多个条件时（如
    "If the Buyer defaults, and unless waived by the Seller, the
    Seller may terminate"），返回全部条件。

    参数：
        tree: 依存树。
        predicate_idx: 主谓词的 1 基索引。

    返回：
        找到的全部 ConditionSpan 对象列表。未检测到条件时为空列表。

    抛出：
        ValueError: predicate_idx 在树中未找到时。
    """
    predicate = tree.get_token(predicate_idx)
    if predicate is None:
        raise ValueError(
            f"Predicate index {predicate_idx} not found in tree "
            f"(token count: {tree.token_count})"
        )

    # 步骤 1：从 UD 树获取原始 advcl+mark 跨度。
    # 这些跨度未分类 —— condition_type 为 NONE。
    raw_spans = find_advcl_with_mark(
        tree, predicate_idx, self._marker_list
    )

    if not raw_spans:
        logger.debug(
            "No advcl+mark condition clauses found for predicate '%s' at index %d",
            predicate.lemma, predicate_idx,
        )
        return []

    # 步骤 2：根据标记词分类各跨度。
    classified_spans: List[ConditionSpan] = []
    for span in raw_spans:
        if span.mark_token is None:
            # 不应发生 —— find_advcl_with_mark 仅返回
            # 带标记词元的跨度。防御性检查。
            logger.debug("Skipping span without mark token")
            continue

        mark_text = span.mark_text.lower().strip()
        condition_type = _classify_condition(self, mark_text)

        # 通过复制原始跨度字段并设置 condition_type，
        # 创建已分类的 ConditionSpan。
        classified = ConditionSpan(
            tokens=span.tokens,
            text=span.text,
            deprel=span.deprel,
            mark_token=span.mark_token,
            condition_type=condition_type,
            mark_text=span.mark_text,
        )
        classified_spans.append(classified)

        logger.debug(
            "Classified condition: type=%s, mark='%s', span='%s...'",
            condition_type.value, mark_text, span.text[:80],
        )

    return classified_spans


def _classify_condition(self, mark_text: str) -> ConditionType:
    """根据标记词分类条件。

    使用从 constraints.yaml 加载的法律领域分类体系：

    TRIGGER（事件触发）：
      "if" —— 典型条件："IF Buyer defaults, Seller may..."
      "provided that" —— 限定条件："provided that the
        Company receives notice..."
      "in the event that" —— 或有事项："in the event that any
        representation proves false..."
      "so long as" —— 持续条件："so long as no Event of
        Default has occurred..."

    TEMPORAL（时间）：
      "when" —— 时间触发："WHEN the Closing occurs..."
      "upon" —— 事件触发："UPON delivery of the Goods..."
      "after" —— 顺序："AFTER the Closing Date..."
      "within" —— 有界："WITHIN 30 days of the date hereof..."

    EXCEPTION（范围限制）：
      "unless" —— 否定条件："UNLESS otherwise agreed..."
      "except" ——  carve-out："EXCEPT as provided in Section 2.3..."
      "notwithstanding" —— 覆盖："NOTWITHSTANDING anything to
        the contrary..."

    对 Stanza 仅标注首词的多词标记
    （如 "provided that" 中的 "provided"），先查单词，
    再尝试前缀匹配。

    参数：
        mark_text: 标记词文本（小写、去首尾空白）。

    返回：
        ConditionType 枚举值。未识别的标记词默认为 TRIGGER
        （保守假设：未知 advcl+mark 最可能为条件构造）。
    """
    # 在分类体系中精确匹配。
    if mark_text in self._markers:
        return self._markers[mark_text]

    # 多词标记：检查 mark_text 是否为某已知标记的前缀。
    # 例如 "provided" 匹配 "provided that"。
    for marker, ctype in self._markers.items():
        if marker.startswith(mark_text + " "):
            logger.debug(
                "Matched mark '%s' to multi-word marker '%s' -> %s",
                mark_text, marker, ctype.value,
            )
            return ctype

    # 回退：也检查标记是否以 mark_text 开头
    #（Stanza 词元包含超出预期的内容时）。
    # 处理罕见分词边界情况。
    for marker, ctype in self._markers.items():
        if mark_text.startswith(marker + " ") or mark_text == marker:
            return ctype

    # 未知标记词 —— 默认为 TRIGGER。
    # 理由：法律英语中，大多数未识别的 advcl+mark
    # 模式类似条件（触发）构造。
    # 时间标记（"when"、"upon"）通常能被 Stanza 可靠解析。
    # 若标记未识别，更可能是较少见的触发词，
    # 而非未识别的时间或例外标记。
    logger.debug(
        "Unrecognized mark word '%s' — defaulting to TRIGGER", mark_text
    )
    return ConditionType.TRIGGER
