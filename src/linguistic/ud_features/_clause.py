"""
从句关系：find_advcl_with_mark、_matches_marker、_extract_condition_span_text。
"""

from __future__ import annotations

from typing import List

from src.extraction.schema import DependencyTree, Token, ConditionSpan, ConditionType
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _matches_marker(mark_text: str, markers: List[str]) -> bool:
    """检查标记词文本是否匹配任一已知条件标记。

    通过检查标记文本是否为多词短语的首词
    （如 "provided that" 中的 "provided"），
    或是否为单词标记的精确匹配，处理多词标记。

    参数：
        mark_text: 小写标记词文本。
        markers: 来自 constraints.yaml 的标记字符串列表。

    返回：
        mark_text 匹配任一标记时返回 True。
    """
    for marker in markers:
        if marker == mark_text:
            return True
        # 多词标记检查："provided" 匹配 "provided that"
        if " " in marker and marker.startswith(mark_text + " "):
            return True
        # 也检查：mark_text 可能为完整多词短语
        #（若 Stanza 将其视为单一词元）。少见但可能
        # 发生于固定表达。
        if mark_text.startswith(marker + " ") or mark_text == marker:
            return True
    return False


def _extract_condition_span_text(
    tree: DependencyTree,
    advcl_head_idx: int,
    mark_token: Token,
) -> str:
    """提取条件从句的完整文本。

    跨度从标记词开始，包含整个 advcl 子树。确保捕获完整条件，如：
      "if Buyer fails to pay any installment when due"
    而非仅：
      "if Buyer fails"

    策略：收集 advcl 子树中全部词元，再向左遍历
    advcl 头以找到标记词及标记与 advcl 头之间的词元。
    按索引排序以重建表层顺序。

    参数：
        tree: 依存树。
        advcl_head_idx: advcl 头（从句主要动词）的索引。
        mark_token: 标记词词元。

    返回：
        表层顺序的完整条件从句文本。
    """
    # 收集 advcl 子树中全部词元索引。
    subtree_indices = set(tree._collect_subtree(advcl_head_idx))

    # 部分解析中标记词可能在 advcl 子树外
    #（UD 允许 mark 作为头的依存）。确保包含。
    subtree_indices.add(mark_token.index)

    # 也包含标记与 advcl 头之间可能非直接后代的词元
    #（如插入副词）。
    min_idx = min(mark_token.index, advcl_head_idx)
    max_idx = max(mark_token.index, advcl_head_idx)
    for i in range(min_idx, max_idx + 1):
        subtree_indices.add(i)

    # 按索引排序以重建表层顺序。
    sorted_indices = sorted(subtree_indices)
    tokens_sorted = [
        tree.get_token(i) for i in sorted_indices
    ]
    tokens_sorted = [t for t in tokens_sorted if t is not None]

    return " ".join(t.text for t in tokens_sorted)


def find_advcl_with_mark(
    tree: DependencyTree,
    predicate_idx: int,
    condition_markers: List[str],
) -> List[ConditionSpan]:
    """查找带条件标记从属连词的状态语从句。

    UD: advcl(predicate, clause_head) —— 状语从句修饰语。
        mark(clause_head, marker)   —— 从属连词。

    法律文本中，advcl+mark 标识条件从句：
      "IF BUYER FAILS TO PAY, Seller may terminate"
        -> advcl(terminate, fails)
        -> mark(fails, If)

    标记词决定条件类型：
      - "if"、"provided that" -> TRIGGER（事件条件）
      - "when"、"upon"、"after" -> TEMPORAL（时间条件）
      - "unless"、"except" -> EXCEPTION（范围限制）

    本函数不分类条件 —— 仅提取跨度。
    分类由 ConditionExtractor 使用 constraints.yaml 分类体系完成。

    参数：
        tree: 依存树。
        predicate_idx: 主谓词的 1 基索引。
        condition_markers: 标示条件的小写标记词列表（来自 constraints.yaml）。

    返回：
        含文本、索引与标记信息的 ConditionSpan 列表。
        未找到带条件标记的 advcl 时为空列表。
    """
    result: List[ConditionSpan] = []

    # 步骤 1：查找谓词的全部 advcl 子节点。
    # advcl 为修饰主动词的状语从句。
    # 法律文本中最常见的 advcl 类型为条件。
    advcl_children = tree.get_children(predicate_idx, deprel="advcl")

    if not advcl_children:
        # 无任何状语从句 —— 无条件。
        return result

    logger.debug(
        "Found %d advcl children of predicate at index %d",
        len(advcl_children), predicate_idx,
    )

    # 步骤 2：对每个 advcl 子节点检查 mark（从属连词）。
    for advcl_head in advcl_children:
        # 标记词为 advcl 头动词的依存。
        # 例："If Buyer fails to pay" -> mark(fails, If)
        mark_children = tree.get_children(advcl_head.index, deprel="mark")

        if not mark_children:
            # 无显式 mark 的 advcl —— 可能为不定式
            # 或分词从句。法律英语中较少见，
            # 通常不表达条件。
            # 例："Seller agrees [to deliver the Goods]"
            #   -> advcl(agrees, deliver)，无标记词。
            continue

        mark_token = mark_children[0]

        # 步骤 3：检查标记词是否为已知条件标记。
        # 小写规范化 —— 法律文本大小写混用。
        mark_text_lower = mark_token.text.lower().strip()

        # 多词标记如 "provided that"、"in the event that"
        # 可能仅首词为 mark 词元。对照单词与
        # 多词首词模式检查。
        if not _matches_marker(
            mark_text_lower, condition_markers
        ):
            # 非条件标记 —— 可能为时间（"when"）、
            # 原因（"because"）或目的（"so that"）。
            # 跳过 —— 法律意义上非条件。
            logger.debug(
                "Skipping advcl at index %d: mark '%s' not a condition marker",
                advcl_head.index, mark_token.text,
            )
            continue

        # 步骤 4：提取完整条件跨度文本。
        # 包含标记词与整个 advcl 子树以捕获完整条件边界。
        # 对 "if Buyer fails to pay any installment when due"，
        # 需要全部文本，而非仅 "if Buyer fails"。
        span_text = _extract_condition_span_text(
            tree, advcl_head.index, mark_token
        )

        # 构建 ConditionSpan（condition_type 初始为 NONE —
        # 由 ConditionExtractor 分类）。
        condition_span = ConditionSpan(
            tokens=sorted(tree._collect_subtree(advcl_head.index)),
            text=span_text,
            deprel="advcl",
            mark_token=mark_token,
            condition_type=ConditionType.NONE,  # 稍后分类
            mark_text=mark_token.text,
        )
        result.append(condition_span)

        logger.debug(
            "Extracted condition span: '%s' (mark: '%s')",
            span_text[:80], mark_token.text,
        )

    return result
