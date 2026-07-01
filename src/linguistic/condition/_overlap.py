"""
条件跨度重叠计算与主句重叠检测。
"""

from __future__ import annotations

from typing import Set

from src.extraction.schema import DependencyTree, ConditionSpan
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_condition_overlap(
    llm_condition_text: str,
    ud_condition_span: ConditionSpan,
    tree: DependencyTree,
) -> float:
    """计算大语言模型条件文本与 UD 推导条件跨度之间的
    词元级重叠。

    在规范化后的词元集上使用 Jaccard 相似度（交并比）。
    这是评估条件边界准确性的主要指标（校验器步骤 5 使用）。

    IoU = |tokens(LLM) ∩ tokens(UD)| / |tokens(LLM) ∪ tokens(UD)|

    IoU 越高表示边界抽取越准确：
      IoU = 1.0  -> 完全匹配（不同分词器下少见）
      IoU >= 0.8 -> 优秀匹配（可接受）
      IoU >= 0.5 -> 中等匹配（需检查边界问题）
      IoU < 0.5  -> 匹配差（可能为边界错误）

    参数：
        llm_condition_text: 大语言模型抽取的条件文本。
        ud_condition_span: UD 推导的 ConditionSpan。
        tree: 依存树（供分词参考）。

    返回：
        0.0 到 1.0 之间的 Jaccard 相似度分数。
    """
    # 将双方文本分词为小写词元集合。
    # 对大语言模型输出使用简单空白分词，
    # 因其可能与 Stanza 分词不完全一致。
    llm_tokens = set(llm_condition_text.lower().split())

    # UD 词元使用跨度中实际 Stanza 词元文本。
    ud_tokens = set()
    for token_idx in ud_condition_span.tokens:
        token = tree.get_token(token_idx)
        if token is not None:
            ud_tokens.add(token.text.lower())

    if not llm_tokens and not ud_tokens:
        # 双方皆空 —— 均无条件。视为匹配。
        return 1.0
    if not llm_tokens or not ud_tokens:
        # 一方有条件、另一方无 —— 完全不匹配。
        return 0.0

    intersection = llm_tokens & ud_tokens
    union = llm_tokens | ud_tokens

    iou = len(intersection) / len(union) if union else 0.0
    logger.debug(
        "Condition IoU: %.3f (LLM: %d tokens, UD: %d tokens, "
        "intersection: %d, union: %d)",
        iou, len(llm_tokens), len(ud_tokens),
        len(intersection), len(union),
    )
    return iou


def is_condition_in_main_clause(
    condition_span: ConditionSpan,
    tree: DependencyTree,
) -> bool:
    """启发式检查：条件跨度是否与主句主语或谓词区域重叠？

    若条件跨度包含主谓词或其主语，
    提取器可能错误划定边界（过度抽取）。

    参数：
        condition_span: 提取的条件跨度。
        tree: 依存树。

    返回：
        条件似乎包含主句成分时返回 True（提示边界错误）。
    """
    root_idx = tree.root_index
    if root_idx is None:
        return False

    # 获取主谓词区域（谓词及其直接论元）。
    main_indices: Set[int] = {root_idx}
    for child in tree.get_children(root_idx):
        if child.deprel in ("nsubj", "nsubj:pass", "obj", "aux", "aux:pass"):
            main_indices.add(child.index)

    condition_indices = set(condition_span.tokens)
    overlap = main_indices & condition_indices

    if overlap:
        logger.debug(
            "Condition span overlaps with main clause elements: %s",
            [tree.get_token(i).text if tree.get_token(i) else "?" for i in overlap],
        )
        return True
    return False
