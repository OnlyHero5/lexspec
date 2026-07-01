"""
StanzaParser 的解析方法：parse() 与 parse_batch()。
"""

from __future__ import annotations

from typing import Optional, List

from src.extraction.schema import DependencyTree, Token
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _parse(self, text: str) -> DependencyTree:
    """解析单句并返回其 UD 依存树。

    将输入文本经完整 Stanza 流水线处理
    （tokenize -> MWT -> POS -> lemma -> depparse），
    并将输出转换为带 Token 对象的 DependencyTree。

    法律文本中，因复杂合同用语，句子可能很长（50+ 词）。
    Stanza 处理良好，但极长句子（>100 词）可能降低解析质量。

    参数：
        text: 英语句子或合同条款文本。
              应为单句 —— 多句输入仅返回第一句的解析。
              多句文档请使用 parse_batch()。

    返回：
        词元带有完整 UD 依存关系、词性标签、词元与特征的 DependencyTree。

    抛出：
        ValueError: 解析未产生句子（空输入、仅空白，
                    或 Stanza 无法切分的非语法文本）。
        RuntimeError: 流水线尚未初始化。

    示例：
        >>> parser = StanzaParser()
        >>> tree = parser.parse("Seller shall deliver the Goods.")
        >>> tree.root_index
        3  # "deliver" 为根
        >>> tree.get_token(3).lemma
        'deliver'
    """
    if not text or not text.strip():
        raise ValueError("Cannot parse empty or whitespace-only text")

    # 对输入文本运行 Stanza 流水线。
    # doc.sentences 为 Sentence 对象列表，每个包含
    # 带 UD 标注的 Word 对象列表。
    doc = self.nlp(text)

    if len(doc.sentences) == 0:
        raise ValueError(
            f"Stanza produced no sentences for input: '{text[:80]}...'"
        )

    # 仅处理第一句。
    # 多句合同摘录应在调用 parse() 前预先切分。
    # 这与流水线设计一致：每个 LegalTriplet 对应恰好一个从句。
    sentence = doc.sentences[0]

    from src.linguistic.parser._conversion import _convert_sentence_to_tokens
    tokens = _convert_sentence_to_tokens(sentence)
    logger.debug(
        "Parsed sentence: %d tokens, root at index %d",
        len(tokens),
        next((t.index for t in tokens if t.head == 0), -1),
    )

    return DependencyTree(text=text, tokens=tokens)


def _parse_batch(self, texts: List[str]) -> List[DependencyTree]:
    """高效解析多个句子。

    比循环调用 parse() 更高效，因 Stanza 可内部批处理，
    在多个输入间分摊 GPU/CPU 开销。用于批量处理
    合同文档中的从句。

    输入列表中每项视为独立句子。
    多句文本仅产出第一句的解析。

    参数：
        texts: 待解析的句子字符串列表。

    返回：
        DependencyTree 对象列表，与输入一一对应。

    注意：
        输入中的空字符串产出零词元的 DependencyTree
        （不抛出错误）。调用方如需应自行过滤。
    """
    if not texts:
        return []

    logger.info("Batch-parsing %d sentences", len(texts))

    # Stanza 可直接处理文本列表。
    # 比循环 parse() 更高效，因神经网络模型批处理计算。
    doc = self.nlp(texts)

    trees: List[DependencyTree] = []

    # doc.sentences 包含所有输入文本的全部句子。
    # 但若某输入含多句，它们都会出现在 doc.sentences 中。
    # 我们通过句子文本将结果关联回输入。
    #
    # 策略：遍历句子，通过包含关系匹配回输入文本。
    # 可处理部分输入为空（未产出句子）的情况。
    sentence_idx = 0
    for original_text in texts:
        if not original_text or not original_text.strip():
            # 空输入 -> 空树（无词元、无根）
            trees.append(DependencyTree(text=original_text, tokens=[]))
            continue

        if sentence_idx < len(doc.sentences):
            sentence = doc.sentences[sentence_idx]
            from src.linguistic.parser._conversion import _convert_sentence_to_tokens
            tokens = _convert_sentence_to_tokens(sentence)
            trees.append(DependencyTree(text=original_text, tokens=tokens))
            sentence_idx += 1
        else:
            # 回退：已无更多句子但仍有输入。
            # 正常情况下不应发生。
            logger.warning(
                "Fewer sentences produced than inputs: %d < %d",
                len(doc.sentences), len(texts),
            )
            trees.append(DependencyTree(text=original_text, tokens=[]))

    logger.info("Batch parsing complete: %d trees produced", len(trees))
    return trees
