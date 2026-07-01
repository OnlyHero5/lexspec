"""
内部转换：Stanza Word -> LexSpec Token。

Stanza API 与 LexSpec 数据模型之间的桥梁。
所有 Stanza 相关细节均封装于此。
"""

from __future__ import annotations

from typing import List, Dict

from src.extraction.schema import Token
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _convert_sentence_to_tokens(self, sentence) -> List[Token]:
    """将 Stanza Sentence 转换为 LexSpec Token 对象列表。

    Stanza API 与 LexSpec 数据模型之间的桥梁。
    所有 Stanza 相关细节（word.to_dict()、word.feats 解析）
    均封装于此，其他模块无需导入 Stanza。

    处理的 UD 约定：
      - word.id: 多词词元可能为元组（如 (1, '-')），
                 或为常规词元的整数。跳过 MWT 占位符。
      - word.head: Stanza 为 0 基，转换为 LexSpec 的 1 基。
                   LexSpec 中 head=0 表示「此为根」。
      - word.feats: 形如 "Tense=Past|Voice=Pass" 的字符串解析为字典。

    参数：
        sentence: 包含 Word 对象的 Stanza Sentence。

    返回：
        带完整 UD 标注的 Token 对象列表。
    """
    tokens: List[Token] = []

    for word in sentence.words:
        # Stanza 用元组 ID 表示多词词元（MWT），
        # 如 (1, '-') 为 MWT 头，(1, 1)、(1, 2) 为展开词。
        # 跳过 MWT 头，保留带实际句法标注的展开词。
        if isinstance(word.id, tuple):
            # word.id 为 (head_index, sub_index)
            # 若 sub_index 为 0 或无效则跳过（MWT 占位符）
            if len(word.id) == 2 and word.id[1] == 0:
                continue

            # 使用第一个分量作为词元索引。
            # Stanza 中，对展开的 MWT 词元，元组第一个元素
            # 为句级词元位置。
            token_index = word.id[0] if len(word.id) > 0 else 0
        else:
            token_index = int(word.id)

        # Stanza 内部使用 0 基 head 索引，但 UD CoNLL-U
        # 格式（及 DependencyTree 模型）使用 1 基索引，
        # 其中 0 表示「根」。相应转换：
        #   Stanza head=0 -> LexSpec head=0（根）
        #   Stanza head=1 -> LexSpec head=1（第一个词元）
        head_index = int(word.head) if word.head is not None else 0

        # 从 Stanza 字符串格式解析形态特征。
        # Stanza 将 feats 表示为 "Tense=Past|Voice=Pass" 或 None。
        feats: Dict[str, str] = {}
        if word.feats:
            for feat_str in word.feats.split("|"):
                if "=" in feat_str:
                    key, val = feat_str.split("=", 1)
                    feats[key.strip()] = val.strip()

        token = Token(
            index=token_index,
            text=word.text,
            lemma=word.lemma if word.lemma else word.text.lower(),
            upos=word.upos if word.upos else "X",
            xpos=word.xpos if word.xpos else "",
            deprel=word.deprel if word.deprel else "dep",
            head=head_index,
            feats=feats,
        )
        tokens.append(token)

    return tokens
