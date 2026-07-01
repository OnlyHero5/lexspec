"""
否定检测：_is_negated 与 _has_lexical_negation。
"""

from __future__ import annotations

from src.extraction.schema import DependencyTree
from src.linguistic.ud_features import find_negation
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _is_negated(self, tree: DependencyTree, predicate_idx: int) -> bool:
    """检查谓词是否被否定。

    否定检测按优先级查找三种模式：

    1. 直接 neg 依存：neg(predicate, not)
       "shall NOT assign" -> neg(assign, not)
       法律英语中最常见的否定模式。

    2. aux 依存上的否定：
       部分解析将 "not" 附在助动词而非主动词：
       "shall NOT" 中 NOT 通过 neg 关系依附 "shall"。
       检查 aux 词元的子节点。

    3. 谓词附近的词项否定词元：
       - "no" 作限定词修饰主语：
         "NO party shall assign..." —— 否定在 "party" 上，
         非直接依附动词，但构成禁止。
       - "neither ... nor" 结构
       - "never" 作副词

    参数：
        tree: 依存树。
        predicate_idx: 谓词的 1 基索引。

    返回：
        检测到否定时返回 True。
    """
    # 模式 1：谓词上的直接 neg 依存。
    neg_token = find_negation(tree, predicate_idx)
    if neg_token is not None:
        logger.debug(
            "Negation detected: neg(%s, %s) at predicate index %d",
            tree.get_token(predicate_idx).text if tree.get_token(predicate_idx) else "?",
            neg_token.text,
            predicate_idx,
        )
        return True

    # 模式 2：检查是否有助动词带有 neg 依存。
    # 部分解析风格中 "not" 附在助动词：
    # aux(deliver, shall) + neg(shall, not)
    for child in tree.get_children(predicate_idx):
        if child.deprel in ("aux", "aux:pass"):
            grandchild_neg = tree.get_children(child.index, deprel="neg")
            if grandchild_neg:
                logger.debug(
                    "Negation detected via aux: neg(%s, %s)",
                    child.text, grandchild_neg[0].text,
                )
                return True

    # 模式 3：谓词附近的词项否定。
    # 检查谓词附近词元中的否定词。
    if _has_lexical_negation(self, tree, predicate_idx):
        logger.debug(
            "Lexical negation detected near predicate index %d",
            predicate_idx,
        )
        return True

    return False


def _has_lexical_negation(
    self,
    tree: DependencyTree,
    predicate_idx: int,
    window: int = 5,
) -> bool:
    """检查谓词附近是否存在词项否定词元。

    扫描谓词两侧 ``window`` 个位置内的词元：
      - "no"（限定词 —— "no party shall"、"no assignment may"）
      - "neither"（关联 —— "neither party shall"）
      - "nor"（关联 —— "...nor shall any party"）
      - "never"（副词 —— "shall never assign"）

    这些否定词与 "not" 具有相同法律效果，
    但在部分解析风格中可能不通过 neg 依存关系附着。

    参数：
        tree: 依存树。
        predicate_idx: 谓词的 1 基索引。
        window: 两侧扫描的词元数。

    返回：
        窗口内发现词项否定词元时返回 True。
    """
    negation_words = {"no", "neither", "nor", "never", "nothing"}

    start = max(1, predicate_idx - window)
    end = min(tree.token_count, predicate_idx + window)

    for i in range(start, end + 1):
        token = tree.get_token(i)
        if token is not None:
            text_lower = token.text.lower().strip()
            if text_lower in negation_words:
                # "no" 仅在前置于主语（名词短语范围）时计为否定。
                # "whether or no" 中的 "no" 不计。
                if text_lower == "no" and token.upos != "DET":
                    # 非限定词的 "no" 不太可能是否定谓词。
                    continue
                return True

    return False
