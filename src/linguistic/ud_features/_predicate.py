"""
谓词识别：find_root_predicate 与 find_all_predicates。
"""

from __future__ import annotations

from typing import Optional, List

from src.extraction.schema import DependencyTree, Token
from src.utils.logging import get_logger

logger = get_logger(__name__)


def find_root_predicate(tree: DependencyTree) -> Optional[Token]:
    """定位句子的根谓词（主句动词）。

    UD 中根词元为 head=0 者。格式良好的句子中，
    通常为 main verb。对法律从句，标识核心法律动作
    （如 "deliver"、"terminate"、"indemnify"）。

    UD 依据：root 关系 —— head=0 标记句法根。

    处理的边界情况：
      - 多根（畸形解析）：返回第一个 VERB 根。
        若无 VERB 根，返回第一个根词元。
      - 无根（空树）：返回 None。
      - 根非 VERB（如系词 "is" 带名词谓语）：
        检查 xcomp 补语是否承载真实动作。

    参数：
        tree: 已解析的依存树。

    返回：
        根 Token，未找到根时返回 None。

    法律文本示例：
        "Seller shall deliver the Goods within 30 days."
        -> root = Token("deliver", upos="VERB", head=0)
    """
    root_idx = tree.root_index
    if root_idx is None:
        logger.debug("No root token found in tree (empty or malformed)")
        return None

    root_token = tree.get_token(root_idx)
    if root_token is None:
        return None

    # 法律文本中根几乎总是 VERB（主要动作）。
    # 若非动词，可能为系词构造（如
    # "The agreement IS binding"），内容谓词在别处。
    # 尝试通过 xcomp 或 ccomp 找到真实谓词。
    if root_token.upos != "VERB" and root_token.upos != "AUX":
        logger.debug(
            "Root token '%s' is %s, not VERB. Searching for xcomp/ccomp.",
            root_token.text, root_token.upos,
        )
        # 查找承载语义谓词的开放从句补语（xcomp）。
        # 例："Seller IS required to deliver"
        # -> root=AUX("is")，xcomp(required, deliver) 指向动作。
        for child in tree.get_children(root_idx):
            if child.deprel in ("xcomp", "ccomp") and child.upos == "VERB":
                logger.debug(
                    "Found predicate via %s: %s (index %d)",
                    child.deprel, child.lemma, child.index,
                )
                return child

    return root_token


def find_all_predicates(tree: DependencyTree) -> List[Token]:
    """查找所有可作谓词的动词词元。

    扫描全部 VERB 词元并按与根的距离排序。
    适用于主谓词非句法根的多从句句子。

    参数：
        tree: 依存树。

    返回：
        VERB Token 列表，根动词优先。
    """
    verbs = tree.find_tokens_by_upos("VERB")
    if not verbs:
        return []

    root_idx = tree.root_index

    # 按（到根距离升序，索引升序）排序
    def _distance_to_root(token: Token) -> int:
        if root_idx is None:
            return 0
        path = tree.get_path_to_root(token.index)
        return len(path)

    verbs.sort(key=lambda t: (_distance_to_root(t), t.index))
    return verbs
