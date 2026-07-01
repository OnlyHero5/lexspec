"""
依存路径合法率指标。

衡量抽取的（主语, 谓词）与（谓词, 宾语）对在 UD 树中是否具有
合法依存路径。「合法」指两词元间存在有向依存路径。
"""

from __future__ import annotations

from typing import Optional, List, Dict

from src.extraction.schema import (
    LegalTriplet, DependencyTree, Token,
)
from src.evaluation.text_normalizer import normalize
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_dependency_path_legality(
    predictions: List[LegalTriplet],
    trees: List[DependencyTree],
) -> float:
    """计算抽取的（主语,谓词）与（谓词,宾语）对在 UD 树中
    具有合法依存路径的比例。

    路径「合法」指 UD 树中存在两词元间的有向依存路径
    （一为另一在依存图中的祖先）。衡量抽取是否在句法上合理 —
    句法上不可能的抽取必为错误。

    对每个预测：
      1. 在树中定位主语、谓词、宾语词元。
         定位使用归一化文本上的词形匹配。
      2. 检查谓词↔主语、谓词↔宾语间是否存在依存路径。
      3. 统计全部预测中合法路径数。
      4. 返回合法路径与期望路径总数之比。

    条件为空（或缺宾语等）的预测，若对应字段非空仍检查主语路径与宾语路径。

    参数:
        predictions: 预测 LegalTriplet 列表，与 trees 1:1 对齐。
        trees: UD 解析得到的 DependencyTree 列表。
               须与 predictions 等长。

    返回:
        [0, 1] 浮点合法率。1.0 表示全部抽取对均有合法路径；0.0 表示均无。

    异常:
        ValueError: predictions 与 trees 长度不同。
    """
    if len(predictions) != len(trees):
        raise ValueError(
            f"predictions and trees must have the same length. "
            f"Got {len(predictions)} and {len(trees)}."
        )

    total_pairs = 0
    legal_pairs = 0

    for pred, tree in zip(predictions, trees):
        # 跳过空树（无词元）— 无法验证路径。
        if tree.token_count == 0:
            continue

        # 用词形匹配在树中找谓词词元。
        pred_token = find_token_in_tree(tree, pred.action.predicate)

        if pred_token is not None:
            # 检查主语 → 谓词路径。
            if pred.subject.text.strip():
                subj_token = find_token_in_tree(tree, pred.subject.text)
                if subj_token is not None:
                    total_pairs += 1
                    if has_directed_path(tree, subj_token.index, pred_token.index):
                        legal_pairs += 1

            # 检查谓词 → 宾语路径。
            if pred.action.object.strip():
                obj_token = find_token_in_tree(tree, pred.action.object)
                if obj_token is not None:
                    total_pairs += 1
                    if has_directed_path(tree, pred_token.index, obj_token.index):
                        legal_pairs += 1

    legality_rate = legal_pairs / total_pairs if total_pairs > 0 else 0.0

    logger.info(
        "Dependency path legality: %d/%d legal = %.4f",
        legal_pairs, total_pairs, legality_rate,
    )
    return legality_rate


def find_token_in_tree(tree: DependencyTree, text: str) -> Optional[Token]:
    """通过词形或文本匹配在依存树中查找词元。

    使用归一化文本比较做模糊匹配。先精确词形匹配，
    再词形包含（多词片段），最后文本匹配兜底。

    参数:
        tree: 待搜索的依存树。
        text: 抽取文本（如主语、谓词、宾语）。

    返回:
        最佳匹配 Token，未找到则 None。
    """
    if not text or not text.strip():
        return None

    # 归一化搜索文本以便匹配。
    search_text = normalize(text, remove_articles=True, number_normalize=False)
    if not search_text:
        return None

    search_tokens = set(search_text.split())

    # 实词 UPOS 在平局时优先（谓词为 VERB，主语/宾语为 NOUN）。
    # 功能词（AUX、ADP、DET）次之 — 很少承载核心语义。
    _CONTENT_UPOS_PRIORITY: Dict[str, int] = {
        "VERB": 3, "NOUN": 3, "PROPN": 3, "ADJ": 2, "ADV": 2,
        "PRON": 1, "AUX": 0, "ADP": 0, "DET": 0, "CCONJ": 0,
        "SCONJ": 0, "PART": 0, "NUM": 2, "X": 0,
    }

    best_token: Optional[Token] = None
    best_score = 0
    best_priority = -1  # 平局决胜：同重叠时优先级高者胜。

    for token in tree.tokens:
        token_lemma_norm = normalize(token.lemma, remove_articles=True, number_normalize=False)
        token_text_norm = normalize(token.text, remove_articles=True, number_normalize=False)

        # 确定该词元的实词优先级。
        priority = _CONTENT_UPOS_PRIORITY.get(token.upos, 0)

        # 得分：搜索词元中有多少与该树词元形式（词形）匹配。
        token_set = set(token_lemma_norm.split())
        overlap = len(search_tokens & token_set)
        # 接受：重叠严格更大，或同重叠但优先级更高（平局）。
        if overlap > best_score or (overlap == best_score and overlap > 0 and priority > best_priority):
            best_score = overlap
            best_priority = priority
            best_token = token

        # 也尝试文本形式（专名如 "Seller" 有用）。
        token_set = set(token_text_norm.split())
        overlap = len(search_tokens & token_set)
        if overlap > best_score or (overlap == best_score and overlap > 0 and priority > best_priority):
            best_score = overlap
            best_priority = priority
            best_token = token

    return best_token if best_score > 0 else None


def has_directed_path(tree: DependencyTree, from_idx: int, to_idx: int) -> bool:
    """检查两词元间是否存在有向依存路径。

    路径存在当且仅当一为另一在依存图中的祖先。
    UD 树为有根有向图，从 from_idx 向上到根是否经过 to_idx，
    或从 to_idx 向上是否经过 from_idx。

    参数:
        tree: 依存树。
        from_idx: 源词元 1-based 索引。
        to_idx: 目标词元 1-based 索引。

    返回:
        两词元任方向连通时为 True。
    """
    # 从 from_idx 向上到根；检查是否经过 to_idx。
    current = tree.get_token(from_idx)
    while current is not None and current.head != 0:
        if current.index == to_idx:
            return True
        current = tree.get_token(current.head)
    # 也检查根词元本身。
    if current is not None and current.index == to_idx:
        return True

    # 从 to_idx 向上到根；检查是否经过 from_idx。
    current = tree.get_token(to_idx)
    while current is not None and current.head != 0:
        if current.index == from_idx:
            return True
        current = tree.get_token(current.head)
    if current is not None and current.index == from_idx:
        return True

    return False
