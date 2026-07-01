"""
依存拓扑：路径、距离、并列。
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Dict

from src.extraction.schema import DependencyTree, Token
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_dependency_path(
    tree: DependencyTree,
    from_idx: int,
    to_idx: int,
) -> Optional[List[Token]]:
    """查找两词元之间的依存路径。

    使用双向 BFS 沿依存树向上。两词元经 head 指针
    向上行走直至路径相交。交点为最近公共祖先（LCA）。

    用于语言学度量中的依存路径合法性检查：
    若大语言模型从与谓词不同树分支的词元抽取主语，
    则抽取在句法上可疑。

    参数：
        tree: 依存树。
        from_idx: 起始词元索引（1 基）。
        to_idx: 结束词元索引（1 基）。

    返回：
        从 from_idx 到 to_idx 路径上的 Token 列表
        （含两端点），无路径时返回 None
        （不连通树或缺失词元）。
    """
    from_token = tree.get_token(from_idx)
    to_token = tree.get_token(to_idx)
    if from_token is None or to_token is None:
        return None

    # 从 'from' 词元向上构建至根的路径。
    # 记录已访问索引及其路径供 LCA 检测。
    path_from: Dict[int, List[Token]] = {}
    current = from_token
    from_path: List[Token] = []
    while current is not None:
        from_path.append(current)
        path_from[current.index] = list(from_path)
        if current.head == 0:
            break
        current = tree.get_token(current.head)

    # 从 'to' 词元向上构建至根的路径，
    # 每步检查是否与 'from' 路径相交。
    current = to_token
    to_path: List[Token] = []
    while current is not None:
        to_path.append(current)
        # 检查是否与 'from' 路径相交。
        if current.index in path_from:
            # 找到 LCA。合并路径：
            # from_path 至（含）LCA，
            # 再 to_path 反转（排除 LCA 避免重复）。
            from_path_to_lca = path_from[current.index]
            # to_path_to_lca 为当前至 LCA 的 to_path，
            # 反转使顺序为 LCA -> from，再跳过首项（LCA）。
            to_path_to_lca = list(reversed(to_path))
            combined = from_path_to_lca + to_path_to_lca[1:]
            return combined
        if current.head == 0:
            break
        current = tree.get_token(current.head)

    # 无交点 —— 词元处于不连通成分。
    return None


def compute_mean_dependency_distance(tree: DependencyTree) -> float:
    """计算句子的平均依存距离（MDD）。

    MDD = mean( |head_index - dependent_index| )，对所有非根词元。
    MDD 越高表示句法结构越复杂、
    长距离句法关系越多。

    本指标用于：
      1. 长距离依存现象分类（设计文档 §5.2）。
      2. 估计句子复杂度以平衡测试集采样。
      3. 识别易导致大语言模型抽取错误的句子。

    参数：
        tree: 依存树。

    返回：
        平均依存距离（浮点数）。
        词元少于 2 个时返回 0.0（无依存）。
    """
    if tree.token_count < 2:
        return 0.0

    total_distance = 0
    non_root_count = 0

    for token in tree.tokens:
        if token.head == 0:
            continue  # 根无依存距离
        distance = abs(token.index - token.head)
        total_distance += distance
        non_root_count += 1

    if non_root_count == 0:
        return 0.0

    return total_distance / non_root_count


def find_long_distance_dependencies(
    tree: DependencyTree,
    threshold: int = 5,
) -> List[Tuple[Token, Token, int]]:
    """查找线性距离超过阈值的依存对。

    长距离依存（>5 个介入词元）是已知的大语言模型
    抽取错误来源，因模型须在单句内大上下文窗口
    中跟踪关系。

    法律文本中长距离依存常见，原因包括：
      - 介入状语短语："Seller shall, within 30 days
        after the Closing Date and subject to the conditions set forth
        in Section 2.3, deliver the Goods."
      - 定义宾语的嵌入式关系从句。

    参数：
        tree: 依存树。
        threshold: 标记为长距离的最小词元距离。

    返回：
        超过阈值的 (dependent, head, distance) 元组列表，
        按距离降序排列。
    """
    result: List[Tuple[Token, Token, int]] = []

    for token in tree.tokens:
        if token.head == 0:
            continue
        distance = abs(token.index - token.head)
        if distance > threshold:
            head_token = tree.get_token(token.head)
            if head_token is not None:
                result.append((token, head_token, distance))

    result.sort(key=lambda x: x[2], reverse=True)
    return result


def get_conjuncts(tree: DependencyTree, token_idx: int) -> List[Token]:
    """获取词元的全部并列项（含词元自身）。

    UD: conj(head, conjunct) —— 并列关系。
    法律文本中：
      "Buyer AND Seller shall deliver"
        -> nsubj(deliver, Buyer)
        -> conj(Buyer, Seller)  -- "Seller" 为 "Buyer" 的并列项

    将单一 nsubj/obj 词元扩展为完整并列短语，
    使 "Buyer and Seller" 作为复合主语而非仅 "Buyer"。

    参数：
        tree: 依存树。
        token_idx: 中心并列项词元的索引。

    返回：
        全部并列词元列表（中心项优先，再按顺序 conj 依存），
        含词元自身。
    """
    result = [tree.get_token(token_idx)]
    if result[0] is None:
        return []

    # 收集全部直接 conj 依存。
    for child in tree.get_children(token_idx, deprel="conj"):
        result.append(child)

    return [t for t in result if t is not None]


def get_conjunct_text(tree: DependencyTree, token_idx: int) -> str:
    """获取并列的完整文本跨度，含连词。

    对 "Buyer and Seller"，返回覆盖全部并列项
    及其间连词（"and"）的文本。

    参数：
        tree: 依存树。
        token_idx: 中心并列项的索引。

    返回：
        完整并列名词短语的重建文本。
    """
    conjuncts = get_conjuncts(tree, token_idx)
    if len(conjuncts) <= 1:
        # 单一词元，无并列。
        tok = tree.get_token(token_idx)
        return tok.text if tok else ""

    # 对并列短语，提取第一与最后并列项之间
    #（含）的全部词元。捕获并列项间的 and/or/cc 词元。
    indices = [c.index for c in conjuncts]
    min_idx = min(indices)
    max_idx = max(indices)

    span_tokens = [
        tree.get_token(i) for i in range(min_idx, max_idx + 1)
    ]
    span_tokens = [t for t in span_tokens if t is not None]
    return " ".join(t.text for t in span_tokens)
