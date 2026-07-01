"""
深度度量计算 —— 附加语言学复杂度度量。

供评估模块进行基于现象的错误分析。
"""

from __future__ import annotations

from src.extraction.schema import DependencyTree
from src.linguistic.ud_features import (
    UDFeatureExtractor,
    find_nsubj,
    find_obj,
    find_nsubj_pass,
    compute_mean_dependency_distance,
)


def compute_depth_metrics(
    tree: DependencyTree,
    predicate_idx: int,
) -> dict:
    """计算附加语言学深度度量供分析使用。

    这些度量量化句子复杂度，帮助解释
    某些句子导致大语言模型抽取错误的原因。
    供评估模块进行基于现象的错误分析。

    参数：
        tree: 依存树。
        predicate_idx: 谓词的 1 基索引。

    返回：
        含以下键的字典：
          - mean_dependency_distance (float): 句子的 MDD。
          - predicate_to_subject_distance (int): 谓词到主语的距离。
          - predicate_to_object_distance (int): 谓词到宾语的距离。
          - has_long_distance (bool): 是否存在 >5 词元的依存。
          - has_acl_relcl (bool): 是否存在关系从句。
          - dependency_depth (int): 从根到叶的最大深度。
    """
    metrics: dict = {}

    # 平均依存距离。
    metrics["mean_dependency_distance"] = compute_mean_dependency_distance(tree)

    # 谓词到论元的距离。
    nsubj = find_nsubj(tree, predicate_idx) or find_nsubj_pass(tree, predicate_idx)
    if nsubj:
        metrics["predicate_to_subject_distance"] = abs(
            predicate_idx - nsubj.index
        )
    else:
        metrics["predicate_to_subject_distance"] = -1

    obj = find_obj(tree, predicate_idx)
    if obj:
        metrics["predicate_to_object_distance"] = abs(
            predicate_idx - obj.index
        )
    else:
        metrics["predicate_to_object_distance"] = -1

    # 长距离依存检查（任一依存 > 5）。
    long_dist = UDFeatureExtractor.find_long_distance_dependencies(
        tree, threshold=5
    )
    metrics["has_long_distance"] = len(long_dist) > 0

    # 关系从句检测（句中任一名词）。
    has_relcl = any(
        UDFeatureExtractor.has_acl_relcl(tree, t.index)
        for t in tree.tokens
    )
    metrics["has_acl_relcl"] = has_relcl

    # 最大依存深度（任一词元到根的最长路径）。
    max_depth = 0
    for token in tree.tokens:
        path = tree.get_path_to_root(token.index)
        if len(path) > max_depth:
            max_depth = len(path)
    metrics["dependency_depth"] = max_depth

    return metrics
