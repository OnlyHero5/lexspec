"""
Depth metrics computation — additional linguistic complexity metrics.

Used by the evaluation module for phenomenon-based error analysis.
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
    """Compute additional linguistic depth metrics for analysis.

    These metrics quantify sentence complexity and help explain
    WHY certain sentences cause LLM extraction errors. Used by
    the evaluation module for phenomenon-based error analysis.

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate.

    Returns:
        Dict with keys:
          - mean_dependency_distance (float): MDD for the sentence.
          - predicate_to_subject_distance (int): Token distance from
            predicate to its subject.
          - predicate_to_object_distance (int): Token distance from
            predicate to its object.
          - has_long_distance (bool): Any dependency > 5 tokens.
          - has_acl_relcl (bool): Relative clause present.
          - dependency_depth (int): Max depth from root to leaf.
    """
    metrics: dict = {}

    # Mean Dependency Distance.
    metrics["mean_dependency_distance"] = compute_mean_dependency_distance(tree)

    # Predicate-to-argument distances.
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

    # Long-distance dependency check (any dependency > 5).
    long_dist = UDFeatureExtractor.find_long_distance_dependencies(
        tree, threshold=5
    )
    metrics["has_long_distance"] = len(long_dist) > 0

    # Relative clause detection (on any noun in the sentence).
    has_relcl = any(
        UDFeatureExtractor.has_acl_relcl(tree, t.index)
        for t in tree.tokens
    )
    metrics["has_acl_relcl"] = has_relcl

    # Maximum dependency depth (longest path from any token to root).
    max_depth = 0
    for token in tree.tokens:
        path = tree.get_path_to_root(token.index)
        if len(path) > max_depth:
            max_depth = len(path)
    metrics["dependency_depth"] = max_depth

    return metrics
