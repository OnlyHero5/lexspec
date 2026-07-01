"""
Dependency Path Legality Rate metric.

Measures whether extracted (subject, predicate) and (predicate, object)
pairs have legal dependency paths in the UD tree. A "legal" path exists
if there is a directed dependency path between the two tokens.
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
    """Compute the rate at which extracted (subject,predicate) and
    (predicate,object) pairs have legal dependency paths in the UD tree.

    A path is "legal" if there exists a directed dependency path between
    the two tokens in the UD tree (either token is an ancestor of the other
    in the dependency graph). This measures whether the extraction is
    syntactically plausible — syntactically impossible extractions are
    definite errors.

    For each prediction:
      1. Locate the subject, predicate, and object tokens in the tree.
         Token location uses lemma matching on the normalized text.
      2. Check if a dependency path exists between predicate↔subject
         and predicate↔object.
      3. Count the number of legal paths across all predictions.
      4. Return the ratio of legal paths to total expected paths.

    A prediction with an empty condition (or missing object, etc.) still
    has its subject-path and object-path checked if those fields are non-empty.

    Args:
        predictions: List of predicted LegalTriplets, aligned 1:1 with trees.
        trees: List of DependencyTree objects from UD parsing.
               Must have the same length as predictions.

    Returns:
        Legality rate as a float in [0, 1]. 1.0 means all extracted pairs
        have legal dependency paths; 0.0 means none do.

    Raises:
        ValueError: If predictions and trees have different lengths.
    """
    if len(predictions) != len(trees):
        raise ValueError(
            f"predictions and trees must have the same length. "
            f"Got {len(predictions)} and {len(trees)}."
        )

    total_pairs = 0
    legal_pairs = 0

    for pred, tree in zip(predictions, trees):
        # Skip empty trees (no tokens) — cannot verify any paths.
        if tree.token_count == 0:
            continue

        # Find the predicate token in the tree using lemma matching.
        pred_token = find_token_in_tree(tree, pred.action.predicate)

        if pred_token is not None:
            # Check subject → predicate path.
            if pred.subject.text.strip():
                subj_token = find_token_in_tree(tree, pred.subject.text)
                if subj_token is not None:
                    total_pairs += 1
                    if has_directed_path(tree, subj_token.index, pred_token.index):
                        legal_pairs += 1

            # Check predicate → object path.
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
    """Find a token in the dependency tree by matching its lemma or text.

    Uses normalized text comparison for fuzzy matching. Searches first by
    exact lemma match, then by lemma containment (for multi-word spans),
    then by text match as a fallback.

    Args:
        tree: The dependency tree to search.
        text: The extraction text (e.g., subject, predicate, object).

    Returns:
        The best-matching Token, or None if no match is found.
    """
    if not text or not text.strip():
        return None

    # Normalize the search text for matching.
    search_text = normalize(text, remove_articles=True, number_normalize=False)
    if not search_text:
        return None

    search_tokens = set(search_text.split())

    # Content-word UPOS tags preferred on tie-breaks (predicates are VERBs,
    # subjects/objects are NOUNs). Function words (AUX, ADP, DET) are
    # secondary — they rarely carry the core semantic content.
    _CONTENT_UPOS_PRIORITY: Dict[str, int] = {
        "VERB": 3, "NOUN": 3, "PROPN": 3, "ADJ": 2, "ADV": 2,
        "PRON": 1, "AUX": 0, "ADP": 0, "DET": 0, "CCONJ": 0,
        "SCONJ": 0, "PART": 0, "NUM": 2, "X": 0,
    }

    best_token: Optional[Token] = None
    best_score = 0
    best_priority = -1  # Tie-breaker: higher priority wins at same overlap.

    for token in tree.tokens:
        token_lemma_norm = normalize(token.lemma, remove_articles=True, number_normalize=False)
        token_text_norm = normalize(token.text, remove_articles=True, number_normalize=False)

        # Determine content-word priority for this token.
        priority = _CONTENT_UPOS_PRIORITY.get(token.upos, 0)

        # Score: how many search tokens match this tree token's form, by lemma.
        token_set = set(token_lemma_norm.split())
        overlap = len(search_tokens & token_set)
        # Accept if strictly better overlap, OR same overlap but higher priority (tie-break).
        if overlap > best_score or (overlap == best_score and overlap > 0 and priority > best_priority):
            best_score = overlap
            best_priority = priority
            best_token = token

        # Also try text form (useful for named entities like "Seller").
        token_set = set(token_text_norm.split())
        overlap = len(search_tokens & token_set)
        if overlap > best_score or (overlap == best_score and overlap > 0 and priority > best_priority):
            best_score = overlap
            best_priority = priority
            best_token = token

    return best_token if best_score > 0 else None


def has_directed_path(tree: DependencyTree, from_idx: int, to_idx: int) -> bool:
    """Check if there is a directed dependency path between two tokens.

    A directed path exists if one token is an ancestor of the other in the
    dependency graph. Since UD trees are rooted directed graphs, we check
    whether walking from from_idx up to root passes through to_idx, or
    whether walking from to_idx up to root passes through from_idx.

    Args:
        tree: The dependency tree.
        from_idx: 1-based index of the source token.
        to_idx: 1-based index of the target token.

    Returns:
        True if a directed path connects the two tokens in either direction.
    """
    # Walk from `from_idx` up to root; check if we pass through `to_idx`.
    current = tree.get_token(from_idx)
    while current is not None and current.head != 0:
        if current.index == to_idx:
            return True
        current = tree.get_token(current.head)
    # Also check the root token itself.
    if current is not None and current.index == to_idx:
        return True

    # Walk from `to_idx` up to root; check if we pass through `from_idx`.
    current = tree.get_token(to_idx)
    while current is not None and current.head != 0:
        if current.index == from_idx:
            return True
        current = tree.get_token(current.head)
    if current is not None and current.index == from_idx:
        return True

    return False
