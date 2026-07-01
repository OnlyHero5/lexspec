"""
Condition span overlap calculation and main-clause-overlap detection.
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
    """Compute the token-level overlap between LLM condition text
    and a UD-derived condition span.

    Uses Jaccard similarity (intersection over union) on token sets
    after normalization. This is the primary metric for evaluating
    condition boundary accuracy (used by validator Step 5).

    IoU = |tokens(LLM) ∩ tokens(UD)| / |tokens(LLM) ∪ tokens(UD)|

    Higher IoU means more accurate boundary extraction:
      IoU = 1.0  -> Perfect match (unlikely with different tokenizers)
      IoU >= 0.8 -> Excellent match (acceptable)
      IoU >= 0.5 -> Moderate match (check for boundary issues)
      IoU < 0.5  -> Poor match (likely boundary error)

    Args:
        llm_condition_text: The condition text from the LLM extraction.
        ud_condition_span: The UD-derived ConditionSpan.
        tree: Dependency tree (for tokenization reference).

    Returns:
        Jaccard similarity score between 0.0 and 1.0.
    """
    # Tokenize both texts into sets of lowercase word tokens.
    # We use simple whitespace tokenization for the LLM output
    # because it may not match the Stanza tokenization exactly.
    llm_tokens = set(llm_condition_text.lower().split())

    # For UD tokens, use the actual Stanza token texts from the span.
    ud_tokens = set()
    for token_idx in ud_condition_span.tokens:
        token = tree.get_token(token_idx)
        if token is not None:
            ud_tokens.add(token.text.lower())

    if not llm_tokens and not ud_tokens:
        # Both empty — no condition in either. This is a match.
        return 1.0
    if not llm_tokens or not ud_tokens:
        # One has a condition, the other doesn't — complete mismatch.
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
    """Heuristic check: does the condition span overlap with the
    main clause subject or predicate region?

    If a condition span includes the main predicate or its subject,
    the extractor has likely drawn the boundary incorrectly (over-extraction).

    Args:
        condition_span: The extracted condition span.
        tree: Dependency tree.

    Returns:
        True if the condition appears to include main clause elements
        (suggesting boundary error).
    """
    root_idx = tree.root_index
    if root_idx is None:
        return False

    # Get the main predicate region (predicate + its immediate args).
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
