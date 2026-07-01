"""
Clause relations: find_advcl_with_mark, _matches_marker, _extract_condition_span_text.
"""

from __future__ import annotations

from typing import List

from src.extraction.schema import DependencyTree, Token, ConditionSpan, ConditionType
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _matches_marker(mark_text: str, markers: List[str]) -> bool:
    """Check if a mark word text matches any known condition marker.

    Handles multi-word markers by checking if the mark text is
    the first word of a multi-word phrase (e.g., "provided" in
    "provided that") or an exact match for a single-word marker.

    Args:
        mark_text: Lowercase mark word text.
        markers: List of marker strings from constraints.yaml.

    Returns:
        True if mark_text matches any marker.
    """
    for marker in markers:
        if marker == mark_text:
            return True
        # Multi-word marker check: "provided" matches "provided that"
        if " " in marker and marker.startswith(mark_text + " "):
            return True
        # Also check: mark_text might be the full multi-word phrase
        # (if Stanza treated it as a single token). This is rare but
        # can happen with fixed expressions.
        if mark_text.startswith(marker + " ") or mark_text == marker:
            return True
    return False


def _extract_condition_span_text(
    tree: DependencyTree,
    advcl_head_idx: int,
    mark_token: Token,
) -> str:
    """Extract the full text of a condition clause.

    The span starts at the mark word and includes the entire advcl
    subtree. This ensures we capture complete conditions like:
      "if Buyer fails to pay any installment when due"
    not just:
      "if Buyer fails"

    Strategy: Collect all tokens in the advcl subtree, then also
    walk LEFT from the advcl head to find the mark word and any
    tokens between mark and advcl head. Sort by index to reconstruct
    surface order.

    Args:
        tree: Dependency tree.
        advcl_head_idx: Index of the advcl head (the clause's main verb).
        mark_token: The mark word token.

    Returns:
        Full condition clause text in surface order.
    """
    # Collect all token indices in the advcl subtree.
    subtree_indices = set(tree._collect_subtree(advcl_head_idx))

    # The mark word may be outside the advcl subtree in some parses
    # (UD allows mark to be attached as a dependent of the head).
    # Ensure it's included.
    subtree_indices.add(mark_token.index)

    # Also include any tokens between the mark and the advcl head
    # that may not be direct descendants (e.g., intervening adverbs).
    min_idx = min(mark_token.index, advcl_head_idx)
    max_idx = max(mark_token.index, advcl_head_idx)
    for i in range(min_idx, max_idx + 1):
        subtree_indices.add(i)

    # Sort by index to reconstruct surface order.
    sorted_indices = sorted(subtree_indices)
    tokens_sorted = [
        tree.get_token(i) for i in sorted_indices
    ]
    tokens_sorted = [t for t in tokens_sorted if t is not None]

    return " ".join(t.text for t in tokens_sorted)


def find_advcl_with_mark(
    tree: DependencyTree,
    predicate_idx: int,
    condition_markers: List[str],
) -> List[ConditionSpan]:
    """Find adverbial clauses with condition-marking subordinators.

    UD: advcl(predicate, clause_head) — an adverbial clause modifier.
        mark(clause_head, marker)   — the subordinating conjunction.

    In legal text, advcl+mark identifies condition clauses:
      "IF BUYER FAILS TO PAY, Seller may terminate"
        -> advcl(terminate, fails)
        -> mark(fails, If)

    The mark word determines the condition type:
      - "if", "provided that" -> TRIGGER (event-based condition)
      - "when", "upon", "after" -> TEMPORAL (time-based condition)
      - "unless", "except" -> EXCEPTION (scope limitation)

    This function does NOT classify conditions — it only extracts
    the spans. Classification is handled by ConditionExtractor
    using the constraints.yaml taxonomy.

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the main predicate.
        condition_markers: List of lowercase mark words that signal
                           conditions (from constraints.yaml).

    Returns:
        List of ConditionSpan objects with text, indices, and mark info.
        Empty list if no advcl with condition-marker is found.
    """
    result: List[ConditionSpan] = []

    # Step 1: Find all advcl children of the predicate.
    # An advcl is an adverbial clause modifying the main verb.
    # In legal text, the most common advcl type is conditional.
    advcl_children = tree.get_children(predicate_idx, deprel="advcl")

    if not advcl_children:
        # No adverbial clauses at all — no conditions present.
        return result

    logger.debug(
        "Found %d advcl children of predicate at index %d",
        len(advcl_children), predicate_idx,
    )

    # Step 2: For each advcl child, check for a mark (subordinator).
    for advcl_head in advcl_children:
        # The mark word is a dependent of the advcl head verb.
        # E.g., "If Buyer fails to pay" -> mark(fails, If)
        mark_children = tree.get_children(advcl_head.index, deprel="mark")

        if not mark_children:
            # advcl without an explicit mark — could be an infinitival
            # or participial clause. These are less common in legal
            # English and typically don't express conditions.
            # Example: "Seller agrees [to deliver the Goods]"
            #   -> advcl(agrees, deliver), no mark word.
            continue

        mark_token = mark_children[0]

        # Step 3: Check if the mark word is a recognized condition marker.
        # Normalize by lowercase — legal text uses mixed capitalization.
        mark_text_lower = mark_token.text.lower().strip()

        # Multi-word markers like "provided that", "in the event that"
        # may have the first word as the mark token. We check against
        # both single-word and first-word-of-multi-word patterns.
        if not _matches_marker(
            mark_text_lower, condition_markers
        ):
            # Not a condition marker — could be temporal ("when"),
            # causal ("because"), or purpose ("so that").
            # Skip these — they're not conditions in the legal sense.
            logger.debug(
                "Skipping advcl at index %d: mark '%s' not a condition marker",
                advcl_head.index, mark_token.text,
            )
            continue

        # Step 4: Extract the full condition span text.
        # We include the mark word and the entire advcl subtree
        # to capture the complete condition boundary.
        # For "if Buyer fails to pay any installment when due",
        # we want ALL of that text, not just "if Buyer fails".
        span_text = _extract_condition_span_text(
            tree, advcl_head.index, mark_token
        )

        # Build a ConditionSpan (condition_type set to NONE initially —
        # the ConditionExtractor will classify it).
        condition_span = ConditionSpan(
            tokens=sorted(tree._collect_subtree(advcl_head.index)),
            text=span_text,
            deprel="advcl",
            mark_token=mark_token,
            condition_type=ConditionType.NONE,  # Classified later
            mark_text=mark_token.text,
        )
        result.append(condition_span)

        logger.debug(
            "Extracted condition span: '%s' (mark: '%s')",
            span_text[:80], mark_token.text,
        )

    return result
