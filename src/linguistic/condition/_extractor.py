"""
Primary extraction interface for ConditionExtractor.
"""

from __future__ import annotations

from typing import Optional, List

from src.extraction.schema import (
    DependencyTree,
    ConditionSpan,
    ConditionType,
)
from src.linguistic.ud_features import find_advcl_with_mark
from src.utils.logging import get_logger

logger = get_logger(__name__)


def extract(
    self,
    tree: DependencyTree,
    predicate_idx: int,
) -> Optional[ConditionSpan]:
    """Extract the primary condition clause for a predicate.

    If multiple condition clauses exist, returns the first one
    (typically the most syntactically prominent — the one closest
    to the main predicate in surface order).

    For legal contracts, having >1 condition clause per main predicate
    is uncommon but possible (e.g., "If X and upon Y, Seller shall Z").
    Use extract_all() to get all conditions.

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the main predicate.

    Returns:
        ConditionSpan with text, type, and mark info, or None if
        no condition clause is detected.

    Raises:
        ValueError: If predicate_idx is not found in the tree.
    """
    spans = extract_all(self, tree, predicate_idx)
    if spans:
        # Return the first (most prominent) condition.
        return spans[0]
    return None


def extract_all(
    self,
    tree: DependencyTree,
    predicate_idx: int,
) -> List[ConditionSpan]:
    """Extract all condition clauses (including multiple conditions).

    When multiple conditions exist on the same predicate (e.g.,
    "If the Buyer defaults, and unless waived by the Seller, the
    Seller may terminate"), both conditions are returned.

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the main predicate.

    Returns:
        List of all ConditionSpan objects found. Empty list if
        no conditions are detected.

    Raises:
        ValueError: If predicate_idx is not found in the tree.
    """
    predicate = tree.get_token(predicate_idx)
    if predicate is None:
        raise ValueError(
            f"Predicate index {predicate_idx} not found in tree "
            f"(token count: {tree.token_count})"
        )

    # Step 1: Get raw advcl+mark spans from the UD tree.
    # These spans are UNCLASSIFIED — condition_type is NONE.
    raw_spans = find_advcl_with_mark(
        tree, predicate_idx, self._marker_list
    )

    if not raw_spans:
        logger.debug(
            "No advcl+mark condition clauses found for predicate '%s' at index %d",
            predicate.lemma, predicate_idx,
        )
        return []

    # Step 2: Classify each span based on its mark word.
    classified_spans: List[ConditionSpan] = []
    for span in raw_spans:
        if span.mark_token is None:
            # Should not happen — find_advcl_with_mark only returns
            # spans with mark tokens. Defensive check.
            logger.debug("Skipping span without mark token")
            continue

        mark_text = span.mark_text.lower().strip()
        condition_type = _classify_condition(self, mark_text)

        # Create a classified ConditionSpan by copying fields from the
        # raw span and setting the condition_type.
        classified = ConditionSpan(
            tokens=span.tokens,
            text=span.text,
            deprel=span.deprel,
            mark_token=span.mark_token,
            condition_type=condition_type,
            mark_text=span.mark_text,
        )
        classified_spans.append(classified)

        logger.debug(
            "Classified condition: type=%s, mark='%s', span='%s...'",
            condition_type.value, mark_text, span.text[:80],
        )

    return classified_spans


def _classify_condition(self, mark_text: str) -> ConditionType:
    """Classify a condition based on its mark word.

    Uses the legal-domain taxonomy loaded from constraints.yaml:

    TRIGGER (event-based):
      "if" — prototypical conditional: "IF Buyer defaults, Seller may..."
      "provided that" — qualified condition: "provided that the
        Company receives notice..."
      "in the event that" — contingency: "in the event that any
        representation proves false..."
      "so long as" — durative condition: "so long as no Event of
        Default has occurred..."

    TEMPORAL (time-based):
      "when" — temporal trigger: "WHEN the Closing occurs..."
      "upon" — event-triggered: "UPON delivery of the Goods..."
      "after" — sequential: "AFTER the Closing Date..."
      "within" — bounded: "WITHIN 30 days of the date hereof..."

    EXCEPTION (scope limitation):
      "unless" — negative condition: "UNLESS otherwise agreed..."
      "except" — carve-out: "EXCEPT as provided in Section 2.3..."
      "notwithstanding" — override: "NOTWITHSTANDING anything to
        the contrary..."

    For multi-word markers where Stanza only tagged the first word
    (e.g., "provided" for "provided that"), we attempt to look up
    the single word first, then try prefix matching.

    Args:
        mark_text: The mark word text (lowercase, stripped).

    Returns:
        ConditionType enum value. Defaults to TRIGGER if the mark
        word is unrecognized (conservative assumption: an unknown
        advcl+mark is most likely a conditional).
    """
    # Exact match lookup in the taxonomy.
    if mark_text in self._markers:
        return self._markers[mark_text]

    # Multi-word marker: check if mark_text starts any known marker.
    # E.g., "provided" matches "provided that".
    for marker, ctype in self._markers.items():
        if marker.startswith(mark_text + " "):
            logger.debug(
                "Matched mark '%s' to multi-word marker '%s' -> %s",
                mark_text, marker, ctype.value,
            )
            return ctype

    # Fallback: also check if marker starts with mark_text
    # (for cases where the Stanza token includes more than expected).
    # This handles rare tokenization edge cases.
    for marker, ctype in self._markers.items():
        if mark_text.startswith(marker + " ") or mark_text == marker:
            return ctype

    # Unknown mark word — default to TRIGGER.
    # Rationale: In legal English, most unrecognized advcl+mark
    # patterns are conditional-like (trigger) constructions.
    # Temporal markers ("when", "upon") are reliably parsed by Stanza.
    # If the mark is unrecognized, it is more likely a less common
    # trigger word than an unrecognized temporal or exception marker.
    logger.debug(
        "Unrecognized mark word '%s' — defaulting to TRIGGER", mark_text
    )
    return ConditionType.TRIGGER
