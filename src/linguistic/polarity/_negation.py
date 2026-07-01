"""
Negation detection: _is_negated and _has_lexical_negation.
"""

from __future__ import annotations

from src.extraction.schema import DependencyTree
from src.linguistic.ud_features import find_negation
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _is_negated(self, tree: DependencyTree, predicate_idx: int) -> bool:
    """Check if the predicate is negated.

    Negation detection looks for three patterns (in priority order):

    1. Direct neg dependent: neg(predicate, not)
       "shall NOT assign" -> neg(assign, not)
       The most common negation pattern in legal English.

    2. Negation in aux dependents:
       Some parses attach "not" to the auxiliary rather than the
       main verb: "shall NOT" where NOT depends on "shall" via
       the neg relation. We check the aux token's children.

    3. Lexical negation tokens near the predicate:
       - "no" as a determiner modifying the subject:
         "NO party shall assign..." — the negation is on "party",
         not directly on the verb, but it creates prohibition.
       - "neither ... nor" constructions
       - "never" as an adverb

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate.

    Returns:
        True if negation is detected.
    """
    # Pattern 1: Direct neg dependent on the predicate.
    neg_token = find_negation(tree, predicate_idx)
    if neg_token is not None:
        logger.debug(
            "Negation detected: neg(%s, %s) at predicate index %d",
            tree.get_token(predicate_idx).text if tree.get_token(predicate_idx) else "?",
            neg_token.text,
            predicate_idx,
        )
        return True

    # Pattern 2: Check if any auxiliary has a neg dependent.
    # In some parse styles, "not" attaches to the auxiliary:
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

    # Pattern 3: Lexical negation in the vicinity of the predicate.
    # Check tokens near the predicate for negation words.
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
    """Check for lexical negation tokens near the predicate.

    Scans tokens within `window` positions of the predicate for:
      - "no" (determiner — "no party shall", "no assignment may")
      - "neither" (correlative — "neither party shall")
      - "nor" (correlative — "...nor shall any party")
      - "never" (adverb — "shall never assign")

    These negation words carry the same legal effect as "not"
    but may not be attached via the neg dependency relation
    in all parse styles.

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate.
        window: Number of tokens to scan on each side.

    Returns:
        True if a lexical negation token is found within the window.
    """
    negation_words = {"no", "neither", "nor", "never", "nothing"}

    start = max(1, predicate_idx - window)
    end = min(tree.token_count, predicate_idx + window)

    for i in range(start, end + 1):
        token = tree.get_token(i)
        if token is not None:
            text_lower = token.text.lower().strip()
            if text_lower in negation_words:
                # "no" as a determiner only counts as negation
                # if it precedes the subject (scope over NP).
                # "no" as part of "whether or no" does not count.
                if text_lower == "no" and token.upos != "DET":
                    # "no" that is not a determiner is unlikely
                    # to be negation on the predicate.
                    continue
                return True

    return False
