"""
Passive detection: is_passive, is_passive_loose, get_passive_features.
"""

from __future__ import annotations

from src.extraction.schema import DependencyTree
from src.linguistic.ud_features import (
    find_nsubj_pass,
    find_obl_agent,
    find_aux_pass,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def is_passive(tree: DependencyTree, predicate_idx: int) -> bool:
    """Detect whether a predicate is in passive voice.

    Detection rules (both should be satisfied for confident detection):

    1. nsubj:pass relation exists: There is a surface subject that is
       semantically patient. This is the DEFINING characteristic of
       passive voice — the argument that would be the object in active
       voice has been promoted to subject position.

    2. aux:pass relation exists: A passive auxiliary "be"/"get" is
       present. This CONFIRMS that the construction is morphological
       passive (verb form: be + past participle), distinguishing it
       from adjectival participial constructions ("The door remained
       closed" — nsubj:pass may exist but no aux:pass, so it is
       adjectival, not passive).

    Lingustic reasoning for requiring both:
      - nsubj:pass alone can occur with adjectival past participles
        ("the documents attached" — "attached" is adjectival, not passive)
        and with certain unaccusative constructions.
      - aux:pass alone (without nsubj:pass) suggests expletive or
        impersonal constructions ("It was decided that..." — the
        expletive "it" is nsubj, not nsubj:pass).
      - Both together = unambiguous morphological passive.

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate token.

    Returns:
        True if passive voice is confidently detected.

    Note:
        For agentless passives ("The Goods were delivered"), this
        returns True because nsubj:pass + aux:pass are both present.
        The absence of obl:agent is handled separately in restore_args().
    """
    # Criterion 1: Is there a passive subject (patient in subject position)?
    has_passive_subject = find_nsubj_pass(tree, predicate_idx) is not None

    # Criterion 2: Is there a passive auxiliary?
    has_passive_aux = find_aux_pass(tree, predicate_idx) is not None

    if has_passive_subject and has_passive_aux:
        logger.debug(
            "Passive detected at predicate index %d: nsubj:pass + aux:pass",
            predicate_idx,
        )
        return True

    if has_passive_subject and not has_passive_aux:
        # Possible adjectival participle or unaccusative.
        # Example: "The documents attached hereto" —
        # "attached" may have nsubj:pass but is adjectival, not passive.
        # In legal text, this is uncommon but worth logging.
        logger.debug(
            "nsubj:pass found at predicate %d but no aux:pass — "
            "may be adjectival, not passive. Treating as non-passive.",
            predicate_idx,
        )
        return False

    if not has_passive_subject and has_passive_aux:
        # Rare: aux:pass without nsubj:pass.
        # Could happen with impersonal passives using expletive "it".
        # Example: "It is agreed that..." — "it" is nsubj (expletive),
        # not nsubj:pass. Treat as passive since aux:pass is present.
        logger.debug(
            "aux:pass found at predicate %d but no nsubj:pass — "
            "possible impersonal passive. Treating as passive.",
            predicate_idx,
        )
        return True

    return False


def is_passive_loose(tree: DependencyTree, predicate_idx: int) -> bool:
    """Loose passive detection — requires only nsubj:pass OR aux:pass.

    This is a more permissive check used for test set sampling
    (phenomenon classification) where we want to catch borderline
    cases. The strict is_passive() is used for validation/correction
    where false positives are more costly than false negatives.

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate token.

    Returns:
        True if either nsubj:pass or aux:pass is present.
    """
    return (
        find_nsubj_pass(tree, predicate_idx) is not None
        or find_aux_pass(tree, predicate_idx) is not None
    )


def get_passive_features(
    tree: DependencyTree,
    predicate_idx: int,
) -> dict:
    """Extract all passive-related features for error analysis.

    Returns a dictionary of features that characterize the passive
    construction. Used by the error analyzer to explain WHY a
    particular passive construction caused an LLM extraction error.

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate token.

    Returns:
        Dict with keys:
          - is_passive (bool): strict passive detection result
          - is_passive_loose (bool): loose detection result
          - has_nsubj_pass (bool): nsubj:pass relation present
          - has_aux_pass (bool): aux:pass relation present
          - has_obl_agent (bool): obl:agent relation present
          - is_agentless (bool): passive without expressed agent
          - subject_text (str): nsubj:pass text (surface subject / patient)
          - agent_text (str): obl:agent text (if present)
    """
    nsubj_pass = find_nsubj_pass(tree, predicate_idx)
    aux_pass = find_aux_pass(tree, predicate_idx)
    obl_agent = find_obl_agent(tree, predicate_idx)

    return {
        "is_passive": is_passive(tree, predicate_idx),
        "is_passive_loose": is_passive_loose(tree, predicate_idx),
        "has_nsubj_pass": nsubj_pass is not None,
        "has_aux_pass": aux_pass is not None,
        "has_obl_agent": obl_agent is not None,
        "is_agentless": (
            is_passive(tree, predicate_idx)
            and obl_agent is None
        ),
        "subject_text": nsubj_pass.text if nsubj_pass else "",
        "agent_text": obl_agent.text if obl_agent else "",
    }
