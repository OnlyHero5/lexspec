"""
Validation Steps 1 & 2: Predicate Location and Voice Detection.

Step 1: Locate the root predicate via UD parse.
Step 2: Detect passive voice and restore semantic arguments.
"""

from __future__ import annotations

from typing import Optional, Tuple

from src.extraction.schema import DependencyTree, Token
from src.linguistic.ud_features import find_root_predicate
from src.linguistic.passive_detector import PassiveDetector
from src.utils.logging import get_logger

logger = get_logger(__name__)


def step1_find_predicate(tree: DependencyTree) -> Optional[Token]:
    """Step 1: Locate the root predicate via UD parse.

    The root token (head=0) is typically the main clause verb.
    This is the starting point for all subsequent analysis because
    all arguments (subject, object, conditions) are dependents of
    the root predicate in UD annotation.

    UD basis: The root of the dependency tree is the token whose
    head field is 0. In a well-formed UD tree, there is exactly
    one root. If there are multiple (malformed parse), we choose
    the first VERB root.

    For legal clauses, the root is almost always a VERB:
      "Seller shall deliver the Goods." -> root = "deliver"
      "The Agreement shall be governed by..." -> root = "governed"

    Edge case: The root may be an AUX (copula) with the semantic
    predicate in an xcomp complement. We try to resolve this:
      "The Agreement IS binding." -> root = AUX "is"
      We then check for xcomp(IS, binding) -> predicate = "binding"

    Args:
        tree: Dependency tree.

    Returns:
        Root predicate Token, or None if no root is found.
    """
    return find_root_predicate(tree)


def step2_detect_voice(
    tree: DependencyTree,
    predicate_idx: int,
) -> Tuple[bool, Optional[Token], Optional[Token]]:
    """Step 2: Detect passive voice and restore semantic arguments.

    The key insight: in passive voice, the UD surface relations
    (nsubj:pass, obl:agent) do NOT correspond to the triplet fields
    (subject=agent, object=patient). We need to map:

    Passive mapping:
      - UD nsubj:pass (surface subject) -> Semantic PATIENT -> Triplet OBJECT
      - UD obl:agent (by-phrase)       -> Semantic AGENT   -> Triplet SUBJECT

    Active mapping:
      - UD nsubj (subject)  -> Semantic AGENT   -> Triplet SUBJECT
      - UD obj (object)     -> Semantic PATIENT -> Triplet OBJECT

    This step produces the UD-derived subject and object tokens
    that are used as ground truth for validating the LLM triplet.

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate.

    Returns:
        (is_passive, ud_subject, ud_object) tuple.
        - is_passive: Whether passive voice was detected.
        - ud_subject: The semantic AGENT (actor).
          - Active: nsubj token.
          - Passive: obl:agent token (may be None for agentless passive).
        - ud_object: The semantic PATIENT (acted-upon).
          - Active: obj token.
          - Passive: nsubj:pass token.
    """
    is_passive = PassiveDetector.is_passive(tree, predicate_idx)

    if is_passive:
        logger.debug("Passive voice detected at predicate index %d", predicate_idx)
        # In passive: the semantic AGENT is obl:agent, PATIENT is nsubj:pass.
        agent, patient = PassiveDetector.restore_passive_args(tree, predicate_idx)
        # Return: ud_subject = agent (the doer), ud_object = patient (undergoer).
        return (True, agent, patient)
    else:
        logger.debug("Active voice at predicate index %d", predicate_idx)
        # In active: semantic AGENT = nsubj, PATIENT = obj.
        agent, patient = PassiveDetector.get_active_args(tree, predicate_idx)
        return (False, agent, patient)
