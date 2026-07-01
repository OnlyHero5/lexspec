"""
Passive argument restoration: restore_passive_args, get_active_args.
"""

from __future__ import annotations

from typing import Optional, Tuple

from src.extraction.schema import DependencyTree, Token
from src.linguistic.ud_features import (
    find_nsubj_pass,
    find_obl_agent,
    find_nsubj,
    find_obj,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def restore_passive_args(
    tree: DependencyTree,
    predicate_idx: int,
) -> Tuple[Optional[Token], Optional[Token]]:
    """Restore semantic agent and patient from a passive construction.

    Maps surface syntax to semantic roles by "undoing" the passive
    transformation:

    Surface syntax (passive)         Semantic roles (active equivalent)
    ----------------------------     ----------------------------------
    nsubj:pass token  (surface subj) → Semantic PATIENT  (→ triplet object)
    obl:agent token   (by-phrase)    → Semantic AGENT    (→ triplet subject)

    This mapping is the critical correction for LLM errors: the LLM
    often treats the nsubj:pass as the agent (subject), but it is
    actually the patient (object).

    Example transformation:
      Input:  "The Goods shall be delivered by Seller within 30 days."
      Output: agent=Token("Seller")  (→ subject in triplet)
              patient=Token("Goods") (→ object in triplet)

    Agentless passive handling:
      "The Goods were delivered."
      -> agent = None  (implied — REFLEXION_REQUIRED)
      -> patient = Token("Goods")  (correct object)

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate token.

    Returns:
        (agent, patient) tuple. Either may be None.
        - agent: obl:agent token (real actor → triplet subject).
                 None for agentless passives.
        - patient: nsubj:pass token (real patient → triplet object).
                   None if passive detection failed.

    Note:
        Caller should verify is_passive() before calling this method.
        If called on a non-passive predicate, patient will be None.
    """
    # The semantic AGENT in passive is the obl:agent (by-phrase actor).
    # This is the entity that performs the action and should become
    # the subject in the corrected triplet.
    agent = find_obl_agent(tree, predicate_idx)

    # The semantic PATIENT in passive is the nsubj:pass (surface subject).
    # This is the entity that undergoes the action and should become
    # the object in the corrected triplet.
    patient = find_nsubj_pass(tree, predicate_idx)

    if agent is None and patient is not None:
        # Agentless passive: the agent is semantically implied but
        # not syntactically expressed. This is common in legal text
        # when the responsible party is clear from context:
        #   "All notices shall be delivered in writing."
        # The validator marks this as REFLEXION_REQUIRED because
        # the LLM must infer who delivers the notices.
        logger.debug(
            "Agentless passive at predicate %d: agent is implied "
            "but not expressed syntactically. Marking for Reflexion.",
            predicate_idx,
        )

    if agent is not None and patient is not None:
        logger.debug(
            "Restored passive args: agent='%s' (index %d), "
            "patient='%s' (index %d)",
            agent.text, agent.index,
            patient.text, patient.index,
        )
    elif agent is not None:
        logger.debug(
            "Partial passive restoration: agent='%s', no patient found",
            agent.text,
        )
    elif patient is not None:
        logger.debug(
            "Partial passive restoration: patient='%s', no agent found "
            "(agentless passive)",
            patient.text,
        )

    return (agent, patient)


def get_active_args(
    tree: DependencyTree,
    predicate_idx: int,
) -> Tuple[Optional[Token], Optional[Token]]:
    """Get the semantic agent and patient for an active voice predicate.

    This is the active-voice counterpart of restore_passive_args().
    In active voice, the mapping is straightforward:
      - Agent  = nsubj (subject = doer)
      - Patient = obj (direct object = undergoer)

    Used by the validator when is_passive() returns False to get
    the active arguments for comparison with LLM output.

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate token.

    Returns:
        (agent, patient) tuple.
        - agent: nsubj token (the actor → triplet subject)
        - patient: obj token (the undergoer → triplet object)
    """
    agent = find_nsubj(tree, predicate_idx)
    patient = find_obj(tree, predicate_idx)
    return (agent, patient)
