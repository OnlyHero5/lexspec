"""
Argument extraction: find_nsubj, find_obj, find_nsubj_pass, find_obl_agent.
"""

from __future__ import annotations

from typing import Optional

from src.extraction.schema import DependencyTree, Token
from src.utils.logging import get_logger

logger = get_logger(__name__)


def find_nsubj(tree: DependencyTree, predicate_idx: int) -> Optional[Token]:
    """Find the nominal subject (active voice actor/agent).

    UD: nsubj(predicate, subject) — the syntactic subject of a clause.
    In active voice, this is the agent/actor — the entity that PERFORMS
    the action described by the predicate.

    In legal text, this identifies the party who owes/does the action:
      "SELLER shall deliver"  -> nsubj(deliver, Seller)
      "BUYER must pay"        -> nsubj(pay, Buyer)

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate token.

    Returns:
        The nsubj Token, or None if no nominal subject exists
        (e.g., imperative clauses, impersonal constructions).

    Note:
        In UD, a verb can have at most one nsubj. Coordination
        ("Buyer and Seller shall deliver") uses the conj relation
        — the first conjunct is nsubj, others are conj dependents.
    """
    children = tree.get_children(predicate_idx, deprel="nsubj")
    if children:
        if len(children) > 1:
            logger.debug(
                "Multiple nsubj candidates for predicate %d — "
                "returning first. This may indicate a parse error.",
                predicate_idx,
            )
        return children[0]
    return None


def find_obj(tree: DependencyTree, predicate_idx: int) -> Optional[Token]:
    """Find the direct object (active voice patient/theme).

    UD: obj(predicate, object) — the direct object of a transitive verb.
    In active voice, this is the patient/theme — the entity that
    UNDERGOES the action.

    In legal text, this identifies WHAT is being acted upon:
      "deliver THE GOODS"       -> obj(deliver, goods)
      "pay ALL AMOUNTS DUE"     -> obj(pay, amounts)
      "indemnify THE COMPANY"   -> obj(indemnify, company)

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate token.

    Returns:
        The obj Token, or None if the predicate is intransitive
        (no direct object, e.g., "The Agreement terminates").

    Note:
        Legal verbs are overwhelmingly transitive (they act ON something).
        Intransitive predicates exist ("The Agreement shall terminate")
        but do not produce an obj — the action has no direct patient.
    """
    children = tree.get_children(predicate_idx, deprel="obj")
    if children:
        return children[0]
    return None


def find_nsubj_pass(tree: DependencyTree,
                    predicate_idx: int) -> Optional[Token]:
    """Find the passive nominal subject (surface subject, semantic patient).

    UD: nsubj:pass(predicate, patient) — the syntactic subject of a
    passive clause. The surface subject is semantically the patient/theme,
    NOT the actor. The actor appears as obl:agent (by-phrase).

    This is the most common source of LLM extraction errors: LLMs often
    treat the surface subject (nsubj:pass) as the agent, producing
    subject-object reversals in extracted triplets.

    Example:
      "THE GOODS shall be delivered by Seller"
        -> nsubj:pass(delivered, goods)  <-- PATIENT (surface subject)
        -> obl:agent(delivered, Seller)  <-- AGENT (real actor)

    The validator uses this function to detect when the LLM has confused
    the passive subject for the agent, and corrects accordingly.

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate token.

    Returns:
        The nsubj:pass Token (semantic patient), or None if the
        predicate is not in passive voice.
    """
    children = tree.get_children(predicate_idx, deprel="nsubj:pass")
    if children:
        return children[0]
    return None


def find_obl_agent(tree: DependencyTree,
                   predicate_idx: int) -> Optional[Token]:
    """Find the oblique agent (by-phrase, semantic actor in passive).

    UD: obl:agent(predicate, agent) — the agent in a passive construction,
    typically introduced by the preposition "by". This is the TRUE
    semantic actor — the entity that performs the action.

    The UD guidelines note that obl:agent is specifically for passive
    agents marked by "by" (or equivalents in non-English languages).
    It is a language-specific subtype of the general obl (oblique) relation.

    In legal text, the obl:agent is the party who actually performs
    the action despite the surface subject being something else:
      "delivered BY SELLER"       -> obl:agent(delivered, Seller)
      "indemnified BY THE COMPANY" -> obl:agent(indemnified, company)

    Edge case: Agentless passive.
      "The goods were delivered." (no "by" phrase)
      -> obl:agent is None. The agent exists semantically but is
         not syntactically expressed. The validator flags this as
         REFLEXION_REQUIRED because the LLM must infer the agent
         from discourse context.

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate token.

    Returns:
        The obl:agent Token (semantic agent/actor), or None.
    """
    children = tree.get_children(predicate_idx, deprel="obl:agent")
    if children:
        return children[0]
    return None
