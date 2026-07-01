"""
Main detection interface for PolarityDetector.
"""

from __future__ import annotations

from typing import Tuple

from src.extraction.schema import DependencyTree, LegalRole
from src.linguistic.ud_features import (
    find_aux_verb,
    find_obl_agent,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def detect(
    self,
    tree: DependencyTree,
    predicate_idx: int,
    predicate_lemma: str = "",
) -> Tuple[LegalRole, str]:
    """Detect the legal role and polarity for the subject of a predicate.

    Decision logic (ordered, first match wins):

    1. Lexical override: If predicate_lemma is "indemnify", the subject
       is INDEMNIFYING_PARTY regardless of modality. This is a legal
       convention — indemnification carries a specific obligation class.

    2. Detect negation: Check if the predicate has a neg dependent
       (neg relation) or if negative particles appear nearby.

    3. Detect modal auxiliary: Find the aux verb (aux relation).
       Extract the lemma form for rule matching.

    4. Lookup in modality_rules: Map (aux_lemma, is_negated) -> LegalRole.

    5. Fallback: If no modal is found, return OTHER.
       This happens in clauses without modals:
       - Bare present: "Seller delivers the Goods."
       - Past tense: "Seller delivered the Goods."
       - Infinitival complements: "Seller agrees to deliver..."

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate.
        predicate_lemma: Lemma form of the predicate verb (for
                         lexical overrides like "indemnify").

    Returns:
        (LegalRole, polarity) tuple.
        - role: Classified legal role enum.
        - polarity: "positive" or "negative" string.
    """
    # Step 1: Lexical override for indemnification.
    # "Indemnify" and its morphological variants always designate
    # the subject as the indemnifying party. This overrides any
    # modal-based classification because the legal semantics of
    # indemnification are distinct from simple obligation.
    if predicate_lemma.lower() in ("indemnify", "indemnified"):
        logger.debug(
            "Lexical override: predicate '%s' -> INDEMNIFYING_PARTY",
            predicate_lemma,
        )
        return (LegalRole.INDEMNIFYING_PARTY, "positive")

    # Step 2: Detect negation.
    is_negated = self._is_negated(tree, predicate_idx)
    polarity = "negative" if is_negated else "positive"

    # Step 3: Detect modal auxiliary.
    modal_word, _ = detect_modality(self, tree, predicate_idx)

    if not modal_word:
        # No modal auxiliary found.
        # The predicate has no deontic marker — cannot determine role
        # from syntax alone. This is common in definition clauses,
        # recitals, and bare statements of fact.
        logger.debug(
            "No modal auxiliary found for predicate at index %d — "
            "role is OTHER", predicate_idx,
        )
        return (LegalRole.OTHER, polarity)

    # Step 4: Lookup (modal, negated) -> role.
    role = self._lookup.get((modal_word.lower(), is_negated))

    if role is not None:
        logger.debug(
            "Role classified: aux='%s', negated=%s -> %s",
            modal_word, is_negated, role.value,
        )
        return (role, polarity)

    # Step 5: The modal was found but doesn't match any rule.
    # This could be a non-deontic modal (e.g., "can" for ability,
    # "will" for future) or a modal not in our rule set.
    # Default to OTHER — the validator will note this as uncertain.
    logger.debug(
        "Modal '%s' (negated=%s) does not match any rule — "
        "role is OTHER", modal_word, is_negated,
    )
    return (LegalRole.OTHER, polarity)


def detect_modality(
    self,
    tree: DependencyTree,
    predicate_idx: int,
) -> Tuple[str, str]:
    """Detect the specific modal auxiliary and polarity.

    More granular than detect() — returns the actual modal word
    for use in linguistic evidence and error explanations.

    This is used by:
      - The validator's _build_linguistic_evidence() to populate
        the modality_aux field.
      - The error analyzer to explain role mismatches.

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate.

    Returns:
        (modal_word, polarity) tuple.
        - modal_word: The lemma of the modal auxiliary (e.g., "shall",
          "may", "must"), or "" if no auxiliary found.
        - polarity: "positive" or "negative" string.
    """
    # Find the auxiliary verb (if any).
    aux_token = find_aux_verb(tree, predicate_idx)

    # Determine polarity from negation.
    is_negated = self._is_negated(tree, predicate_idx)
    polarity = "negative" if is_negated else "positive"

    if aux_token is None:
        return ("", polarity)

    # Use the lemma form for consistent rule matching.
    # Stanza lemmatizes modals to their canonical form:
    # "must" -> "must", "shall" -> "shall", "may" -> "may".
    modal_word = aux_token.lemma.lower() if aux_token.lemma else aux_token.text.lower()

    logger.debug(
        "Detected modality: aux='%s' (lemma='%s'), polarity='%s'",
        aux_token.text, modal_word, polarity,
    )

    return (modal_word, polarity)


def detect_role_with_voice(
    self,
    tree: DependencyTree,
    predicate_idx: int,
    predicate_lemma: str = "",
    is_passive: bool = False,
) -> LegalRole:
    """Detect legal role, accounting for passive voice.

    In passive constructions, the surface subject is the patient,
    not the agent. The role (obligor, etc.) should be assigned to
    the semantic AGENT (obl:agent), not the surface subject.

    This method handles that correctly by:
      1. For active voice: role applies to nsubj (surface subject = agent).
      2. For passive voice: role applies to obl:agent (semantic agent).
      3. For agentless passive: role cannot be determined — returns OTHER.

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate.
        predicate_lemma: Lemma form of the predicate.
        is_passive: Whether the predicate is in passive voice.

    Returns:
        LegalRole enum value.
    """
    role, _ = detect(self, tree, predicate_idx, predicate_lemma)

    if not is_passive:
        return role

    # In passive, check if the obl:agent exists.
    # If the agent is expressed via obl:agent, the role applies
    # to that agent. If agentless, the role is "free-floating"
    # — we know what modality applies but not WHO it applies to.
    agent = find_obl_agent(tree, predicate_idx)
    if agent is None:
        # Agentless passive with a modality: e.g.,
        # "All notices shall be delivered in writing."
        # We know there's an obligation (shall), but the
        # syntactic agent is not expressed.
        # The LLM will need to infer from document context
        # which party bears the obligation.
        logger.debug(
            "Agentless passive with modality — role cannot be "
            "assigned to a syntactic agent."
        )
        return LegalRole.OTHER

    return role


def get_modality_evidence(
    self,
    tree: DependencyTree,
    predicate_idx: int,
) -> dict:
    """Extract all modality-related evidence for error analysis.

    Returns a comprehensive dict of modality features used by
    the error analyzer to explain role classification decisions.

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate.

    Returns:
        Dict with keys:
          - has_aux (bool): aux relation present
          - aux_lemma (str): lemma of the auxiliary (if any)
          - aux_text (str): surface text of the auxiliary
          - is_negated (bool): negation detected
          - polarity (str): "positive" or "negative"
          - found_roles (list): all roles matching the (aux, neg) pair
    """
    aux_token = find_aux_verb(tree, predicate_idx)
    is_negated = self._is_negated(tree, predicate_idx)

    evidence = {
        "has_aux": aux_token is not None,
        "aux_lemma": aux_token.lemma.lower() if aux_token and aux_token.lemma else "",
        "aux_text": aux_token.text.lower() if aux_token else "",
        "is_negated": is_negated,
        "polarity": "negative" if is_negated else "positive",
        "matched_roles": [],
    }

    # Check which roles match this (aux, negated) combination.
    if aux_token and aux_token.lemma:
        aux_lower = aux_token.lemma.lower()
        for (role_aux, role_neg), role in self._lookup.items():
            if role_aux == aux_lower and role_neg == is_negated:
                evidence["matched_roles"].append(role.value)

    return evidence
