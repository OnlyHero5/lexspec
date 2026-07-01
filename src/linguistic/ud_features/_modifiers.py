"""
Modifier extraction: aux, aux:pass, negation, acl:relcl, noun phrase span, copular check.
"""

from __future__ import annotations

from typing import Optional, List, Tuple

from src.extraction.schema import DependencyTree, Token
from src.utils.logging import get_logger

logger = get_logger(__name__)


def find_aux_verb(tree: DependencyTree,
                  predicate_idx: int) -> Optional[Token]:
    """Find the auxiliary verb modifying the predicate.

    UD: aux(predicate, auxiliary) — an auxiliary verb.
    In legal text, the auxiliary carries deontic modality:
      "Seller SHALL deliver" -> aux(deliver, shall)
      "Buyer MAY terminate"  -> aux(terminate, may)
      "Party MUST pay"       -> aux(pay, must)

    The aux verb is critical for legal role classification:
      shall/must  -> obligation -> OBLIGOR
      may         -> permission -> RIGHT_HOLDER
      shall not   -> prohibition -> PROHIBITED_PARTY

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate token.

    Returns:
        The aux Token, or None if no auxiliary is present
        (e.g., bare infinitive in subordinate clauses).
    """
    children = tree.get_children(predicate_idx, deprel="aux")
    if children:
        # Return the first aux. Multiple aux are possible in complex
        # verb phrases ("shall have been delivered") — we take the
        # first (leftmost) one which carries the primary modality.
        return children[0]
    return None


def find_aux_pass(tree: DependencyTree,
                  predicate_idx: int) -> Optional[Token]:
    """Find the passive auxiliary (be/get) on the predicate.

    UD: aux:pass(predicate, be_aux) — the passive auxiliary.
    In legal text:
      "the Goods ARE delivered"   -> aux:pass(delivered, are)
      "the Agreement was breached" -> aux:pass(breached, was)

    The presence of aux:pass confirms that the construction is
    morphological passive, not an adjectival past participle.

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate token.

    Returns:
        The aux:pass Token, or None.
    """
    children = tree.get_children(predicate_idx, deprel="aux:pass")
    if children:
        return children[0]
    return None


def find_negation(tree: DependencyTree,
                  predicate_idx: int) -> Optional[Token]:
    """Find the negation marker on the predicate.

    UD: neg(predicate, negation) — negation modifier.
    In legal text:
      "shall NOT assign"  -> neg(assign, not)
      "may NOT disclose"  -> neg(disclose, not)
      "NO party shall"    -> neg(shall, no) [rare — usually neg attaches to verb]

    Negation is critical for distinguishing obligations from prohibitions:
      "shall deliver"    -> OBLIGATION (duty to deliver)
      "shall NOT deliver" -> PROHIBITION (duty NOT to deliver)

    Args:
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate token.

    Returns:
        The neg Token, or None if the predicate is not negated.
    """
    children = tree.get_children(predicate_idx, deprel="neg")
    if children:
        return children[0]
    return None


def has_acl_relcl(tree: DependencyTree, token_idx: int) -> bool:
    """Check if a token has a relative clause modifier.

    UD: acl:relcl(head, clause) — a relative clause modifying a nominal.
    In legal text:
      "the Party WHO receives notice" -> acl:relcl(Party, receives)
      "any amount THAT exceeds the cap" -> acl:relcl(amount, exceeds)

    Relative clauses embed predicate-argument structures inside noun
    phrases, which can confuse LLM extraction when the LLM tries to
    extract a predicate from within the relative clause instead of
    the main clause.

    Args:
        tree: Dependency tree.
        token_idx: Token index to check for relative clause modifiers.

    Returns:
        True if the token has at least one acl:relcl dependent.
    """
    children = tree.get_children(token_idx, deprel="acl:relcl")
    return len(children) > 0


def find_acl_relcl_head(tree: DependencyTree,
                        token_idx: int) -> Optional[Token]:
    """Get the head of the relative clause modifying a token.

    Args:
        tree: Dependency tree.
        token_idx: Token index of the head noun.

    Returns:
        The acl:relcl head Token, or None.
    """
    children = tree.get_children(token_idx, deprel="acl:relcl")
    if children:
        return children[0]
    return None


def get_noun_phrase_span(tree: DependencyTree,
                         head_idx: int) -> Tuple[str, List[int]]:
    """Extract the full text of a noun phrase given its syntactic head.

    Walks the subtree of the noun to collect determiners, adjectives,
    prepositional modifiers, and relative clauses that are part of
    the noun phrase. This produces the "maximal NP" span that an
    LLM extractor should produce as the subject or object text.

    Example: For "all outstanding amounts due under this Agreement":
      head = "amounts" (obj or nsubj)
      span = "all outstanding amounts due under this Agreement"

    Args:
        tree: Dependency tree.
        head_idx: Index of the noun phrase head token.

    Returns:
        (full_np_text, list_of_token_indices) tuple.
    """
    # Collect the full subtree of the head noun.
    subtree_indices = set(tree._collect_subtree(head_idx))

    # Also include any pre-modifiers (determiners, adjectives) that
    # may be attached as `det`, `amod`, `nummod` dependents.
    # These are already in the subtree via _collect_subtree since
    # they are dependents of the head noun.
    #
    # For complex NPs, we also need to include any tokens that are
    # dependents of the head via `nmod`, `obl`, `acl`, etc.
    # _collect_subtree already handles this transitively.

    sorted_indices = sorted(subtree_indices)
    tokens_sorted = [
        tree.get_token(i) for i in sorted_indices
    ]
    tokens_sorted = [t for t in tokens_sorted if t is not None]
    text = " ".join(t.text for t in tokens_sorted)

    return text, sorted_indices


def is_copular_construction(tree: DependencyTree,
                            predicate_idx: int) -> bool:
    """Check if the predicate is part of a copular construction.

    UD: cop(predicate, be) — copula relation.
    In legal text: "The Agreement IS binding" -> cop(binding, is)

    Copular constructions use "be" as an auxiliary with an adjective
    or nominal predicate. In such cases, the real "action" may be
    in a complement clause or the nominal predicate itself.

    Args:
        tree: Dependency tree.
        predicate_idx: Index of the potential predicate.

    Returns:
        True if the token has a cop dependent.
    """
    cop_children = tree.get_children(predicate_idx, deprel="cop")
    return len(cop_children) > 0
