"""
Predicate identification: find_root_predicate and find_all_predicates.
"""

from __future__ import annotations

from typing import Optional, List

from src.extraction.schema import DependencyTree, Token
from src.utils.logging import get_logger

logger = get_logger(__name__)


def find_root_predicate(tree: DependencyTree) -> Optional[Token]:
    """Locate the root predicate (main clause verb) of the sentence.

    In UD, the root token is the one with head=0. In well-formed
    sentences, this is typically the main verb. For legal clauses,
    this identifies the core legal action (e.g., "deliver", "terminate",
    "indemnify").

    UD basis: root relation — head=0 marks the syntactic root.

    Edge cases handled:
      - Multiple roots (malformed parse): returns the first VERB root.
        If no VERB root, returns the first root token.
      - No root (empty tree): returns None.
      - Root is not a VERB (e.g., copular "is" with nominal predicate):
        checks for xcomp complement that carries the real action.

    Args:
        tree: Parsed dependency tree.

    Returns:
        The root Token, or None if no root found.

    Legal text example:
        "Seller shall deliver the Goods within 30 days."
        -> root = Token("deliver", upos="VERB", head=0)
    """
    root_idx = tree.root_index
    if root_idx is None:
        logger.debug("No root token found in tree (empty or malformed)")
        return None

    root_token = tree.get_token(root_idx)
    if root_token is None:
        return None

    # In legal text, the root is almost always a VERB (the main action).
    # If it's not a verb, we may have a copular construction (e.g.,
    # "The agreement IS binding") where the content predicate is
    # elsewhere. Try to find the real predicate via xcomp or ccomp.
    if root_token.upos != "VERB" and root_token.upos != "AUX":
        logger.debug(
            "Root token '%s' is %s, not VERB. Searching for xcomp/ccomp.",
            root_token.text, root_token.upos,
        )
        # Look for an open clausal complement (xcomp) that carries
        # the semantic predicate. Example: "Seller IS required to deliver"
        # -> root=AUX("is"), xcomp(required, deliver) points to action.
        for child in tree.get_children(root_idx):
            if child.deprel in ("xcomp", "ccomp") and child.upos == "VERB":
                logger.debug(
                    "Found predicate via %s: %s (index %d)",
                    child.deprel, child.lemma, child.index,
                )
                return child

    return root_token


def find_all_predicates(tree: DependencyTree) -> List[Token]:
    """Find all verb tokens that could serve as predicates.

    Scans for all VERB tokens and ranks them by proximity to the root.
    Useful for sentences with multiple clauses where the main predicate
    is not the syntactic root.

    Args:
        tree: Dependency tree.

    Returns:
        List of VERB Token objects, root-verb first.
    """
    verbs = tree.find_tokens_by_upos("VERB")
    if not verbs:
        return []

    root_idx = tree.root_index

    # Sort by: (distance to root ascending, index ascending)
    def _distance_to_root(token: Token) -> int:
        if root_idx is None:
            return 0
        path = tree.get_path_to_root(token.index)
        return len(path)

    verbs.sort(key=lambda t: (_distance_to_root(t), t.index))
    return verbs
