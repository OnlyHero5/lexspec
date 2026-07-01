"""
Dependency topology: paths, distances, coordination.
"""

from __future__ import annotations

from typing import Optional, List, Tuple, Dict

from src.extraction.schema import DependencyTree, Token
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_dependency_path(
    tree: DependencyTree,
    from_idx: int,
    to_idx: int,
) -> Optional[List[Token]]:
    """Find the dependency path between two tokens.

    Uses bidirectional BFS up the dependency tree. Both tokens walk
    upward via their head pointers until their paths intersect.
    The intersection is the lowest common ancestor (LCA).

    This is used for dependency path legality checking in linguistic
    metrics: if the LLM extracts a subject from a token that is on
    a different branch of the tree from the predicate, the extraction
    is syntactically suspect.

    Args:
        tree: Dependency tree.
        from_idx: Starting token index (1-based).
        to_idx: Ending token index (1-based).

    Returns:
        List of Token objects on the path from from_idx to to_idx
        (inclusive of both endpoints), or None if no path exists
        (disconnected tree or missing tokens).
    """
    from_token = tree.get_token(from_idx)
    to_token = tree.get_token(to_idx)
    if from_token is None or to_token is None:
        return None

    # Build upward path from the 'from' token to the root.
    # We track visited indices with their paths for LCA detection.
    path_from: Dict[int, List[Token]] = {}
    current = from_token
    from_path: List[Token] = []
    while current is not None:
        from_path.append(current)
        path_from[current.index] = list(from_path)
        if current.head == 0:
            break
        current = tree.get_token(current.head)

    # Build upward path from the 'to' token to the root,
    # checking for intersection at each step.
    current = to_token
    to_path: List[Token] = []
    while current is not None:
        to_path.append(current)
        # Check if we've intersected with the 'from' path.
        if current.index in path_from:
            # LCA found. Combine the paths:
            # from_path up to (and including) LCA,
            # then to_path reversed (excluding LCA to avoid duplication).
            from_path_to_lca = path_from[current.index]
            # to_path_to_lca is the current to_path up to LCA,
            # reversed so it goes LCA -> from, then without the first (LCA).
            to_path_to_lca = list(reversed(to_path))
            combined = from_path_to_lca + to_path_to_lca[1:]
            return combined
        if current.head == 0:
            break
        current = tree.get_token(current.head)

    # No intersection — the tokens are in disconnected components.
    return None


def compute_mean_dependency_distance(tree: DependencyTree) -> float:
    """Compute Mean Dependency Distance (MDD) for the sentence.

    MDD = mean( |head_index - dependent_index| ) over all non-root tokens.
    Higher MDD indicates more complex sentence structure with
    longer-distance syntactic relationships.

    This metric is used for:
      1. Long-distance dependency phenomenon classification (§5.2 of design).
      2. Estimating sentence complexity for balanced test set sampling.
      3. Identifying sentences likely to cause LLM extraction errors.

    Args:
        tree: Dependency tree.

    Returns:
        Mean dependency distance as a float.
        Returns 0.0 if the tree has fewer than 2 tokens (no dependencies).
    """
    if tree.token_count < 2:
        return 0.0

    total_distance = 0
    non_root_count = 0

    for token in tree.tokens:
        if token.head == 0:
            continue  # Root has no dependency distance
        distance = abs(token.index - token.head)
        total_distance += distance
        non_root_count += 1

    if non_root_count == 0:
        return 0.0

    return total_distance / non_root_count


def find_long_distance_dependencies(
    tree: DependencyTree,
    threshold: int = 5,
) -> List[Tuple[Token, Token, int]]:
    """Find dependency pairs where the linear distance exceeds threshold.

    Long-distance dependencies (>5 intervening tokens) are a known
    source of LLM extraction errors because the model must track
    relationships across a large context window within a single sentence.

    In legal text, long-distance dependencies are common due to:
      - Intervening adverbial phrases: "Seller shall, within 30 days
        after the Closing Date and subject to the conditions set forth
        in Section 2.3, deliver the Goods."
      - Embedded relative clauses defining the object.

    Args:
        tree: Dependency tree.
        threshold: Minimum token distance to flag as long-distance.

    Returns:
        List of (dependent, head, distance) tuples for pairs exceeding
        the threshold, sorted by distance descending.
    """
    result: List[Tuple[Token, Token, int]] = []

    for token in tree.tokens:
        if token.head == 0:
            continue
        distance = abs(token.index - token.head)
        if distance > threshold:
            head_token = tree.get_token(token.head)
            if head_token is not None:
                result.append((token, head_token, distance))

    result.sort(key=lambda x: x[2], reverse=True)
    return result


def get_conjuncts(tree: DependencyTree, token_idx: int) -> List[Token]:
    """Get all conjuncts of a token (including the token itself).

    UD: conj(head, conjunct) — coordination relation.
    In legal text:
      "Buyer AND Seller shall deliver"
        -> nsubj(deliver, Buyer)
        -> conj(Buyer, Seller)  -- "Seller" is a conjunct of "Buyer"

    This expands a single nsubj/obj token into the full coordinated
    phrase, so that "Buyer and Seller" is treated as a compound subject
    rather than just "Buyer".

    Args:
        tree: Dependency tree.
        token_idx: Index of the head conjunct token.

    Returns:
        List of all conjunct tokens (head first, then conj dependents
        in order), including the token itself.
    """
    result = [tree.get_token(token_idx)]
    if result[0] is None:
        return []

    # Collect all immediate conj dependents.
    for child in tree.get_children(token_idx, deprel="conj"):
        result.append(child)

    return [t for t in result if t is not None]


def get_conjunct_text(tree: DependencyTree, token_idx: int) -> str:
    """Get the full text span of a coordination, including conjunctions.

    For "Buyer and Seller", returns the text covering all conjunct
    tokens plus the conjunction token ("and") between them.

    Args:
        tree: Dependency tree.
        token_idx: Index of the head conjunct.

    Returns:
        Reconstructed text of the full coordinated noun phrase.
    """
    conjuncts = get_conjuncts(tree, token_idx)
    if len(conjuncts) <= 1:
        # Single token, no coordination.
        tok = tree.get_token(token_idx)
        return tok.text if tok else ""

    # For coordinated phrases, extract all tokens between the
    # first conjunct and the last conjunct (inclusive).
    # This captures the and/or/cc token between conjuncts.
    indices = [c.index for c in conjuncts]
    min_idx = min(indices)
    max_idx = max(indices)

    span_tokens = [
        tree.get_token(i) for i in range(min_idx, max_idx + 1)
    ]
    span_tokens = [t for t in span_tokens if t is not None]
    return " ".join(t.text for t in span_tokens)
