"""
依赖树模型 —— Universal Dependencies 依存句法树
===============================================

封装 UD 解析器的输出，提供类型化的树遍历、子树提取
和查询接口。
"""

from __future__ import annotations

from typing import Optional, List

from pydantic import BaseModel, Field

from .dependency import Token, ClauseSpan


class DependencyTree(BaseModel):
    """
    Universal Dependencies parse tree for a single sentence.

    This is the central linguistic data structure. It wraps the output of a
    UD parser (Stanza) into a clean, typed interface with helper methods for
    tree traversal. All linguistic modules consume DependencyTree objects —
    they never touch raw Stanza output directly. This isolation allows the
    parser backend to be swapped without affecting downstream code.

    Usage:
        tree = DependencyTree(text="Seller shall deliver Goods.", tokens=[...])
        root = tree.get_token(tree.root_index)
        subjects = tree.get_children(root.index, deprel="nsubj")
        condition_span = tree.get_subtree_span(advcl_head.index)
    """
    text: str = Field(
        description="Original sentence text, preserved for reference"
    )
    tokens: List[Token] = Field(
        description="Ordered list of tokens with complete UD dependency annotations"
    )

    # ------------------------------------------------------------------
    # Properties — derived information from the token list
    # ------------------------------------------------------------------

    @property
    def root_index(self) -> Optional[int]:
        """
        Index of the syntactic root token (the token whose head == 0).

        In a well-formed UD tree this is typically the main clause predicate
        (finite verb). Returns None if no root is found (e.g., empty token list).
        """
        for t in self.tokens:
            if t.head == 0:
                return t.index
        return None

    @property
    def token_count(self) -> int:
        """Total number of tokens in the sentence."""
        return len(self.tokens)

    # ------------------------------------------------------------------
    # Accessor methods — token retrieval and navigation
    # ------------------------------------------------------------------

    def get_token(self, index: int) -> Optional[Token]:
        """
        Retrieve a token by its 1-based index.

        Args:
            index: 1-based token index.

        Returns:
            Token object if found, None otherwise.
        """
        for t in self.tokens:
            if t.index == index:
                return t
        return None

    def get_children(self, head_index: int, deprel: Optional[str] = None) -> List[Token]:
        """
        Get all immediate dependents (children) of a token in the dependency tree.

        Args:
            head_index: 1-based index of the head token.
            deprel:     Optional filter — only return children with this dependency
                        relation label (e.g., 'nsubj', 'obj', 'advcl').

        Returns:
            List of child Token objects. Empty list if the token has no dependents
            or no children match the filter.
        """
        result = [t for t in self.tokens if t.head == head_index]
        if deprel is not None:
            result = [t for t in result if t.deprel == deprel]
        return result

    def get_head(self, token_index: int) -> Optional[Token]:
        """
        Get the governor (head) of a token.

        Args:
            token_index: 1-based index of the dependent token.

        Returns:
            The head Token object, or None if token_index is the root or not found.
        """
        token = self.get_token(token_index)
        if token is None or token.head == 0:
            return None
        return self.get_token(token.head)

    def get_path_to_root(self, token_index: int) -> List[Token]:
        """
        Compute the dependency path from a token up to the root.

        Useful for measuring dependency distance and identifying long-distance
        dependencies that are challenging for LLM extraction.

        Args:
            token_index: 1-based index of the starting token.

        Returns:
            Ordered list of tokens from the given token (inclusive) up to
            the root (inclusive). Returns empty list if token not found.
        """
        path = []
        current = self.get_token(token_index)
        if current is None:
            return path
        path.append(current)
        # Walk up the tree via head pointers until we reach root (head == 0)
        while current.head != 0:
            current = self.get_token(current.head)
            if current is None:
                break  # Defensive: broken tree
            path.append(current)
        return path

    # ------------------------------------------------------------------
    # Subtree methods — extracting clausal spans
    # ------------------------------------------------------------------

    def get_subtree_span(self, head_index: int) -> ClauseSpan:
        """
        Extract the full text span of a subtree rooted at head_index.

        Performs a depth-first traversal to collect all tokens dominated by
        the given head (including the head itself). Tokens are sorted by
        their surface order (1-based index) before concatenation.

        This is the primary method for extracting condition clause text:
            advcl_heads = [c for c in tree.get_children(root_idx) if c.deprel == 'advcl']
            for advcl in advcl_heads:
                span = tree.get_subtree_span(advcl.index)

        Args:
            head_index: 1-based index of the subtree root token.

        Returns:
            ClauseSpan with the sorted token indices and concatenated text.
        """
        subtree_indices = self._collect_subtree(head_index)
        subtree_indices.sort()
        tokens_sorted = [
            t for i in subtree_indices
            if (t := self.get_token(i)) is not None
        ]
        text = " ".join(t.text for t in tokens_sorted)
        # Determine the dependency relation of the subtree head
        head_token = self.get_token(head_index)
        head_deprel = head_token.deprel if head_token else ""
        return ClauseSpan(tokens=subtree_indices, text=text, deprel=head_deprel)

    def get_subtree_tokens(self, head_index: int) -> List[Token]:
        """
        Return all Token objects in the subtree rooted at head_index,
        sorted by surface order.

        Args:
            head_index: 1-based index of the subtree root.

        Returns:
            Sorted list of Token objects in the subtree.
        """
        indices = sorted(self._collect_subtree(head_index))
        return [t for i in indices if (t := self.get_token(i)) is not None]

    def _collect_subtree(self, head_index: int) -> List[int]:
        """
        Depth-first traversal to collect all token indices in the subtree
        dominated by head_index (including the head itself).

        This is a transitive closure over the ``head`` relation — we collect
        every token whose dependency path passes through head_index.

        Args:
            head_index: 1-based index of the subtree root.

        Returns:
            List of 1-based token indices (may be unsorted).
        """
        result = [head_index]
        for child in self.get_children(head_index):
            result.extend(self._collect_subtree(child.index))
        return result

    # ------------------------------------------------------------------
    # Query methods — pattern matching on the tree
    # ------------------------------------------------------------------

    def find_tokens_by_upos(self, upos: str) -> List[Token]:
        """
        Find all tokens with a given Universal POS tag.

        Args:
            upos: Universal POS tag, e.g. 'VERB', 'NOUN', 'AUX'.

        Returns:
            List of matching Token objects.
        """
        return [t for t in self.tokens if t.upos == upos]

    def find_tokens_by_deprel(self, deprel: str) -> List[Token]:
        """
        Find all tokens with a given dependency relation label.

        Args:
            deprel: Dependency relation, e.g. 'nsubj', 'obj', 'nsubj:pass'.

        Returns:
            List of matching Token objects.
        """
        return [t for t in self.tokens if t.deprel == deprel]

    def has_deprel(self, deprel: str) -> bool:
        """
        Check whether any token in the tree bears the given dependency relation.

        Args:
            deprel: Dependency relation label to search for.

        Returns:
            True if at least one token has this deprel, False otherwise.
        """
        return any(t.deprel == deprel for t in self.tokens)

    def get_dependency_distance(self, child_index: int, head_index: int) -> int:
        """
        Compute the linear distance (in tokens) between a dependent and its head.

        Large distances (> 3) indicate long-distance dependencies that are
        challenging for both LLM extraction and rule-based analysis.

        Args:
            child_index: 1-based index of the dependent token.
            head_index:  1-based index of the head token.

        Returns:
            Absolute difference in token indices. Returns -1 if either token
            is not found.
        """
        child = self.get_token(child_index)
        head = self.get_token(head_index)
        if child is None or head is None:
            return -1
        return abs(child.index - head.index)
