"""
Internal conversion: Stanza Word -> LexSpec Token.

This is the bridge between the Stanza API and the LexSpec data model.
All Stanza-specific details are encapsulated here.
"""

from __future__ import annotations

from typing import List, Dict

from src.extraction.schema import Token
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _convert_sentence_to_tokens(self, sentence) -> List[Token]:
    """Convert a Stanza Sentence into a list of LexSpec Token objects.

    This is the bridge between the Stanza API and the LexSpec data model.
    All Stanza-specific details (word.to_dict(), word.feats parsing) are
    encapsulated here so that no other module needs to import Stanza.

    UD Conventions handled:
      - word.id: May be a tuple for multi-word tokens (e.g., (1, '-'))
                 or an integer for regular tokens. We skip MWT placeholders.
      - word.head: 0-based in Stanza, converted to 1-based for LexSpec.
                   head=0 in LexSpec means "this is the root".
      - word.feats: String like "Tense=Past|Voice=Pass" parsed into dict.

    Args:
        sentence: A Stanza Sentence object containing Word objects.

    Returns:
        List of Token objects with complete UD annotations.
    """
    tokens: List[Token] = []

    for word in sentence.words:
        # Stanza represents multi-word tokens (MWT) with tuple IDs
        # like (1, '-') for the MWT head and (1, 1), (1, 2) for the
        # expanded words. We skip the MWT head and keep the expanded
        # words, which carry the actual syntactic annotations.
        if isinstance(word.id, tuple):
            # word.id is (head_index, sub_index)
            # If sub_index is 0 or invalid, skip (MWT placeholder)
            if len(word.id) == 2 and word.id[1] == 0:
                continue

            # Use the first component as the token index.
            # In Stanza, for expanded MWT tokens, the first element
            # of the tuple is the sentence-level token position.
            token_index = word.id[0] if len(word.id) > 0 else 0
        else:
            token_index = int(word.id)

        # Stanza uses 0-based head indexing internally, but UD CoNLL-U
        # format (and our DependencyTree model) uses 1-based indexing
        # where 0 means "root". Convert accordingly:
        #   Stanza head=0 -> LexSpec head=0 (root)
        #   Stanza head=1 -> LexSpec head=1 (first token)
        head_index = int(word.head) if word.head is not None else 0

        # Parse morphological features from Stanza's string format.
        # Stanza represents feats as "Tense=Past|Voice=Pass" or None.
        feats: Dict[str, str] = {}
        if word.feats:
            for feat_str in word.feats.split("|"):
                if "=" in feat_str:
                    key, val = feat_str.split("=", 1)
                    feats[key.strip()] = val.strip()

        token = Token(
            index=token_index,
            text=word.text,
            lemma=word.lemma if word.lemma else word.text.lower(),
            upos=word.upos if word.upos else "X",
            xpos=word.xpos if word.xpos else "",
            deprel=word.deprel if word.deprel else "dep",
            head=head_index,
            feats=feats,
        )
        tokens.append(token)

    return tokens
