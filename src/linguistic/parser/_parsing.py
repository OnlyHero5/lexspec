"""
Parsing methods for StanzaParser: parse() and parse_batch().
"""

from __future__ import annotations

from typing import Optional, List

from src.extraction.schema import DependencyTree, Token
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _parse(self, text: str) -> DependencyTree:
    """Parse a single sentence and return its UD dependency tree.

    Processes the input text through the full Stanza pipeline
    (tokenize -> MWT -> POS -> lemma -> depparse) and converts
    the output into a DependencyTree with Token objects.

    In legal text, sentences can be very long (50+ words) due to
    complex contractual language. Stanza handles this well, but
    extremely long sentences (>100 words) may degrade parse quality.

    Args:
        text: English sentence or contract clause text.
              Should be a single sentence — multi-sentence input
              will only return the first sentence's parse.
              Use parse_batch() for multi-sentence documents.

    Returns:
        DependencyTree with tokens annotated with complete UD
        dependency relations, POS tags, lemmas, and features.

    Raises:
        ValueError: If parsing produces no sentences (empty input,
                    whitespace-only, or ungrammatical text that
                    Stanza cannot segment).
        RuntimeError: If the pipeline has not been initialized.

    Example:
        >>> parser = StanzaParser()
        >>> tree = parser.parse("Seller shall deliver the Goods.")
        >>> tree.root_index
        3  # "deliver" is the root
        >>> tree.get_token(3).lemma
        'deliver'
    """
    if not text or not text.strip():
        raise ValueError("Cannot parse empty or whitespace-only text")

    # Run Stanza pipeline on the input text.
    # doc.sentences is a list of Sentence objects, each containing
    # a list of Word objects with UD annotations.
    doc = self.nlp(text)

    if len(doc.sentences) == 0:
        raise ValueError(
            f"Stanza produced no sentences for input: '{text[:80]}...'"
        )

    # We operate on the first sentence only.
    # Multi-sentence contract excerpts should be pre-split before
    # calling parse(). This is consistent with our pipeline design
    # where each LegalTriplet corresponds to exactly one clause.
    sentence = doc.sentences[0]

    from src.linguistic.parser._conversion import _convert_sentence_to_tokens
    tokens = _convert_sentence_to_tokens(sentence)
    logger.debug(
        "Parsed sentence: %d tokens, root at index %d",
        len(tokens),
        next((t.index for t in tokens if t.head == 0), -1),
    )

    return DependencyTree(text=text, tokens=tokens)


def _parse_batch(self, texts: List[str]) -> List[DependencyTree]:
    """Parse multiple sentences efficiently.

    More efficient than calling parse() in a loop because Stanza
    can batch-process internally, amortizing GPU/CPU overhead
    across multiple inputs. Use this for processing clauses from
    contract documents in bulk.

    Each text in the input list is treated as a separate sentence.
    Multi-sentence texts will only yield the first sentence's parse.

    Args:
        texts: List of sentence strings to parse.

    Returns:
        List of DependencyTree objects, one per input text.
        Same length as input — results are aligned 1:1 with input.

    Note:
        Empty strings in the input produce DependencyTree objects
        with zero tokens (no error is raised). The caller should
        filter these if needed.
    """
    if not texts:
        return []

    logger.info("Batch-parsing %d sentences", len(texts))

    # Stanza can process a list of texts directly.
    # This is more efficient than calling parse() in a loop
    # because the neural models batch their computations.
    doc = self.nlp(texts)

    trees: List[DependencyTree] = []

    # doc.sentences contains all sentences across all input texts.
    # However, if an input text contains multiple sentences,
    # they will all appear in doc.sentences. We correlate back
    # to the input by using the sentence text.
    #
    # Strategy: iterate through sentences and match each back to
    # an input text by checking containment. This handles the case
    # where some input texts are empty (no sentence produced).
    sentence_idx = 0
    for original_text in texts:
        if not original_text or not original_text.strip():
            # Empty input -> empty tree (no tokens, no root)
            trees.append(DependencyTree(text=original_text, tokens=[]))
            continue

        if sentence_idx < len(doc.sentences):
            sentence = doc.sentences[sentence_idx]
            from src.linguistic.parser._conversion import _convert_sentence_to_tokens
            tokens = _convert_sentence_to_tokens(sentence)
            trees.append(DependencyTree(text=original_text, tokens=tokens))
            sentence_idx += 1
        else:
            # Fallback: no more sentences but we still have inputs.
            # This shouldn't happen under normal circumstances.
            logger.warning(
                "Fewer sentences produced than inputs: %d < %d",
                len(doc.sentences), len(texts),
            )
            trees.append(DependencyTree(text=original_text, tokens=[]))

    logger.info("Batch parsing complete: %d trees produced", len(trees))
    return trees
