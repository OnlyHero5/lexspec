"""
Text Matching and Normalization Utilities
==========================================
Self-contained utilities for comparing LLM-extracted text against UD tokens.
These are extracted from the ConstraintValidator to keep that module focused
on the core validation algorithm.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.extraction.schema import Token

from src.utils.logging import get_logger

logger = get_logger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for comparison.

    Steps applied in order:
    1. Lowercase the entire string.
    2. Remove leading articles: "the Seller" -> "seller",
       "a Party" -> "party", "an Agreement" -> "agreement".
    3. Remove trailing punctuation: periods, commas, semicolons.
    4. Collapse multiple whitespace characters to single space.
    5. Strip leading/trailing whitespace.

    This normalization ensures that superficial differences in
    capitalization, articles, and punctuation do not cause false
    mismatch detections. The goal is to compare semantic content,
    not surface form.

    Args:
        text: Raw text string from LLM or UD.

    Returns:
        Normalized text suitable for comparison.
    """
    if not text:
        return ""

    normalized = text.lower().strip()

    # Remove leading articles ("the", "a", "an").
    # We use word-boundary check to avoid removing "the" from
    # the middle of words (e.g., "theory").
    normalized = re.sub(
        r'^\s*(the|a|an)\s+', '', normalized, count=1
    )

    # Remove trailing punctuation.
    normalized = normalized.rstrip(".,;:!?\"'()[]{}")

    # Collapse whitespace.
    normalized = re.sub(r'\s+', ' ', normalized)

    # Final strip.
    normalized = normalized.strip()

    return normalized


def match_text(llm_text: str, ud_token: "Token") -> bool:
    """Check if LLM-extracted text matches a UD token.

    Matching is performed after normalization (lowercase, strip
    articles, strip punctuation). We check:

    1. Exact match after normalization of both sides.
    2. Substring match: the UD token text is a substring of the
       LLM text (handles coordination expansions like "Buyer and
       Seller" matching UD "Buyer") or vice versa.
    3. Token overlap: both strings share at least one content word
       (noun, verb, adjective). This catches cases where the LLM
       extracted a longer NP but the head noun matches the UD token.

    Args:
        llm_text: Text from LLM extraction.
        ud_token: Token from UD parse.

    Returns:
        True if texts match after normalization.
    """
    llm_norm = normalize_text(llm_text)
    ud_norm = normalize_text(ud_token.text)

    # Exact match after normalization.
    if llm_norm == ud_norm:
        return True

    # Substring match (handles coordination and NP expansion).
    if ud_norm in llm_norm or llm_norm in ud_norm:
        logger.debug(
            "Substring match: LLM='%s' contains UD='%s'",
            llm_norm, ud_norm,
        )
        return True

    # Token overlap: split both into word sets and check for
    # at least one common content word.
    llm_words = set(llm_norm.split())
    ud_words = set(ud_norm.split())

    # Remove function words from consideration.
    function_words = {
        "the", "a", "an", "of", "in", "to", "for", "on", "at",
        "by", "with", "from", "and", "or", "not", "no", "any",
        "all", "each", "every", "such", "its", "his", "her",
    }
    llm_content = llm_words - function_words
    ud_content = ud_words - function_words

    common = llm_content & ud_content
    if common:
        logger.debug(
            "Content-word overlap match: %s", common,
        )
        return True

    return False
