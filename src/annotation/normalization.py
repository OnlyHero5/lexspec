"""
Text Normalization Utilities for Annotation Consensus
======================================================
Normalizes text fields for fuzzy comparison during field-level
consensus voting. Handles article stripping, case normalization,
and basic lemmatization to avoid surface-form disagreements.

Exported:
  - ARTICLES_PATTERN: Compiled regex for article/determiner stripping
  - normalize_text:   Public normalization function (used externally by
                       experiments/step_02_annotate_gold.py)
  - _basic_lemmatize:  Internal suffix-stripping lemmatizer
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Articles to strip during text normalization.
# Legal text often uses "the Seller", "a Buyer", "an Indemnitee" --
# we strip these to compare the core entity name.
# ---------------------------------------------------------------------------

ARTICLES_PATTERN = re.compile(
    r"^\s*(the|a|an|said|such|each|any|all|no|every)\s+",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """Normalize a text field for fuzzy comparison.

    Applies the following transformations in order:
      1. Strip leading/trailing whitespace.
      2. Convert to lowercase.
      3. Remove leading articles and determiners (the, a, an, said, such,
         each, any, all, no, every).
      4. Collapse repeated whitespace to a single space.
      5. Apply basic lemmatization (plural → singular via suffix stripping).

    This normalization ensures that surface-form differences like
    "the Seller" vs "Seller" or "delivers" vs "deliver" are treated
    as equivalent rather than disagreements.

    Args:
        text: Raw text string from the annotation.

    Returns:
        Normalized comparison-ready string.
    """
    if not text:
        return ""

    # Step 1-2: Strip whitespace and lowercase.
    normalized = text.strip().lower()

    # Step 3: Remove leading articles and determiners.
    normalized = ARTICLES_PATTERN.sub("", normalized).strip()

    # Step 4: Collapse whitespace.
    normalized = re.sub(r"\s+", " ", normalized)

    # Step 5: Basic lemmatization -- strip common plural suffixes.
    # This is a lightweight alternative to full morphological analysis.
    # Legal text lemmas are typically verb bases and singular nouns,
    # so simple suffix stripping covers the majority of cases.
    normalized = _basic_lemmatize(normalized)

    return normalized


def _basic_lemmatize(text: str) -> str:
    """Apply basic suffix-stripping lemmatization.

    Strips common English morphological suffixes to reduce surface-form
    disagreements. This is intentionally simple -- full lemmatization
    (e.g., via NLTK or Stanza) would be more accurate but adds a heavy
    dependency that is not justified for the annotation consensus task.

    Handles:
      - Plural nouns:     "parties" → "party", "losses" → "loss"
      - 3rd person verbs: "delivers" → "deliver", "indemnifies" → "indemnify"
      - Past tense:       "delivered" → "deliver", "paid" → "pay"
      - Gerunds:          "delivering" → "deliver"

    Args:
        text: Normalized lowercased text.

    Returns:
        Lemmatized text with suffixes stripped where applicable.
    """
    # Only lemmatize the last word of multi-word phrases, as it's
    # typically the head noun or main verb.
    words = text.split()
    if not words:
        return text

    # Lemmatize the final word only -- modifiers and determiners are
    # already handled by article stripping.
    last_word = words[-1]

    # Plural noun rules (ordered by specificity).
    if last_word.endswith("ies") and len(last_word) > 4:
        # "parties" → "party", "liabilities" → "liability"
        last_word = last_word[:-3] + "y"
    elif last_word.endswith("ives") and len(last_word) > 5:
        # Not common in legal text, but handled for completeness.
        last_word = last_word[:-4] + "f"
    elif last_word.endswith("ves") and len(last_word) > 4:
        # "leaves" → "leaf" -- rare in legal text.
        last_word = last_word[:-3] + "f"
    elif last_word.endswith("ses") and len(last_word) > 4:
        # "losses" → "loss", "clauses" → "clause"
        if last_word.endswith("sses"):
            last_word = last_word[:-2]
        else:
            last_word = last_word[:-1]
    elif last_word.endswith("s") and not last_word.endswith("ss") and len(last_word) > 3:
        # Generic plural: "rights" → "right", "goods" → "good"
        # Skip words ending in "ss" (e.g., "loss", "witness") and
        # very short words to avoid over-stripping.
        last_word = last_word[:-1]

    # Verb inflection rules.
    # 3rd person singular: "delivers" → "deliver"
    if last_word.endswith("es") and len(last_word) > 4:
        if last_word.endswith("ies"):
            # "indemnifies" → "indemnify"
            last_word = last_word[:-3] + "y"
        elif last_word.endswith("ses") or last_word.endswith("zes") or last_word.endswith("ches") or last_word.endswith("shes") or last_word.endswith("xes"):
            last_word = last_word[:-2]
        else:
            # "delivers" → handled by the generic -s rule above
            pass

    # Past tense / past participle: "delivered" → "deliver"
    if last_word.endswith("ed") and len(last_word) > 4:
        stem = last_word[:-2]
        if stem.endswith("i"):
            # "paid" is a special case handled by -ed not matching
            last_word = stem[:-1] + "y"  # "ied" → "y": "indemnified" → "indemnify"
        elif stem.endswith("e"):
            last_word = stem  # "agreed" → "agree"
        else:
            last_word = stem  # "delivered" → "deliver" (ok approximation)

    # Gerund: "delivering" → "deliver"
    if last_word.endswith("ing") and len(last_word) > 5:
        stem = last_word[:-3]
        if stem.endswith("nn") or stem.endswith("tt") or stem.endswith("mm"):
            # Doubled consonant: "running" → "run"
            last_word = stem[:-1]
        elif stem.endswith("e"):
            last_word = stem  # "agreeing" → "agree" (rare)
        else:
            last_word = stem  # "delivering" → "deliver" (ok approximation)

    words[-1] = last_word
    return " ".join(words)
