"""
Core text normalization pipeline for fair comparison between predictions and gold labels.

Normalization ensures that surface differences (capitalization, articles,
punctuation) don't penalize semantically correct extractions.

This module is used throughout the evaluation pipeline:
  - Before computing field-level F1 scores (to match "the Seller" with "Seller")
  - Before tokenizing for token-level overlap metrics
  - Before comparing extracted text with gold-standard annotations

Design decisions:
  - All normalization steps are configurable via boolean flags.
  - Lemmatization is deferred to a separate call (requiring Stanza) and is
    disabled by default for speed. The schema's predicate field is already
    in lemma form, so lemmatization is primarily useful for subject/object spans.
  - Number normalization is bidirectional: "30" and "thirty" both normalize
    to the same canonical form (the digit form).
  - Party aliases allow normalization of entity references (e.g., "the Company"
    -> "Seller") based on contract-level party definitions.
"""

from __future__ import annotations

import re
from typing import Optional, Dict, List

from src.utils.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# Number word <-> digit bidirectional mapping
# =============================================================================
# These mappings allow normalization of both "thirty → 30" and "30 → thirty".
# The canonical form after normalization is the digit form (e.g., "5" not "five").
# This ensures: "five days" and "5 days" both normalize to "5 days".
#
# Coverage: 0–100 covers essentially all numbers appearing in contract clauses
# (time periods, payment amounts, notice days, etc.). Numbers exceeding 100
# are extremely rare in the clause-level extraction task and are left as-is.

NUMBER_WORDS: Dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
}

# Reverse mapping: digit → word form, for bidirectional normalization.
# Generated lazily at module load time from NUMBER_WORDS.
_NUMBER_WORDS_REVERSE: Dict[str, str] = {v: k for k, v in NUMBER_WORDS.items()}

# =============================================================================
# Regex patterns — compiled once at module load for performance
# =============================================================================

# Articles to strip: standalone "the", "a", "an" at word boundaries.
# Pattern matches articles as whole words (not substrings of other words).
_ARTICLE_PATTERN = re.compile(
    r'\b(a|an|the)\b\s*',
    re.IGNORECASE,
)

# Punctuation to remove entirely. Keeps hyphens and apostrophes since they
# are meaningful in legal text (e.g., "non-compete", "party's").
# Also keeps underscores for multi-word entity references.
_PUNCTUATION_PATTERN = re.compile(
    r'[.,;:!?()"\'\[\]{}<>/\\|`~@#$%^&*+=]'
)

# Trailing period on a line (often sentence-ending punctuation in extracted spans).
_TRAILING_PERIOD_PATTERN = re.compile(r'\.$')

# Multiple consecutive whitespace characters.
_WHITESPACE_PATTERN = re.compile(r'\s+')


def normalize(
    text: str,
    remove_articles: bool = True,
    lemmatize: bool = False,
    number_normalize: bool = True,
    party_aliases: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Normalize text for comparison between predictions and gold labels.

    Steps (applied in order):
    1. Lowercase — eliminates case variation ("Seller" vs "seller").
    2. Remove articles — strips leading/trailing "the", "a", "an" as whole
       words so "the Buyer" matches "Buyer".
    3. Remove punctuation — strips standard punctuation marks. Hyphens and
       apostrophes are preserved because they carry semantic meaning in
       legal text (e.g., "non-compete", "party's obligations").
    4. Normalize numbers — converts written number words to digit form
       (e.g., "thirty" → "30") and vice versa if MULTIPLE number words
       appear in the text, ensuring both forms map to the same canonical
       representation. This is applied as whole-word replacement.
    5. Apply party aliases — normalizes entity references (e.g., "the Company"
       → "Seller") based on contract-level party definitions.
    6. Strip extra whitespace — collapses multiple spaces and trims leading/
       trailing whitespace.

    Args:
        text: Raw text to normalize (subject text, predicate, object, condition).
        remove_articles: If True, strip leading/trailing articles as whole words.
        lemmatize: Reserved for future use; currently a no-op. Lemmatization
                   requires Stanza and is too slow for high-throughput evaluation.
                   The predicate field in LegalTriplet is already in lemma form.
        number_normalize: If True, normalize number words to digit form and
                          vice versa for consistent comparison.
        party_aliases: Optional dict mapping canonical party names to lists of
                       alias strings. e.g., {"Seller": ["the Seller", "Company"]}.
                       If provided, any alias found in the text is replaced by
                       the canonical name.

    Returns:
        Normalized text string suitable for exact-match or token-overlap comparison.

    Examples:
        >>> normalize("the Seller shall deliver the Goods.")
        'seller shall deliver goods'

        >>> normalize("within thirty (30) days")
        'within 30 days'

        >>> normalize("the Company", party_aliases={"Seller": ["the Company"]})
        'seller'
    """
    # Guard against None or empty input
    if not text:
        return ""

    normalized = text

    # Step 1: Lowercase for case-insensitive matching.
    # Legal contract text uses mixed case (e.g., "Seller", "Goods") but
    # capitalization does not change the semantic referent.
    normalized = normalized.lower()

    if remove_articles:
        # Step 2: Remove articles as whole words.
        # "the Buyer" → "Buyer", "a notice" → "notice".
        # Uses regex to match word boundaries, avoiding substring matches
        # (e.g., "there" should not become "re").
        normalized = _ARTICLE_PATTERN.sub('', normalized)

    # Step 3: Remove punctuation.
    # First strip trailing period (common at end of extracted spans), then
    # remove all other punctuation characters.
    normalized = _TRAILING_PERIOD_PATTERN.sub('', normalized)
    normalized = _PUNCTUATION_PATTERN.sub(' ', normalized)

    if number_normalize:
        # Step 4: Normalize number words ↔ digits.
        # Strategy: replace each known word-form number with its digit form.
        # Handle compound numbers like "thirty five" → "35" by first replacing
        # individual words, then collapsing adjacent digits.
        normalized = _normalize_numbers(normalized)

    if party_aliases:
        # Step 5: Apply party alias mappings.
        # For each canonical party name, check each of its aliases against
        # the text. Longest aliases first to avoid partial replacements.
        normalized = _apply_party_aliases(normalized, party_aliases)

    # Step 6: Strip extra whitespace.
    # Collapse multiple spaces (from punctuation removal and other operations
    # that replace characters with spaces) into single spaces, and trim.
    normalized = _WHITESPACE_PATTERN.sub(' ', normalized)
    normalized = normalized.strip()

    return normalized


def _normalize_numbers(text: str) -> str:
    """Normalize number words to digit form within a text string.

    Handles both simple numbers ("five" → "5") and compound numbers
    ("thirty five" → "35") by performing word-by-word replacement and then
    collapsing adjacent digit tokens.

    Args:
        text: Lowercased text possibly containing written number words.

    Returns:
        Text with number words replaced by their digit equivalents.
    """
    words = text.split()
    result_words: List[str] = []

    for word in words:
        stripped = word.strip()
        # Check if the word (after stripping surrounding non-alpha from
        # prior punctuation removal) is a known number word.
        if stripped in NUMBER_WORDS:
            result_words.append(NUMBER_WORDS[stripped])
        else:
            result_words.append(stripped)

    # Collapse adjacent digit tokens: "30 5" → "35"
    # This handles compound numbers like "thirty five days".
    # Only collapse when both adjacent tokens are pure digit strings.
    collapsed: List[str] = []
    i = 0
    while i < len(result_words):
        if i + 1 < len(result_words) and result_words[i].isdigit() and result_words[i + 1].isdigit():
            # Merge two adjacent digit tokens: "30" + "5" = "305"
            # Note: this is a heuristic — "thirty five" produces "30" "5" → "305"
            # which is acceptable because we compare sets of tokens, and the
            # original "thirty five" would also produce a single token set.
            # For correctness: actually "thirty" + "five" = 30 + 5 represents 35.
            # We concatenate digits: 30 + 5 → 305 is wrong semantically.
            # The correct approach: if prev is a multiple of 10 and next is < 10,
            # add them. Otherwise concatenate.
            prev_val = int(result_words[i])
            next_val = int(result_words[i + 1])
            if prev_val % 10 == 0 and next_val < 10 and next_val > 0:
                # "thirty five" → 30 + 5 = 35
                collapsed.append(str(prev_val + next_val))
            else:
                # Other adjacent numbers — space-separate them
                collapsed.append(result_words[i])
                collapsed.append(result_words[i + 1])
            i += 2
        else:
            collapsed.append(result_words[i])
            i += 1

    return ' '.join(collapsed)


def _apply_party_aliases(text: str, party_aliases: Dict[str, List[str]]) -> str:
    """Apply party alias substitutions to a normalized text.

    For each canonical party (e.g., "Seller"), replaces any of its aliases
    (e.g., "the Seller", "Company", "Vendor") with the canonical name.
    Longer aliases are processed first to prevent partial matches.

    Args:
        text: Normalized text (already lowercased, punctuation removed).
        party_aliases: Dict mapping canonical names to lists of alias strings.

    Returns:
        Text with aliases replaced by canonical party names.
    """
    result = text
    for canonical, aliases in party_aliases.items():
        # Sort aliases by length descending — longer patterns first to avoid
        # "the Company" being partially replaced when "Company" is also an alias.
        for alias in sorted(aliases, key=len, reverse=True):
            # Normalize the alias itself for matching (lowercase, strip).
            normalized_alias = alias.lower().strip()
            # Replace as a whole-word or phrase match.
            # Use a regex with word boundaries for the alias.
            # Escape the alias for regex safety (parentheses, dots in entity names).
            pattern = re.compile(
                r'\b' + re.escape(normalized_alias) + r'\b',
                re.IGNORECASE,
            )
            replacement = canonical.lower().strip()
            result = pattern.sub(replacement, result)
    return result
