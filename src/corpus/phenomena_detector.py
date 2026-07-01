"""
Language phenomenon detection on UD dependency parse trees.

Each detection rule is based on UD v2 dependency relations, as described in
the LexSpec design document sections 4.2-4.6.
"""

from __future__ import annotations

import re
from typing import Dict

from src.linguistic.stanza_parser import StanzaParser
from src.linguistic.ud_features import (
    UDFeatureExtractor,
    compute_mean_dependency_distance,
)
from src.linguistic.passive_detector import PassiveDetector
from src.extraction.schema import DependencyTree

# ---------------------------------------------------------------------------
# Non-substantive text filtering
# ---------------------------------------------------------------------------

# Template text to exclude from sentence segmentation results
# (headers, signatures, table of contents, etc.).
_BOILERPLATE_RE = re.compile(
    r"^(page\s+\d+|exhibit\s+[a-z0-9]+|schedule\s+[a-z0-9]+|"
    r"table of contents|signature page|in witness whereof|"
    r"cc:\s|very truly yours)",
    re.IGNORECASE,
)

# Columns to skip when loading CUAD spans.
_SKIP_COLUMNS = frozenset({
    "Filename", "Document Name", "Document Name-Answer",
    "Parties", "Parties-Answer",
})


# ======================================================================
# Language phenomenon detection
# ======================================================================


def detect_phenomena(tree: DependencyTree) -> Dict[str, bool]:
    """Detect language phenomena in a dependency parse tree.

    Detection logic:
      - passive:         any verb has nsubj:pass or aux:pass relation
      - conditional:     dep-level advcl + mark combination, or
                         text-level lexical marker fallback
      - relative_clause: acl:relcl relation present
      - long_distance:   mean dependency distance exceeds threshold (6.0)
      - negation:        neg relation present
      - is_definition:   clause is a term definition ("X means Y" pattern)

    Args:
        tree: DependencyTree parsed from a single contract clause.

    Returns:
        Dict with boolean flags for each phenomenon.
    """
    phenomena: Dict[str, bool] = {}

    # --- Passive voice ---
    passive_detected = False
    for token in tree.find_tokens_by_upos("VERB"):
        if PassiveDetector.is_passive_loose(tree, token.index):
            passive_detected = True
            break
    phenomena["passive"] = passive_detected

    # --- Conditional clause: dependency detection (advcl + mark) ---
    has_advcl = tree.has_deprel("advcl")
    has_mark = tree.has_deprel("mark")
    phenomena["conditional"] = has_advcl and has_mark

    # --- Conditional clause: lexical fallback detection ---
    # Some conditional clauses (especially those introduced by
    # "if"/"unless"/"subject to") may not be parsed as advcl+mark
    # by Stanza. Use lexical markers in text as fallback.
    if not phenomena["conditional"]:
        text_lower = tree.text.lower()
        lexical_conditionals = [
            " if ", " unless ", " provided that ", " so long as ",
            " subject to ", " in the event that ", " in the event of ",
            " on condition that ", " conditioned upon ",
            " except ", " except as ", " other than ",
            " upon ", " within ", " after ", " before ",
        ]
        for marker in lexical_conditionals:
            if marker in text_lower:
                phenomena["conditional"] = True
                break

    # --- Relative clause ---
    phenomena["relative_clause"] = tree.has_deprel("acl:relcl")

    # --- Long-distance dependency (mean dependency distance > 6.0) ---
    # Legal texts have naturally high MDD due to complex noun phrase
    # structures. Threshold raised from 4.0 to 6.0 to capture genuine
    # long-distance dependencies rather than normal legal syntactic
    # complexity.
    mdd = compute_mean_dependency_distance(tree)
    phenomena["long_distance"] = mdd > 6.0

    # --- Negation ---
    phenomena["negation"] = tree.has_deprel("neg")

    # --- Definition clause detection ---
    phenomena["is_definition"] = _is_definition_clause(tree)

    return phenomena


def _is_definition_clause(tree: DependencyTree) -> bool:
    """Detect whether a clause is a term definition rather than an operative clause.

    Recognizes patterns:
      - "X means Y" / "X shall mean Y"
      - '"Term" means ...' (quoted term + means)
      - '1.4 "Term" means ...' (numbered + quoted term + definition)

    Args:
        tree: DependencyTree parsed from a single contract clause.

    Returns:
        True if the clause is a definition.
    """
    text = tree.text.strip()
    text_lower = text.lower()

    # Pattern 1: "X means Y" -- subject is a quoted term or short noun phrase.
    if re.search(r'\bmeans\b', text_lower):
        if text.startswith('"') or text[0].isdigit():
            return True
        if re.search(r'["\']?\w+["\']?\s+(?:shall\s+)?means?\b', text_lower):
            return True

    # Pattern 2: Numbered + quoted term start (e.g. "1.4 'Term' means...").
    if re.match(r'^\d+\.\d+\s+["\']', text):
        return True

    return False


def is_boilerplate_clause(text: str) -> bool:
    """Heuristic filter for non-operative text fragments."""
    t = text.strip()
    if len(t) < 15:
        return True
    if _BOILERPLATE_RE.match(t):
        return True
    words = t.split()
    if len(words) <= 8 and t.isupper():
        return True
    return False
