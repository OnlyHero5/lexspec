"""
Triplet-level normalization and party alias loading.

This module provides:
  - load_party_aliases(): YAML config loading for party alias mappings.
  - normalize_triplet(): Normalize all text fields in a LegalTriplet.

The core text normalization pipeline (normalize(), NUMBER_WORDS, etc.) is
defined in text_normalizer.py. This module re-exports those symbols so that
existing imports like ``from src.evaluation.normalization import normalize``
continue to work unchanged.
"""

from __future__ import annotations

from typing import Optional, Dict, List

from src.extraction.schema import LegalTriplet
from src.evaluation.text_normalizer import normalize, NUMBER_WORDS  # re-export
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_party_aliases(constraints_path: str = "configs/constraints.yaml") -> Dict[str, List[str]]:
    """Load party alias mappings from the constraints configuration file.

    The constraints.yaml file may contain a 'party_aliases' section that
    defines canonical party names and their variants. If no such section
    exists, returns an empty dict.

    Typical usage:
        aliases = load_party_aliases()
        normalized = normalize("the Seller", party_aliases=aliases)

    Args:
        constraints_path: Path to the constraints YAML configuration file.
                          Defaults to "configs/constraints.yaml" relative to
                          the project root.

    Returns:
        Dict mapping canonical party names to lists of alias strings.
        e.g., {"Seller": ["the Seller", "Seller", "Company", "Vendor"],
               "Buyer": ["the Buyer", "Buyer", "Purchaser"]}

        Returns an empty dict if the file is not found or the section is absent
        (treated as non-fatal since party aliases are an optional feature).
    """
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed; cannot load party aliases from %s", constraints_path)
        return {}

    from pathlib import Path
    path = Path(constraints_path)
    if not path.is_file():
        logger.debug("Constraints file not found at %s; no party aliases loaded", constraints_path)
        return {}

    try:
        with open(path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
    except Exception as e:
        logger.warning("Failed to parse constraints config %s: %s", constraints_path, e)
        return {}

    if not isinstance(config, dict):
        return {}

    # Look for a top-level 'party_aliases' section.
    aliases_raw = config.get("party_aliases", None)
    if aliases_raw is None:
        # Also check within the 'normalization' section (nested path).
        normalization = config.get("normalization", {})
        aliases_raw = normalization.get("party_aliases", None) if isinstance(normalization, dict) else None

    if aliases_raw is None or not isinstance(aliases_raw, dict):
        logger.debug("No party_aliases section found in %s", constraints_path)
        return {}

    # Validate structure: canonical → list of alias strings.
    result: Dict[str, List[str]] = {}
    for canonical, aliases in aliases_raw.items():
        if isinstance(aliases, list):
            result[str(canonical)] = [str(a) for a in aliases]
        elif isinstance(aliases, str):
            result[str(canonical)] = [str(aliases)]
        else:
            logger.debug("Skipping invalid alias entry for canonical='%s': %s", canonical, aliases)

    logger.info("Loaded %d party alias mappings from %s", len(result), constraints_path)
    return result


def normalize_triplet(
    triplet: LegalTriplet,
    party_aliases: Optional[Dict[str, List[str]]] = None,
) -> LegalTriplet:
    """Normalize all text fields in a LegalTriplet for fair comparison.

    Returns a NEW triplet (does not modify the input). This is critical
    because evaluation functions need both the original and normalized
    versions — the original for error reporting, the normalized for
    comparison.

    Fields normalized:
      - subject.text:  Full normalization (articles, punctuation, numbers, aliases)
      - subject.role:  Preserved as-is (enum values are already canonical)
      - action.predicate: Normalize text (articles removed, lowercase, etc.)
      - action.object: Normalize text
      - condition.text: Normalize text; condition.type preserved as-is

    Args:
        triplet: The original LegalTriplet to normalize.
        party_aliases: Optional party alias mappings for entity normalization.

    Returns:
        A new LegalTriplet with all text fields normalized. The original
        triplet is not modified.
    """
    from src.extraction.schema import Subject, Action, Condition, LegalRole, ConditionType

    # Normalize subject text — keep the role unchanged.
    normalized_subject = Subject(
        text=normalize(triplet.subject.text, party_aliases=party_aliases),
        role=triplet.subject.role,
    )

    # Normalize action fields — predicate and object text.
    normalized_action = Action(
        predicate=normalize(triplet.action.predicate, party_aliases=party_aliases),
        object=normalize(triplet.action.object, party_aliases=party_aliases),
    )

    # Normalize condition text — keep the condition type unchanged.
    normalized_condition = Condition(
        text=normalize(triplet.condition.text, party_aliases=party_aliases),
        type=triplet.condition.type,
    )

    return LegalTriplet(
        subject=normalized_subject,
        action=normalized_action,
        condition=normalized_condition,
    )
