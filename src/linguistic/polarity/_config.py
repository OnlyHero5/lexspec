"""
Configuration loading for PolarityDetector: YAML parsing and lookup building.

Loads modality rules from constraints YAML.
No hardcoded fallback — a missing or malformed config is an error.
"""

from __future__ import annotations

from typing import Dict
from pathlib import Path

import yaml

from src.extraction.schema import LegalRole
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_rules(self, constraints_path: str) -> None:
    """Load modality rules from YAML config.

    Parses the modality_rules section which defines:
      obligor:        {aux_verbs: [shall, must, will], negated: false}
      right_holder:   {aux_verbs: [may, can], negated: false}
      prohibited_party: {aux_verbs: [shall, may, must], negated: true}

    Builds two internal structures:
      1. _modality_rules: Full rules dict for introspection
      2. _lookup: Fast (aux_lemma, is_negated) -> LegalRole mapping

    Args:
        constraints_path: Path to constraints YAML file.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
        KeyError: If ``modality_rules`` is missing or incomplete.
    """
    config_path = Path(constraints_path)
    if not config_path.is_absolute():
        resolved = config_path.resolve()
        if not resolved.exists():
            alt_paths = [
                Path.cwd() / "configs" / "constraints.yaml",
                Path(__file__).parent.parent.parent.parent / "configs" / "constraints.yaml",
            ]
            for alt in alt_paths:
                if alt.exists():
                    config_path = alt
                    break

    if not config_path.exists():
        raise FileNotFoundError(
            f"Constraints config not found at '{constraints_path}'. "
            "Create it from configs/constraints.yaml or check your working directory."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    if not config_data or "modality_rules" not in config_data:
        raise KeyError(
            f"No 'modality_rules' section in '{config_path}'. "
            "The constraints YAML must define obligor / right_holder / prohibited_party rules."
        )

    self._modality_rules = config_data["modality_rules"]

    required_roles = ("obligor", "right_holder", "prohibited_party")
    for role_key in required_roles:
        if role_key not in self._modality_rules:
            raise KeyError(
                f"Required modality rule '{role_key}' not found in '{config_path}'. "
                "All three roles (obligor, right_holder, prohibited_party) are required."
            )
        rule = self._modality_rules[role_key]
        if not rule.get("aux_verbs"):
            raise ValueError(
                f"Modality rule '{role_key}' in '{config_path}' has empty aux_verbs list."
            )

    logger.info("Loaded modality rules from %s", config_path)

    _build_lookup(self)


def _build_lookup(self) -> None:
    """Build the (aux_lemma, is_negated) -> LegalRole lookup table.

    Maps each combination of auxiliary verb and negation status
    to the corresponding legal role for O(1) classification.
    """
    role_map = {
        "obligor": LegalRole.OBLIGOR,
        "right_holder": LegalRole.RIGHT_HOLDER,
        "prohibited_party": LegalRole.PROHIBITED_PARTY,
    }

    self._lookup.clear()

    for role_key, role_enum in role_map.items():
        rule = self._modality_rules[role_key]
        aux_verbs = rule.get("aux_verbs", [])
        negated = rule.get("negated", False)

        for aux in aux_verbs:
            aux_lower = aux.lower().strip()
            if aux_lower:
                self._lookup[(aux_lower, negated)] = role_enum

    logger.debug(
        "Built modality lookup: %d (aux, negated) -> role mappings",
        len(self._lookup),
    )
