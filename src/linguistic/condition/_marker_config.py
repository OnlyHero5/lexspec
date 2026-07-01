"""
Marker configuration: YAML loading and marker section parsing.

Loads the condition marker taxonomy from constraints YAML.
No hardcoded fallback — a missing or malformed config is an error.
"""

from __future__ import annotations

from typing import Dict, List
from pathlib import Path

import yaml

from src.extraction.schema import ConditionType
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_markers(self, constraints_path: str) -> None:
    """Load condition marker taxonomy from YAML config.

    Parses the condition_markers section of constraints.yaml.
    Each category (trigger, temporal, exception) has ``mark_words`` lists.

    All markers are lowercased and indexed in a flat dict for
    O(1) lookup during extraction.

    Args:
        constraints_path: Path to the YAML config file.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
        KeyError: If ``condition_markers`` is missing or empty.
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

    if not config_data or "condition_markers" not in config_data:
        raise KeyError(
            f"No 'condition_markers' section in '{config_path}'. "
            "The constraints YAML must define trigger/temporal/exception marker lists."
        )

    markers_section = config_data["condition_markers"]
    self._markers = _parse_markers_section(markers_section)

    if not self._markers:
        raise ValueError(
            f"Condition marker taxonomy is empty in '{config_path}'. "
            "At least one marker must be defined per category."
        )

    self._marker_list = list(self._markers.keys())

    logger.info(
        "Condition marker taxonomy loaded: %d markers total "
        "(trigger=%d, temporal=%d, exception=%d)",
        len(self._markers),
        sum(1 for v in self._markers.values() if v == ConditionType.TRIGGER),
        sum(1 for v in self._markers.values() if v == ConditionType.TEMPORAL),
        sum(1 for v in self._markers.values() if v == ConditionType.EXCEPTION),
    )


def _parse_markers_section(section: dict) -> Dict[str, ConditionType]:
    """Parse a condition_markers YAML section into a lookup dict.

    Args:
        section: The condition_markers dict from YAML.

    Returns:
        Dict mapping lowercase marker text -> ConditionType.

    Raises:
        KeyError: If a required category is missing.
    """
    result: Dict[str, ConditionType] = {}

    category_map = {
        "trigger": ConditionType.TRIGGER,
        "temporal": ConditionType.TEMPORAL,
        "exception": ConditionType.EXCEPTION,
    }

    for category_name, condition_type in category_map.items():
        if category_name not in section:
            raise KeyError(
                f"Category '{category_name}' not found in condition_markers config. "
                "All three categories (trigger, temporal, exception) are required."
            )

        category_data = section[category_name]

        if isinstance(category_data, dict):
            words = category_data.get("mark_words", [])
        elif isinstance(category_data, list):
            words = category_data
        else:
            raise TypeError(
                f"Unexpected format for category '{category_name}': "
                f"expected dict or list, got {type(category_data).__name__}."
            )

        for word in words:
            word_lower = word.lower().strip()
            if word_lower and word_lower not in result:
                result[word_lower] = condition_type

    return result
