"""
Validation thresholds loader — reads condition_overlap, subject_match, object_match
from constraints YAML. No defaults — a missing config is an error.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_validation_thresholds(
    constraints_path: str = "configs/constraints.yaml",
) -> tuple:
    """Load validation thresholds from constraints YAML.

    Reads the ``validation`` section:
      - condition_overlap: Minimum IoU for condition span match
      - subject_match: Minimum token overlap for subject match
      - object_match: Minimum token overlap for object match

    Args:
        constraints_path: Path to constraints YAML file.

    Returns:
        (condition_overlap, subject_match, object_match) tuple of floats.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
        KeyError: If the ``validation`` section or its keys are missing.
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
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Constraints config '{config_path}' is empty.")

    validation_cfg = config.get("validation")
    if not validation_cfg:
        raise KeyError(
            f"No 'validation' section in '{config_path}'. "
            "The constraints YAML must define condition_overlap, subject_match, and object_match."
        )

    condition_overlap = validation_cfg.get("condition_overlap")
    subject_match = validation_cfg.get("subject_match")
    object_match = validation_cfg.get("object_match")

    if condition_overlap is None or subject_match is None or object_match is None:
        raise KeyError(
            f"Incomplete 'validation' section in '{config_path}'. "
            "All three keys (condition_overlap, subject_match, object_match) are required."
        )

    logger.info(
        "Validation thresholds loaded: condition_overlap=%.2f, "
        "subject_match=%.2f, object_match=%.2f",
        condition_overlap, subject_match, object_match,
    )

    return (condition_overlap, subject_match, object_match)
