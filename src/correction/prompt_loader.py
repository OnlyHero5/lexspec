"""
Reflexion Prompt Loader
=======================
Loads the Reflexion feedback prompt template and system prompt from
YAML configuration. No hardcoded fallbacks — a missing config is an error.

Exported:
  - load_reflexion_prompt:    Load feedback template from YAML config
  - load_system_prompt:       Load extraction system prompt from YAML config
"""

from __future__ import annotations

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_reflexion_prompt(prompts_path: str) -> str:
    """Load the reflexion feedback prompt template from YAML config.

    Reads ``reflexion.feedback_template`` from the YAML configuration file.
    Adapts the YAML template's placeholders to the standard format used
    by ReflexionGenerator.generate_feedback().

    Args:
        prompts_path: Path to the prompts YAML configuration file.

    Returns:
        The feedback prompt template string with standardized placeholders.

    Raises:
        FileNotFoundError: If the prompts file does not exist.
        yaml.YAMLError: If the YAML is malformed.
        KeyError: If the reflexion section or feedback_template is missing.
    """
    with open(prompts_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise TypeError(f"Prompts config '{prompts_path}' is not a valid YAML mapping.")

    reflexion_section = config.get("reflexion")
    if not isinstance(reflexion_section, dict):
        raise KeyError(
            f"No 'reflexion' section in '{prompts_path}'. "
            "The prompts YAML must contain a 'reflexion' top-level key."
        )

    template = reflexion_section.get("feedback_template")
    if not template or not isinstance(template, str):
        raise KeyError(
            f"No 'reflexion.feedback_template' in '{prompts_path}'. "
            "The YAML must contain a non-empty feedback_template string."
        )

    logger.info("Loaded reflexion prompt template from %s", prompts_path)
    return template


def load_system_prompt(prompts_path: str) -> str:
    """Load the extraction system prompt for re-use during Reflexion.

    Reads ``extraction.baseline.system`` from the YAML config.

    Args:
        prompts_path: Path to the prompts YAML configuration file.

    Returns:
        The system prompt string for re-extraction LLM calls.

    Raises:
        FileNotFoundError: If the prompts file does not exist.
        KeyError: If the extraction section is missing.
    """
    with open(prompts_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise TypeError(f"Prompts config '{prompts_path}' is not a valid YAML mapping.")

    extraction = config.get("extraction")
    if not isinstance(extraction, dict):
        raise KeyError(
            f"No 'extraction' section in '{prompts_path}'. "
            "Cannot load system prompt for Reflexion."
        )

    baseline = extraction.get("baseline")
    if not isinstance(baseline, dict):
        raise KeyError(
            f"No 'extraction.baseline' section in '{prompts_path}'."
        )

    system_prompt = baseline.get("system")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise KeyError(
            f"No non-empty 'extraction.baseline.system' in '{prompts_path}'."
        )

    logger.info("Loaded extraction system prompt from %s", prompts_path)
    return system_prompt.strip()
