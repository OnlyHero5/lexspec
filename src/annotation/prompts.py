"""
Annotation Prompt Templates
============================
Prompt loading for the LLM annotation pipeline.

All prompts must come from configs/prompts.yaml.
No hardcoded fallbacks — a missing or incomplete config is an error.

Exported:
  - load_annotation_prompts: Load prompts from YAML config
"""

from __future__ import annotations

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_annotation_prompts(prompts_path: str) -> tuple[str, str]:
    """Load annotation prompt templates from a YAML configuration file.

    Reads ``annotation.system`` and ``annotation.user`` from the YAML.
    No fallback — raises on any error.

    Args:
        prompts_path: Path to the prompts YAML configuration file.

    Returns:
        (system_prompt, user_template) tuple.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML is malformed.
        KeyError: If annotation.system or annotation.user is missing.
    """
    with open(prompts_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise TypeError(f"Prompts config '{prompts_path}' is not a valid YAML mapping.")

    annotation_section = config.get("annotation")
    if not isinstance(annotation_section, dict):
        raise KeyError(
            f"No 'annotation' section in '{prompts_path}'. "
            "The prompts YAML must contain 'annotation.system' and 'annotation.user'."
        )

    system_prompt = annotation_section.get("system")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise KeyError(
            f"Missing or empty 'annotation.system' in '{prompts_path}'."
        )
    system_prompt = system_prompt.strip()

    user_template = annotation_section.get("user")
    if not isinstance(user_template, str) or not user_template.strip():
        raise KeyError(
            f"Missing or empty 'annotation.user' in '{prompts_path}'."
        )
    user_template = user_template.strip()

    if "{dependency_info}" in user_template:
        user_template = user_template.replace("{dependency_info}", "")
        logger.debug("Removed {dependency_info} placeholder from annotation user template")

    logger.info("Loaded annotation prompts from %s", prompts_path)
    return system_prompt, user_template
