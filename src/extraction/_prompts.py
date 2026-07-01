"""
提示词加载 —— 从 YAML 配置文件加载提示词模板
============================================

从 ``configs/prompts.yaml`` 加载系统提示词和用户提示词模板。
配置缺失或格式错误时直接抛出异常，禁止静默回退。
"""

from __future__ import annotations

from typing import Dict
from pathlib import Path

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_prompts(prompts_path: str) -> Dict[str, str]:
    """Load prompt templates from a YAML configuration file.

    Looks for the keys ``extraction.baseline.system`` and
    ``extraction.baseline.user`` in the YAML document.

    Args:
        prompts_path: Path to the ``prompts.yaml`` file.

    Returns:
        A dict with keys ``"system"``, ``"user"``, and ``"source"``.

    Raises:
        FileNotFoundError: If the prompts file does not exist.
        yaml.YAMLError: If the YAML is malformed.
        KeyError: If the expected ``extraction.baseline.system`` or
                  ``extraction.baseline.user`` keys are missing.
    """
    prompts_file = Path(prompts_path)

    if not prompts_file.exists():
        raise FileNotFoundError(
            f"Prompts config file not found: '{prompts_file}'. "
            "Create it from configs/prompts.yaml or check your working directory."
        )

    if not prompts_file.is_file():
        raise FileNotFoundError(
            f"Prompts path '{prompts_file}' is not a regular file."
        )

    with open(prompts_file, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if config is None:
        raise ValueError(
            f"Prompts config '{prompts_file}' is empty (null YAML document)."
        )

    extraction = config.get("extraction")
    if extraction is None:
        raise KeyError(
            f"No 'extraction' section found in '{prompts_file}'. "
            "The prompts YAML must contain an 'extraction' top-level key."
        )

    baseline = extraction.get("baseline")
    if baseline is None:
        raise KeyError(
            f"No 'extraction.baseline' section found in '{prompts_file}'. "
            "The prompts YAML must contain 'extraction.baseline.system' and "
            "'extraction.baseline.user' keys."
        )

    system_prompt = baseline.get("system")
    user_prompt = baseline.get("user")

    if not system_prompt or not user_prompt:
        raise KeyError(
            f"Incomplete 'extraction.baseline' in '{prompts_file}': "
            "both 'system' and 'user' keys are required and must be non-empty."
        )

    system_prompt = system_prompt.strip()
    user_prompt = user_prompt.strip()

    logger.info("Loaded prompts from '%s' (extraction.baseline)", prompts_file)
    return {
        "system": system_prompt,
        "user": user_prompt,
        "source": str(prompts_file),
    }
