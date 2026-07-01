"""
统一提示词加载器 —— 所有大语言模型提示词模板均来自 configs/prompts.yaml。

配置缺失或不完整时立即抛出异常；无硬编码回退提示词。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_prompts_document(prompts_path: str) -> Dict:
    path = Path(prompts_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Prompts config file not found: '{prompts_path}'. "
            "Create it from configs/prompts.yaml."
        )
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if not isinstance(config, dict) or not config:
        raise ValueError(f"Prompts config '{prompts_path}' is empty or invalid.")
    return config


def _require_string(section: Dict, key: str, full_path: str, prompts_path: str) -> str:
    value = section.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise KeyError(
        f"Missing or empty '{full_path}' in '{prompts_path}'. "
        "All prompts must be defined in configs/prompts.yaml."
    )


def load_extraction_prompts(prompts_path: str) -> Dict[str, str]:
    """加载基线抽取的系统/用户提示词。"""
    config = _load_prompts_document(prompts_path)
    extraction = config.get("extraction")
    if not isinstance(extraction, dict):
        raise KeyError(f"No 'extraction' section in '{prompts_path}'.")
    baseline = extraction.get("baseline")
    if not isinstance(baseline, dict):
        raise KeyError(f"No 'extraction.baseline' section in '{prompts_path}'.")

    system = _require_string(baseline, "system", "extraction.baseline.system", prompts_path)
    user = _require_string(baseline, "user", "extraction.baseline.user", prompts_path)
    logger.info("Loaded extraction prompts from '%s'", prompts_path)
    return {"system": system, "user": user, "source": str(Path(prompts_path).resolve())}


def load_annotation_prompts(prompts_path: str) -> Tuple[str, str]:
    """加载标注（system, user_template）提示词。"""
    config = _load_prompts_document(prompts_path)
    annotation = config.get("annotation")
    if not isinstance(annotation, dict):
        raise KeyError(f"No 'annotation' section in '{prompts_path}'.")

    system = _require_string(annotation, "system", "annotation.system", prompts_path)
    user = _require_string(annotation, "user", "annotation.user", prompts_path)
    if "{dependency_info}" in user:
        user = user.replace("{dependency_info}", "")
        logger.debug("Removed {dependency_info} placeholder from annotation user template")

    logger.info("Loaded annotation prompts from '%s'", prompts_path)
    return system, user


def load_review_prompts(prompts_path: str) -> Tuple[str, str]:
    """加载跨模型审核（system, user_template）提示词。"""
    config = _load_prompts_document(prompts_path)
    review = config.get("annotation", {}).get("review")
    if not isinstance(review, dict):
        raise KeyError(f"No 'annotation.review' section in '{prompts_path}'.")
    system = _require_string(review, "system", "annotation.review.system", prompts_path)
    user = _require_string(review, "user", "annotation.review.user", prompts_path)
    logger.info("Loaded review prompts from '%s'", prompts_path)
    return system, user


def load_reflexion_config(prompts_path: str) -> Tuple[str, str, Dict[str, str]]:
    """加载 Reflexion 反馈模板、系统提示词与错误提示。"""
    config = _load_prompts_document(prompts_path)
    reflexion = config.get("reflexion")
    if not isinstance(reflexion, dict):
        raise KeyError(f"No 'reflexion' section in '{prompts_path}'.")

    feedback = _require_string(
        reflexion, "feedback_template", "reflexion.feedback_template", prompts_path
    )
    hints = reflexion.get("error_hints")
    if not isinstance(hints, dict) or not hints:
        raise KeyError(
            f"Missing or empty 'reflexion.error_hints' in '{prompts_path}'."
        )
    if "default" not in hints:
        raise KeyError(
            f"'default' error hint missing from '{prompts_path}'. "
            "A fallback hint with key 'default' is required."
        )

    hints_clean = {
        str(key): str(value).strip()
        for key, value in hints.items()
        if isinstance(value, str) and value.strip()
    }
    missing = set(hints) - set(hints_clean)
    if missing:
        raise KeyError(
            f"Empty error hints in '{prompts_path}' for keys: {sorted(missing)}"
        )

    extraction = load_extraction_prompts(prompts_path)
    logger.info(
        "Loaded reflexion config from '%s' (%d error hints)",
        prompts_path,
        len(hints_clean),
    )
    return feedback, extraction["system"], hints_clean
