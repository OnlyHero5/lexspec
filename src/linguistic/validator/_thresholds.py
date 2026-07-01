"""
校验阈值加载器 —— 从约束 YAML 读取全部 validation.* 键。
"""

from __future__ import annotations

from src.utils.constraints import get_validation_thresholds, load_constraints_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_validation_thresholds(
    constraints_path: str = "configs/constraints.yaml",
) -> tuple:
    """为校验器加载 (condition_overlap, subject_match, object_match)。"""
    config = load_constraints_config(constraints_path)
    thresholds = get_validation_thresholds(config, constraints_path)
    condition_overlap = thresholds["condition_overlap"]
    subject_match = thresholds["subject_match"]
    object_match = thresholds["object_match"]
    logger.info(
        "Validation thresholds loaded: condition_overlap=%.2f, "
        "subject_match=%.2f, object_match=%.2f",
        condition_overlap, subject_match, object_match,
    )
    return (condition_overlap, subject_match, object_match)
