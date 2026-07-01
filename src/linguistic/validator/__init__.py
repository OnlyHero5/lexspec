"""
校验器包 —— 七步约束校验算法的模块化步骤。

各步骤提取为独立模块以便维护。
validator.py 中的 ConstraintValidator 类保持为薄编排器，
委托给这些独立函数。
"""

from src.linguistic.validator.validator import ConstraintValidator
from src.linguistic.validator._depth import compute_depth_metrics

__all__ = [
    "ConstraintValidator",
    "compute_depth_metrics",
]
