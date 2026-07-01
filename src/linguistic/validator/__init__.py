"""
Validator package — modular steps for the 7-step constraint validation algorithm.

Each step is extracted into its own module for maintainability.
The ConstraintValidator class in validator.py remains a thin orchestrator
that delegates to these standalone functions.
"""

from src.linguistic.validator.validator import ConstraintValidator
from src.linguistic.validator._depth import compute_depth_metrics

__all__ = [
    "ConstraintValidator",
    "compute_depth_metrics",
]
