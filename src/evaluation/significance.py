"""
Statistical significance testing for experiment comparisons.

Methods:
  - Paired bootstrap resampling (primary method) — bootstrap.py
  - Wilcoxon signed-rank test (supplementary) — wilcoxon.py
  - Multi-experiment comparison + stratified testing — comparisons.py

This module is a re-export facade that preserves the original import paths.
"""

from src.evaluation.bootstrap import paired_bootstrap
from src.evaluation.wilcoxon import wilcoxon_test
from src.evaluation.comparisons import run_all_comparisons, stratified_significance

__all__ = [
    "paired_bootstrap",
    "wilcoxon_test",
    "run_all_comparisons",
    "stratified_significance",
]
