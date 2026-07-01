"""
实验比较的统计显著性检验。

方法:
  - 配对 bootstrap 重采样（主方法）— bootstrap.py
  - Wilcoxon 符号秩检验（补充）— wilcoxon.py
  - 多实验比较 + 分层检验 — comparisons.py

本模块为再导出门面，保留原有导入路径。
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
