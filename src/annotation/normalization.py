"""
标注共识用的文本规范化工具
============================
通过 normalize_for_comparison() 使用 constraints.yaml 中的设置。
"""

from __future__ import annotations

from src.utils.constraints import normalize_for_comparison

# 为与 step_02 导入保持向后兼容而保留。
ARTICLES_PATTERN = None  # 已弃用 — 规范化规则见 constraints.yaml


def normalize_text(
    text: str,
    constraints_path: str = "configs/constraints.yaml",
) -> str:
    """在共识投票期间对文本字段进行规范化，用于模糊比较。"""
    return normalize_for_comparison(text, constraints_path)
