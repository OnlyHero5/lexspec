"""
错误报告：分布统计、摘要生成与标签辅助函数。

提供 error_distribution_report()、generate_error_summary()，
以及将枚举值映射为可读标签的工具函数。
"""

from __future__ import annotations

import time
import random
from typing import List, Dict, Any, Tuple

from src.extraction.schema import ErrorCase
from src.utils.logging import get_logger

logger = get_logger(__name__)


def error_distribution_report(error_cases: List[ErrorCase]) -> Dict[str, Any]:
    """根据已分类错误案例生成分布统计。

    计算：
      - 主类别分布（语言学现象频次）
      - 次类别分布（字段错误类型频次）
      - 交叉表（主 × 次联合频次）
      - 最常见错误模式（按频次排序）

    供评估报告使用的定量错误类型分解。

    参数:
        error_cases: classify_errors() 返回的 ErrorCase 列表。

    返回:
        字典，键包括：
        - total_errors: int — 错误案例总数。
        - primary_distribution: Dict[str, int] — 各主类别计数。
        - secondary_distribution: Dict[str, int] — 各次类别计数。
        - cross_tabulation: Dict[str, Dict[str, int]] — 联合分布。
        - most_common_patterns: List[Tuple[str, int]] — 按频次排序的 top 模式。

        error_cases 为空时返回零/空集合。
    """
    from collections import Counter, defaultdict

    total = len(error_cases)
    if total == 0:
        return {
            "total_errors": 0,
            "primary_distribution": {},
            "secondary_distribution": {},
            "cross_tabulation": {},
            "most_common_patterns": [],
        }

    # 分别统计主、次类别。
    primary_counter: Counter = Counter()
    secondary_counter: Counter = Counter()

    # 交叉表：主 -> 次 -> 计数
    cross_tab: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for case in error_cases:
        prim = case.primary_category.value
        sec = case.secondary_category.value

        primary_counter[prim] += 1
        secondary_counter[sec] += 1
        cross_tab[prim][sec] += 1

    # 将交叉表展平为可排序模式。
    patterns: List[Tuple[str, int]] = []
    for prim, sec_counts in cross_tab.items():
        for sec, count in sec_counts.items():
            patterns.append((f"{prim} + {sec}", count))
    patterns.sort(key=lambda x: x[1], reverse=True)

    # 将 defaultdict 转为普通 dict 以便序列化。
    cross_tab_serializable = {
        prim: dict(sec_counts) for prim, sec_counts in cross_tab.items()
    }

    logger.info(
        "Error distribution: %d errors, %d primary categories, %d secondary categories",
        total, len(primary_counter), len(secondary_counter),
    )

    return {
        "total_errors": total,
        "primary_distribution": dict(primary_counter),
        "secondary_distribution": dict(secondary_counter),
        "cross_tabulation": cross_tab_serializable,
        "most_common_patterns": patterns[:10],  # 前 10 个模式。
    }


def generate_error_summary(error_cases: List[ErrorCase]) -> str:
    """生成人类可读的错误摘要字符串。

    输出格式化的类 Markdown 字符串，适合写入评估报告。
    含总体错误统计、主/次类别分布及代表性示例。

    参数:
        error_cases: classify_errors() 返回的 ErrorCase 列表。

    返回:
        多行格式化的错误分析摘要字符串。
    """
    if not error_cases:
        return "## Error Analysis Summary\n\nNo errors found. All predictions match the gold standard.\n"

    dist = error_distribution_report(error_cases)

    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("ERROR ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    # 总体统计。
    lines.append(f"Total errors: {dist['total_errors']}")
    lines.append("")

    # 主类别分布。
    lines.append("-" * 50)
    lines.append("Primary Category Distribution (Linguistic Phenomenon)")
    lines.append("-" * 50)
    for cat, count in sorted(dist["primary_distribution"].items(),
                              key=lambda x: x[1], reverse=True):
        pct = count / dist["total_errors"] * 100
        cat_label = error_category_label(cat)
        lines.append(f"  {cat_label:40s} {count:5d} ({pct:5.1f}%)")
    lines.append("")

    # 次类别分布。
    lines.append("-" * 50)
    lines.append("Secondary Category Distribution (Field Error Type)")
    lines.append("-" * 50)
    for cat, count in sorted(dist["secondary_distribution"].items(),
                              key=lambda x: x[1], reverse=True):
        pct = count / dist["total_errors"] * 100
        cat_label = field_error_label(cat)
        lines.append(f"  {cat_label:40s} {count:5d} ({pct:5.1f}%)")
    lines.append("")

    # 最常见模式。
    lines.append("-" * 50)
    lines.append("Most Common Error Patterns (Primary + Secondary)")
    lines.append("-" * 50)
    for pattern, count in dist["most_common_patterns"]:
        pct = count / dist["total_errors"] * 100
        lines.append(f"  {pattern:50s} {count:5d} ({pct:5.1f}%)")
    lines.append("")

    # 代表性示例（前 3 个错误案例）。
    lines.append("-" * 50)
    lines.append("Representative Error Examples")
    lines.append("-" * 50)
    for i, case in enumerate(error_cases[:3]):
        lines.append(f"\nExample {i + 1} (ID: {case.error_id})")
        lines.append(f"  Primary:   {case.primary_category.value}")
        lines.append(f"  Secondary: {case.secondary_category.value}")
        # 展示解释前两行（标题 + 首条细节）。
        expl_lines = case.linguistic_explanation.split("\n")
        for line in expl_lines[:3]:
            if line.strip():
                lines.append(f"  {line[:120]}")
        lines.append(f"  Prediction: subj={case.prediction.subject.text}, "
                      f"pred={case.prediction.action.predicate}, "
                      f"obj={case.prediction.action.object}")
        lines.append(f"  Gold:       subj={case.gold.subject.text}, "
                      f"pred={case.gold.action.predicate}, "
                      f"obj={case.gold.action.object}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF ERROR ANALYSIS")
    lines.append("=" * 70)

    return "\n".join(lines)


def error_category_label(cat_value: str) -> str:
    """将 ErrorCategory 枚举值映射为可读标签。"""
    labels = {
        "passive_voice": "Passive Voice",
        "conditional_boundary": "Conditional Boundary",
        "relative_clause": "Relative Clause",
        "long_distance_dependency": "Long-Distance Dependency",
        "negation_exception": "Negation/Exception",
        "other": "Other",
    }
    return labels.get(cat_value, cat_value)


def field_error_label(cat_value: str) -> str:
    """将 FieldErrorType 枚举值映射为可读标签。"""
    labels = {
        "subject": "Subject Error",
        "role": "Role Error",
        "predicate": "Predicate Error",
        "object": "Object Error",
        "condition_omission": "Condition Omission",
        "condition_overextension": "Condition Over-extension",
    }
    return labels.get(cat_value, cat_value)


def generate_error_id() -> str:
    """生成唯一错误标识符。

    使用简单计数思路。生产环境可换为 UUID 以支持分布式。

    返回:
        形如 "E-0001" 的字符串。
    """
    # 简单计数：单次运行内用时间戳后缀保证合理唯一性。
    # 非全局唯一，但对单次运行的错误分析足够。
    ts = int(time.time() * 1000) % 100000
    rnd = random.randint(0, 999)
    return f"E-{ts:05d}-{rnd:03d}"
