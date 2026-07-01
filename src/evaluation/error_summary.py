"""
Error reporting: distribution statistics, summary generation, and label helpers.

Provides error_distribution_report(), generate_error_summary(), and utility
functions for translating enum values to human-readable labels.
"""

from __future__ import annotations

import time
import random
from typing import List, Dict, Any, Tuple

from src.extraction.schema import ErrorCase
from src.utils.logging import get_logger

logger = get_logger(__name__)


def error_distribution_report(error_cases: List[ErrorCase]) -> Dict[str, Any]:
    """Generate error distribution statistics from classified error cases.

    Computes:
      - Primary category distribution (linguistic phenomenon frequencies)
      - Secondary category distribution (field error type frequencies)
      - Cross-tabulation (primary x secondary joint frequencies)
      - Most common error patterns (ranked by frequency)

    These statistics are designed for inclusion in evaluation reports,
    providing a quantitative breakdown of error types.

    Args:
        error_cases: List of ErrorCase objects from classify_errors().

    Returns:
        Dict with keys:
        - total_errors: int — total number of error cases.
        - primary_distribution: Dict[str, int] — count per primary category.
        - secondary_distribution: Dict[str, int] — count per secondary category.
        - cross_tabulation: Dict[str, Dict[str, int]] — joint distribution.
        - most_common_patterns: List[Tuple[str, int]] — top patterns sorted by frequency.

        Returns zeros/empty collections if error_cases is empty.
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

    # Count primary and secondary categories independently.
    primary_counter: Counter = Counter()
    secondary_counter: Counter = Counter()

    # Cross-tabulation: primary -> {secondary -> count}
    cross_tab: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for case in error_cases:
        prim = case.primary_category.value
        sec = case.secondary_category.value

        primary_counter[prim] += 1
        secondary_counter[sec] += 1
        cross_tab[prim][sec] += 1

    # Flatten cross-tabulation into sortable patterns.
    patterns: List[Tuple[str, int]] = []
    for prim, sec_counts in cross_tab.items():
        for sec, count in sec_counts.items():
            patterns.append((f"{prim} + {sec}", count))
    patterns.sort(key=lambda x: x[1], reverse=True)

    # Normalize cross_tab defaultdicts to regular dicts for serialization.
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
        "most_common_patterns": patterns[:10],  # Top 10 patterns.
    }


def generate_error_summary(error_cases: List[ErrorCase]) -> str:
    """Generate a human-readable error summary string.

    Produces a formatted markdown-style string suitable for inclusion
    in the evaluation report. Includes overall error statistics, primary
    and secondary category distributions, and representative examples.

    Args:
        error_cases: List of ErrorCase objects from classify_errors().

    Returns:
        Formatted multi-line string with error analysis summary.
    """
    if not error_cases:
        return "## Error Analysis Summary\n\nNo errors found. All predictions match the gold standard.\n"

    dist = error_distribution_report(error_cases)

    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("ERROR ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    # Overall statistics.
    lines.append(f"Total errors: {dist['total_errors']}")
    lines.append("")

    # Primary category distribution.
    lines.append("-" * 50)
    lines.append("Primary Category Distribution (Linguistic Phenomenon)")
    lines.append("-" * 50)
    for cat, count in sorted(dist["primary_distribution"].items(),
                              key=lambda x: x[1], reverse=True):
        pct = count / dist["total_errors"] * 100
        cat_label = error_category_label(cat)
        lines.append(f"  {cat_label:40s} {count:5d} ({pct:5.1f}%)")
    lines.append("")

    # Secondary category distribution.
    lines.append("-" * 50)
    lines.append("Secondary Category Distribution (Field Error Type)")
    lines.append("-" * 50)
    for cat, count in sorted(dist["secondary_distribution"].items(),
                              key=lambda x: x[1], reverse=True):
        pct = count / dist["total_errors"] * 100
        cat_label = field_error_label(cat)
        lines.append(f"  {cat_label:40s} {count:5d} ({pct:5.1f}%)")
    lines.append("")

    # Most common patterns.
    lines.append("-" * 50)
    lines.append("Most Common Error Patterns (Primary + Secondary)")
    lines.append("-" * 50)
    for pattern, count in dist["most_common_patterns"]:
        pct = count / dist["total_errors"] * 100
        lines.append(f"  {pattern:50s} {count:5d} ({pct:5.1f}%)")
    lines.append("")

    # Representative examples (first 3 error cases).
    lines.append("-" * 50)
    lines.append("Representative Error Examples")
    lines.append("-" * 50)
    for i, case in enumerate(error_cases[:3]):
        lines.append(f"\nExample {i + 1} (ID: {case.error_id})")
        lines.append(f"  Primary:   {case.primary_category.value}")
        lines.append(f"  Secondary: {case.secondary_category.value}")
        # Show first 2 lines of explanation (the header + first detail line).
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
    """Map an ErrorCategory enum value to a human-readable label."""
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
    """Map a FieldErrorType enum value to a human-readable label."""
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
    """Generate a unique error identifier.

    Uses a simple counter approach. In production, this could be replaced
    with UUID-based identifiers for distributed environments.

    Returns:
        String like "E-0001".
    """
    # Simple counter: use timestamp-based suffix for reasonable uniqueness
    # within a single run. Not globally unique but sufficient for single-run
    # error analysis.
    ts = int(time.time() * 1000) % 100000
    rnd = random.randint(0, 999)
    return f"E-{ts:05d}-{rnd:03d}"
