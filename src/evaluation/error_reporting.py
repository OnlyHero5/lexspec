"""
Error reporting: per-experiment error analysis and cross-experiment comparison.

Provides functions to analyze errors for a single experiment and print
comparison tables across all experiments.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.extraction.schema import LegalTriplet, DependencyTree, ErrorCase
from src.evaluation.data_loading import load_predictions_as_triplets
from src.evaluation.error_analyzer import (
    classify_errors, save_error_cases,
)
from src.evaluation.error_summary import (
    error_distribution_report, generate_error_summary,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def analyze_experiment_errors(
    experiment_name: str,
    predictions: List[Dict],
    gold: List[LegalTriplet],
    trees: List[DependencyTree],
    output_dir: str,
) -> Dict[str, Any]:
    """Run full error analysis for one experiment.

    Steps:
      1. Convert predictions to LegalTriplet objects.
      2. Run ``classify_errors()`` to generate ErrorCase objects.
      3. Compute error distribution statistics.
      4. Save categorized error cases to JSONL files.
      5. Return a summary dict for the comparison table.

    Args:
        experiment_name:  e.g., "baseline", "ours_dep", "ours_reflexion".
        predictions:      List of prediction dicts from the experiment.
        gold:             List of gold-standard LegalTriplets.
        trees:            List of DependencyTree objects (same length).
        output_dir:       Directory for error case files.

    Returns:
        Dict with error distribution statistics and summary.
    """
    logger.info("Analyzing errors for experiment: %s", experiment_name)

    # Convert predictions.
    pred_triplets = load_predictions_as_triplets(predictions)

    # Align lengths.
    n = min(len(pred_triplets), len(gold), len(trees))
    if n == 0:
        logger.warning("No data for %s -- skipping error analysis", experiment_name)
        return {
            "experiment": experiment_name,
            "total_samples": 0,
            "error_count": 0,
            "error_rate": 0.0,
            "primary_distribution": {},
            "secondary_distribution": {},
        }
    pred_triplets = pred_triplets[:n]
    gold = gold[:n]
    trees_used = trees[:n]

    # Classify errors.
    error_cases: List[ErrorCase] = classify_errors(
        predictions=pred_triplets,
        gold=gold,
        trees=trees_used,
    )

    # Compute distribution.
    dist = error_distribution_report(error_cases)

    # Save categorized error cases.
    exp_error_dir = Path(output_dir) / experiment_name
    save_error_cases(error_cases, str(exp_error_dir))

    # Generate and print summary.
    summary_text = generate_error_summary(error_cases)
    print(f"\n{summary_text}")

    # Build return dict.
    return {
        "experiment": experiment_name,
        "total_samples": n,
        "error_count": len(error_cases),
        "error_rate": len(error_cases) / n if n > 0 else 0.0,
        "primary_distribution": dist.get("primary_distribution", {}),
        "secondary_distribution": dist.get("secondary_distribution", {}),
        "cross_tabulation": dist.get("cross_tabulation", {}),
        "most_common_patterns": dist.get("most_common_patterns", []),
    }


def print_error_comparison_table(
    results: Dict[str, Dict[str, Any]],
) -> None:
    """Print a comparison table of error statistics across experiments.

    Shows error count, error rate, and primary category distribution
    side-by-side for quick comparison.

    Args:
        results:  Dict mapping experiment name -> error analysis summary dict.
    """
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS COMPARISON -- ALL EXPERIMENTS")
    print("=" * 80)

    # Per-experiment overall stats.
    print(f"\n{'Experiment':<20s} {'Samples':>8s} {'Errors':>8s} {'ErrorRate':>10s}")
    print("-" * 46)
    for name in ["baseline", "ours_dep", "ours_reflexion"]:
        r = results.get(name, {})
        if not r:
            continue
        print(
            f"{name:<20s} "
            f"{r.get('total_samples', 0):>8d} "
            f"{r.get('error_count', 0):>8d} "
            f"{r.get('error_rate', 0):>9.1%}"
        )

    # Primary category distribution.
    print("\n--- Primary Category Distribution (Linguistic Phenomenon) ---")
    all_categories = sorted(set(
        cat for r in results.values()
        for cat in r.get("primary_distribution", {}).keys()
    ))
    header = f"{'Experiment':<20s}"
    for cat in all_categories:
        header += f" {cat:>18s}"
    print(header)
    print("-" * (20 + 20 * len(all_categories)))
    for name in ["baseline", "ours_dep", "ours_reflexion"]:
        r = results.get(name, {})
        if not r:
            continue
        line = f"{name:<20s}"
        for cat in all_categories:
            cnt = r.get("primary_distribution", {}).get(cat, 0)
            line += f" {cnt:>18d}"
        print(line)

    # Secondary category distribution.
    print("\n--- Secondary Category Distribution (Field Error Type) ---")
    all_fields = sorted(set(
        f for r in results.values()
        for f in r.get("secondary_distribution", {}).keys()
    ))
    header = f"{'Experiment':<20s}"
    for f in all_fields:
        header += f" {f:>18s}"
    print(header)
    print("-" * (20 + 20 * len(all_fields)))
    for name in ["baseline", "ours_dep", "ours_reflexion"]:
        r = results.get(name, {})
        if not r:
            continue
        line = f"{name:<20s}"
        for f in all_fields:
            cnt = r.get("secondary_distribution", {}).get(f, 0)
            line += f" {cnt:>18d}"
        print(line)

    print("=" * 80)
