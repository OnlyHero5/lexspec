"""
Multi-experiment comparison and stratified significance testing.

Provides run_all_comparisons() for pairwise comparisons across experiments
and stratified_significance() for phenomenon-specific analysis.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

from src.evaluation.bootstrap import paired_bootstrap
from src.evaluation.wilcoxon import wilcoxon_test
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_all_comparisons(
    experiment_results: Dict[str, List[float]],
    experiment_names: Optional[List[str]] = None,
    n_resamples: int = 10000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Run all pairwise significance tests between experiments.

    For a set of experiments (e.g., baseline, constrained, reflexion),
    this function runs both bootstrap and Wilcoxon tests for every pair,
    producing a comprehensive comparison matrix.

    The results are structured for easy inclusion in evaluation reports
    and tables. Each comparison is keyed by "A_vs_B" where A and B are
    the experiment names.

    Args:
        experiment_results: Dict mapping experiment name to list of per-sample
                            scores. e.g., {"baseline": [0.3, 0.4, ...],
                                          "constrained": [0.5, 0.6, ...]}
        experiment_names: Optional ordered list of experiment names. If None,
                          uses sorted keys from experiment_results.
        n_resamples: Number of bootstrap resamples per comparison.
        random_seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
        - comparisons: Dict[str, Dict] — per-pair comparison results.
          Each value contains "bootstrap" and "wilcoxon" sub-dicts.
        - summary: Dict with a matrix view of significance results.
        - n_pairs: int — number of paired samples per experiment.

    Raises:
        ValueError: If fewer than 2 experiments are provided, or if scores
                    have different lengths across experiments.
    """
    if len(experiment_results) < 2:
        raise ValueError(
            f"Need at least 2 experiments for comparison. Got {len(experiment_results)}."
        )

    # Determine experiment order.
    names = experiment_names if experiment_names else sorted(experiment_results.keys())

    # Validate that all experiments have the same number of samples.
    n_pairs = None
    for name in names:
        if name not in experiment_results:
            raise ValueError(f"Experiment '{name}' not found in experiment_results.")
        n = len(experiment_results[name])
        if n_pairs is None:
            n_pairs = n
        elif n != n_pairs:
            raise ValueError(
                f"All experiments must have the same number of samples. "
                f"'{name}' has {n}, expected {n_pairs}."
            )

    comparisons: Dict[str, Dict] = {}

    # Run all pairwise comparisons.
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_a = names[i]
            name_b = names[j]
            key = f"{name_a}_vs_{name_b}"

            logger.info("Comparing %s vs %s...", name_a, name_b)

            scores_a = experiment_results[name_a]
            scores_b = experiment_results[name_b]

            try:
                bootstrap_result = paired_bootstrap(
                    scores_a, scores_b, n_resamples=n_resamples, random_seed=random_seed,
                )
            except Exception as e:
                logger.error("Bootstrap failed for %s: %s", key, e)
                bootstrap_result = {"error": str(e)}

            try:
                wilcoxon_result = wilcoxon_test(scores_a, scores_b)
            except Exception as e:
                logger.error("Wilcoxon failed for %s: %s", key, e)
                wilcoxon_result = {"error": str(e)}

            comparisons[key] = {
                "experiment_a": name_a,
                "experiment_b": name_b,
                "bootstrap": bootstrap_result,
                "wilcoxon": wilcoxon_result,
            }

    # Build a significance matrix for the summary.
    # Rows and columns are experiment names; cells are significance indicators.
    sig_matrix: Dict[str, Dict[str, str]] = {}
    for name in names:
        sig_matrix[name] = {}
        for other in names:
            if name == other:
                sig_matrix[name][other] = "-"
            else:
                # Find the comparison key (order-independent).
                key1 = f"{name}_vs_{other}"
                key2 = f"{other}_vs_{name}"
                comp = comparisons.get(key1) or comparisons.get(key2)
                if comp and "bootstrap" in comp and not isinstance(comp["bootstrap"].get("significant_at_0.05"), str):
                    bs = comp["bootstrap"]
                    if bs.get("significant_at_0.05", False):
                        # Determine direction: does the second experiment win?
                        mean_diff = bs.get("mean_diff", 0)
                        if key1 in comparisons:
                            direction = ">" if mean_diff > 0 else "<"
                        else:
                            direction = ">" if mean_diff < 0 else "<"
                        sig_matrix[name][other] = f"sig({direction})"
                    else:
                        sig_matrix[name][other] = "ns"
                else:
                    sig_matrix[name][other] = "?"

    summary = {
        "n_experiments": len(names),
        "experiment_names": names,
        "n_pairs_per_experiment": n_pairs,
        "significance_matrix": sig_matrix,
    }

    return {
        "comparisons": comparisons,
        "summary": summary,
    }


def stratified_significance(
    scores_a: List[float],
    scores_b: List[float],
    phenomenon_labels: List[str],
    phenomenon_name: str,
    n_resamples: int = 10000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Run significance test on a specific linguistic phenomenon subset.

    This allows fine-grained comparison: "Does system B outperform system A
    specifically on passive voice constructions?" or "Is the improvement
    limited to long-distance dependencies?"

    The function filters the per-sample scores to only those samples that
    exhibit the specified linguistic phenomenon, then runs the paired
    bootstrap test on the filtered subset.

    Args:
        scores_a: Full per-sample scores for system A.
        scores_b: Full per-sample scores for system B.
        phenomenon_labels: Per-sample labels indicating which phenomenon
                           each sample exhibits. Must be the same length
                           as scores_a and scores_b. Labels are typically
                           from ErrorCategory enum values or custom tags.
        phenomenon_name: Which phenomenon value to filter by. Only samples
                         with this label are included in the test.
        n_resamples: Number of bootstrap resamples.
        random_seed: Random seed for reproducibility.

    Returns:
        Dict with the bootstrap result for the filtered subset, plus:
        - phenomenon: str — the phenomenon name.
        - subset_size: int — number of samples with this phenomenon.
        - total_size: int — total number of samples.
        - subset_ratio: float — proportion of samples with this phenomenon.

    Raises:
        ValueError: If input lengths don't match or if no samples match
                    the specified phenomenon.
    """
    n = len(scores_a)
    if len(scores_b) != n or len(phenomenon_labels) != n:
        raise ValueError(
            f"All input lists must have the same length. "
            f"Got scores_a={len(scores_a)}, scores_b={len(scores_b)}, "
            f"labels={len(phenomenon_labels)}."
        )

    # Filter to the specified phenomenon.
    filtered_a = []
    filtered_b = []
    for sa, sb, label in zip(scores_a, scores_b, phenomenon_labels):
        if label == phenomenon_name:
            filtered_a.append(sa)
            filtered_b.append(sb)

    subset_size = len(filtered_a)
    if subset_size == 0:
        raise ValueError(
            f"No samples found with phenomenon='{phenomenon_name}'. "
            f"Available labels: {sorted(set(phenomenon_labels))}"
        )

    # Run bootstrap on the filtered subset.
    bootstrap_result = paired_bootstrap(
        filtered_a, filtered_b,
        n_resamples=n_resamples,
        random_seed=random_seed,
    )

    result = {
        "phenomenon": phenomenon_name,
        "subset_size": subset_size,
        "total_size": n,
        "subset_ratio": subset_size / n if n > 0 else 0.0,
    }
    result.update(bootstrap_result)

    logger.info(
        "Stratified significance for '%s': n=%d/%d (%.1f%%), "
        "mean_diff=%.4f, p=%.4f, sig=%s",
        phenomenon_name, subset_size, n, result["subset_ratio"] * 100,
        result["mean_diff"], result["p_value"], result["significant_at_0.05"],
    )

    return result
