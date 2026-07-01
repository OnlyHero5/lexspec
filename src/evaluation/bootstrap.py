"""
Paired bootstrap resampling for significance testing.

Non-parametric, makes no distributional assumptions, directly estimates
the sampling distribution of the mean difference via resampling.
Recommended by ACL/EMNLP guidelines for NLP evaluation.

Theory references:
  - Efron & Tibshirani (1993). An Introduction to the Bootstrap.
  - Berg-Kirkpatrick et al. (2012). An Empirical Investigation of Statistical
    Significance in NLP. EMNLP 2012.
"""

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def paired_bootstrap(
    scores_a: List[float],
    scores_b: List[float],
    n_resamples: int = 10000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap resampling for significance testing.

    Procedure (following Berg-Kirkpatrick et al. 2012):
    1. Compute per-sample difference: diff_i = score_b_i - score_a_i.
       Positive values mean system B outperforms system A.
    2. Resample n pairs WITH REPLACEMENT n_resamples times.
       Each resample draws n indices from {0, ..., n-1} uniformly with
       replacement, then computes the mean of diff at those indices.
    3. This builds the bootstrap distribution of the mean difference.
    4. Compute the 95% confidence interval from the bootstrap distribution
       using the percentile method (2.5th and 97.5th percentiles).
    5. Compute p-value (one-sided): proportion of bootstrap resamples where
       mean_diff <= 0. This tests the null hypothesis that system B is
       NOT better than system A (i.e., mean_diff <= 0).

    Interpretation:
      - If ci_95_lower > 0: system B is significantly better at alpha=0.05
        (the entire 95% CI is above zero).
      - If ci_95_upper < 0: system A is significantly better.
      - If ci_95_* straddles 0: no significant difference at alpha=0.05.
      - p_value < 0.05 confirms the finding.

    Args:
        scores_a: Per-sample scores for system A (baseline). Must be aligned
                  1:1 with scores_b. e.g., per-sample F1 from compute_per_sample_f1().
        scores_b: Per-sample scores for system B (treatment).
        n_resamples: Number of bootstrap resamples. Default 10,000 gives
                     stable estimates. Use 1,000 for quick checks, 100,000
                     for publication-quality results.
        random_seed: Random seed for reproducibility of the bootstrap
                     distribution. Uses numpy's RandomState for isolation
                     from global numpy random state.

    Returns:
        Dict with keys:
        - mean_diff: float — mean(B - A) on the original data. Positive means
          B outperforms A.
        - ci_95_lower: float — lower bound of 95% bootstrap CI.
        - ci_95_upper: float — upper bound of 95% bootstrap CI.
        - p_value: float — one-sided bootstrap p-value. Proportion of bootstrap
          samples with mean_diff <= 0. Lower values indicate stronger evidence
          that B > A.
        - significant_at_0.05: bool — True if ci_95_lower > 0 and p_value < 0.05.
        - n_pairs: int — number of paired samples.
        - bootstrap_std: float — standard deviation of the bootstrap distribution.

    Raises:
        ValueError: If scores_a and scores_b have different lengths or
                    contain fewer than 2 pairs.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"scores_a and scores_b must have the same length. "
            f"Got {len(scores_a)} and {len(scores_b)}."
        )

    n = len(scores_a)
    if n < 2:
        raise ValueError(
            f"Need at least 2 paired samples for bootstrap. Got {n}."
        )

    # Convert to numpy arrays for efficient vectorized operations.
    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)

    # Compute per-sample differences on the original data.
    diffs = b - a  # Positive = B better than A.
    mean_diff_original = float(np.mean(diffs))

    # Initialize reproducible random state.
    rng = np.random.RandomState(random_seed)

    # Bootstrap: resample indices with replacement and compute mean diff.
    # This is vectorized for performance: generate all indices at once
    # and compute means in a single operation.
    # Shape: (n_resamples, n)
    resample_indices = rng.randint(0, n, size=(n_resamples, n))
    # For each resample, take the diffs at those indices and compute mean.
    # Shape: (n_resamples,)
    bootstrap_means = np.mean(diffs[resample_indices], axis=1)

    # Compute 95% confidence interval using percentile method.
    # np.percentile uses linear interpolation between data points.
    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))

    # One-sided p-value: proportion of bootstrap samples where mean_diff <= 0.
    # This tests H0: mean(B - A) <= 0 vs H1: mean(B - A) > 0.
    p_value = float(np.mean(bootstrap_means <= 0))

    # Significance determination: both p-value and CI must agree.
    # p < 0.05 AND entire CI above zero.
    significant = (p_value < 0.05) and (ci_lower > 0)

    bootstrap_std = float(np.std(bootstrap_means, ddof=1))

    logger.info(
        "Bootstrap: n=%d, resamples=%d, mean_diff=%.4f, "
        "95%%CI=[%.4f, %.4f], p=%.4f, significant=%s",
        n, n_resamples, mean_diff_original, ci_lower, ci_upper, p_value, significant,
    )

    return {
        "mean_diff": mean_diff_original,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "p_value": p_value,
        "significant_at_0.05": significant,
        "n_pairs": n,
        "n_resamples": n_resamples,
        "bootstrap_std": bootstrap_std,
    }
