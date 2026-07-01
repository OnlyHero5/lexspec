"""
Wilcoxon signed-rank test as a supplementary significance test.

Classic non-parametric paired test. Less interpretable than bootstrap
CIs but widely recognized in the NLP literature.

Theory reference:
  - Dror et al. (2018). The Hitchhiker's Guide to Testing Statistical
    Significance in Natural Language Processing. ACL 2018.
"""

from __future__ import annotations

from typing import List, Dict, Any

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


def wilcoxon_test(
    scores_a: List[float],
    scores_b: List[float],
) -> Dict[str, Any]:
    """Wilcoxon signed-rank test as a supplementary significance test.

    This is a non-parametric paired difference test. Unlike the bootstrap,
    it makes the assumption that the distribution of differences is symmetric
    (which is usually reasonable for per-sample F1 scores).

    The test ranks the absolute differences between paired scores, sums
    the ranks for positive and negative differences separately, and computes
    a test statistic from the smaller rank sum.

    Interpretation:
      - p_value < 0.05: reject the null hypothesis that the median difference
        is zero. Conclude that the two systems have significantly different
        performance.
      - The sign of the statistic indicates direction: positive means more
        pairs favor B > A.

    Note: This is a TWO-SIDED test (different from bootstrap which is one-sided
    by default). For consistency in reporting, consider both results together.

    Args:
        scores_a: Per-sample scores for system A (baseline).
        scores_b: Per-sample scores for system B (treatment).

    Returns:
        Dict with keys:
        - statistic: float — the Wilcoxon W statistic (the smaller of the
          two rank sums).
        - p_value: float — two-sided p-value.
        - significant_at_0.05: bool — True if p_value < 0.05.
        - n_pairs: int — number of paired samples.
        - median_diff: float — median of the differences (B - A).
        - method: str — "wilcoxon_signed_rank" for identification.

    Raises:
        ValueError: If scores have different lengths.
        ImportError: If scipy is not installed.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"scores_a and scores_b must have the same length. "
            f"Got {len(scores_a)} and {len(scores_b)}."
        )

    n = len(scores_a)
    if n == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_at_0.05": False,
            "n_pairs": 0,
            "median_diff": 0.0,
            "method": "wilcoxon_signed_rank",
        }

    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon
    except ImportError:
        raise ImportError(
            "scipy is required for the Wilcoxon test. "
            "Install it with: pip install scipy"
        )

    # scipy's wilcoxon performs the paired signed-rank test.
    # It tests H0: the distribution of (x - y) is symmetric about zero.
    # By default it computes a two-sided p-value.
    # We use method='approx' for faster computation (exact is infeasible for n > 25).
    result = scipy_wilcoxon(scores_b, scores_a, zero_method="wilcox", method="approx")

    statistic = float(result.statistic)
    p_value = float(result.pvalue)

    # Compute median of differences for additional context.
    diffs = np.array(scores_b) - np.array(scores_a)
    median_diff = float(np.median(diffs))

    significant = p_value < 0.05

    logger.info(
        "Wilcoxon: n=%d, W=%.4f, p=%.4f, median_diff=%.4f, significant=%s",
        n, statistic, p_value, median_diff, significant,
    )

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant_at_0.05": significant,
        "n_pairs": n,
        "median_diff": median_diff,
        "method": "wilcoxon_signed_rank",
    }
