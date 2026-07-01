"""
Wilcoxon 符号秩检验 — 补充显著性检验。

经典非参数配对检验。可解释性弱于 bootstrap 置信区间，
但在 NLP 文献中广泛认可。

理论参考:
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
    """Wilcoxon 符号秩检验 — 补充显著性检验。

    非参数配对差异检验。与 bootstrap 不同，假设差值分布对称
    （对逐样本 F1 通常合理）。

    对配对得分差的绝对值排序，分别对正、负差求秩和，
    由较小秩和计算检验统计量。

    解读:
      - p_value < 0.05：拒绝差值中位数为零的原假设，
        认为两系统性能显著不同。
      - 统计量符号表示方向：为正表示更多配对 favor B > A。

    注意：默认为双侧检验（bootstrap 默认可为单侧）。
    报告时建议同时参考两种结果。

    参数:
        scores_a: 系统 A（基线）的逐样本得分。
        scores_b: 系统 B（处理）的逐样本得分。

    返回:
        字典，键包括：
        - statistic: float — Wilcoxon W 统计量（两个秩和中较小者）。
        - p_value: float — 双侧 p 值。
        - significant_at_0.05: bool — p_value < 0.05 时为 True。
        - n_pairs: int — 配对样本数。
        - median_diff: float — 差值 (B - A) 的中位数。
        - method: str — 标识为 "wilcoxon_signed_rank"。

    异常:
        ValueError: 两列得分长度不同。
        ImportError: 未安装 scipy。
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

    # scipy 的 wilcoxon 执行配对符号秩检验。
    # 检验 H0：(x - y) 的分布关于零对称。
    # 默认计算双侧 p 值。
    # n > 25 时用 method='approx' 加速（精确法不可行）。
    result = scipy_wilcoxon(scores_b, scores_a, zero_method="wilcox", method="approx")

    statistic = float(result.statistic)
    p_value = float(result.pvalue)

    # 计算差值中位数作为补充信息。
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
