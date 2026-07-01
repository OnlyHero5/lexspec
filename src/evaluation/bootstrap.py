"""
配对 bootstrap 重采样 — 显著性检验。

非参数方法，不假设分布形式，通过重采样直接估计均值差的抽样分布。
ACL/EMNLP 评估指南推荐用于 NLP。

理论参考:
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
    """配对 bootstrap 重采样 — 显著性检验。

    流程（遵循 Berg-Kirkpatrick et al. 2012）:
    1. 计算逐样本差：diff_i = score_b_i - score_a_i。
       正值表示系统 B 优于系统 A。
    2. 有放回重采样 n 对，共 n_resamples 次。
       每次从 {0, ..., n-1} 均匀有放回抽取 n 个索引，
       计算这些索引处 diff 的均值。
    3. 由此得到均值差的 bootstrap 分布。
    4. 用百分位法（2.5 与 97.5 分位）计算 95% 置信区间。
    5. 计算 p 值（单侧）：bootstrap 中 mean_diff <= 0 的比例。
       检验原假设：系统 B 不优于系统 A（即 mean_diff <= 0）。

    解读:
      - 若 ci_95_lower > 0：在 alpha=0.05 下系统 B 显著更好
        （整个 95% CI 在零以上）。
      - 若 ci_95_upper < 0：系统 A 显著更好。
      - 若 ci_95_* 跨零：alpha=0.05 下无显著差异。
      - p_value < 0.05 与上述结论一致。

    参数:
        scores_a: 系统 A（基线）逐样本得分，与 scores_b 1:1 对齐。
                  例如 compute_per_sample_f1() 的逐样本 F1。
        scores_b: 系统 B（处理）逐样本得分。
        n_resamples: bootstrap 重采样次数。默认 10000 估计较稳；
                     快速检查可用 1000，发表级可用 100000。
        random_seed: 随机种子以保证 bootstrap 分布可复现。
                     使用 numpy RandomState，与全局随机状态隔离。

    返回:
        字典，键包括：
        - mean_diff: float — 原始数据上 mean(B - A)。为正表示 B 优于 A。
        - ci_95_lower: float — 95% bootstrap CI 下界。
        - ci_95_upper: float — 95% bootstrap CI 上界。
        - p_value: float — 单侧 bootstrap p 值。mean_diff <= 0 的
          bootstrap 样本比例。越小越支持 B > A。
        - significant_at_0.05: bool — ci_95_lower > 0 且 p_value < 0.05 时为 True。
        - n_pairs: int — 配对样本数。
        - bootstrap_std: float — bootstrap 分布的标准差。

    异常:
        ValueError: scores_a 与 scores_b 长度不同，或配对少于 2 对。
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

    # 转为 numpy 数组以便向量化。
    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)

    # 在原始数据上计算逐样本差。
    diffs = b - a  # 为正表示 B 优于 A。
    mean_diff_original = float(np.mean(diffs))

    # 初始化可复现随机状态。
    rng = np.random.RandomState(random_seed)

    # Bootstrap：有放回重采样索引并计算均值差。
    # 向量化：一次生成全部索引再批量求均值。
    # 形状：(n_resamples, n)
    resample_indices = rng.randint(0, n, size=(n_resamples, n))
    # 对每个重采样取对应 diff 并求均值。
    # 形状：(n_resamples,)
    bootstrap_means = np.mean(diffs[resample_indices], axis=1)

    # 百分位法计算 95% 置信区间。
    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))

    # 单侧 p 值：bootstrap 中 mean_diff <= 0 的比例。
    # 检验 H0: mean(B - A) <= 0 vs H1: mean(B - A) > 0。
    p_value = float(np.mean(bootstrap_means <= 0))

    # 显著性：p 值与 CI 须一致。
    # p < 0.05 且整个 CI 在零以上。
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
