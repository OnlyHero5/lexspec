"""
多实验比较与分层显著性检验。

提供 run_all_comparisons() 用于实验间成对比较，
以及 stratified_significance() 用于按现象分层分析。
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

from src.evaluation.bootstrap import paired_bootstrap
from src.evaluation.wilcoxon import wilcoxon_test
from src.utils.progress import progress_bar
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_all_comparisons(
    experiment_results: Dict[str, List[float]],
    experiment_names: Optional[List[str]] = None,
    n_resamples: int = 10000,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """在实验间运行全部成对显著性检验。

    对一组实验（如 baseline、constrained、reflexion），
    对每一对同时运行 bootstrap 与 Wilcoxon 检验，
    生成完整比较矩阵。

    结果结构便于写入评估报告与表格。每对比较键为 "A_vs_B"，
    A、B 为实验名。

    参数:
        experiment_results: 实验名 -> 逐样本得分列表 的映射。
                            例如 {"baseline": [0.3, 0.4, ...],
                                  "constrained": [0.5, 0.6, ...]}
        experiment_names: 可选实验名有序列表。为 None 时使用
                          experiment_results 的排序键。
        n_resamples: 每次比较的 bootstrap 重采样次数。
        random_seed: 随机种子以保证可复现。

    返回:
        字典，键包括：
        - comparisons: Dict[str, Dict] — 各对比较结果。
          每项含 "bootstrap" 与 "wilcoxon" 子字典。
        - summary: Dict — 显著性结果矩阵视图。
        - n_pairs: int — 每个实验的配对样本数。

    异常:
        ValueError: 实验少于 2 个，或各实验得分长度不一致。
    """
    if len(experiment_results) < 2:
        raise ValueError(
            f"Need at least 2 experiments for comparison. Got {len(experiment_results)}."
        )

    # 确定实验顺序。
    names = experiment_names if experiment_names else sorted(experiment_results.keys())

    # 校验各实验样本数相同。
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

    # 运行全部成对比较。
    pair_list = [
        (names[i], names[j])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    for name_a, name_b in progress_bar(
        pair_list, desc="Pairwise comparisons", unit="pair",
    ):
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

    # 构建摘要显著性矩阵。
    # 行列为实验名；单元格为显著性指示。
    sig_matrix: Dict[str, Dict[str, str]] = {}
    for name in names:
        sig_matrix[name] = {}
        for other in names:
            if name == other:
                sig_matrix[name][other] = "-"
            else:
                # 查找比较键（与顺序无关）。
                key1 = f"{name}_vs_{other}"
                key2 = f"{other}_vs_{name}"
                comp = comparisons.get(key1) or comparisons.get(key2)
                if comp and "bootstrap" in comp and not isinstance(comp["bootstrap"].get("significant_at_0.05"), str):
                    bs = comp["bootstrap"]
                    if bs.get("significant_at_0.05", False):
                        # 判断方向：第二个实验是否更优？
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
    """在指定语言学现象子集上运行显著性检验。

    支持细粒度比较，例如「系统 B 是否在被动语态上优于 A？」
    或「提升是否限于长距离依存？」

    将逐样本得分过滤为仅含指定现象的样本，
    再在过滤子集上运行配对 bootstrap。

    参数:
        scores_a: 系统 A 的完整逐样本得分。
        scores_b: 系统 B 的完整逐样本得分。
        phenomenon_labels: 逐样本标签，表示各样本所属现象。
                           须与 scores_a、scores_b 等长。标签通常来自
                           ErrorCategory 枚举值或自定义标记。
        phenomenon_name: 要过滤的现象值。仅含该标签的样本参与检验。
        n_resamples: bootstrap 重采样次数。
        random_seed: 随机种子以保证可复现。

    返回:
        过滤子集的 bootstrap 结果，并附加：
        - phenomenon: str — 现象名。
        - subset_size: int — 该现象样本数。
        - total_size: int — 总样本数。
        - subset_ratio: float — 该现象样本占比。

    异常:
        ValueError: 输入长度不一致，或无样本匹配指定现象。
    """
    n = len(scores_a)
    if len(scores_b) != n or len(phenomenon_labels) != n:
        raise ValueError(
            f"All input lists must have the same length. "
            f"Got scores_a={len(scores_a)}, scores_b={len(scores_b)}, "
            f"labels={len(phenomenon_labels)}."
        )

    # 过滤为指定现象。
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

    # 在过滤子集上运行 bootstrap。
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
