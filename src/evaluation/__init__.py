"""
LexSpec 评估模块
================

通过三个互补维度评估法律三元组抽取质量：

1. **加权三元组 F1**（主指标）
   将抽取质量分解为 5 个字段级得分，权重可配置，反映各组成部分
   对法律合同分析的相对重要性。

2. **语言学指标**（补充维度，按 PDF 要求）
   四项语言学专用指标，用于诊断特定句法现象上的表现：依存路径合法性、
   被动语态恢复、条件边界 IoU，以及验证器修正率。

3. **错误分析**（诊断维度，按 PDF 要求）
   两级错误分类（语言学现象 × 字段错误类型），附中英双语语言学解释，
   并引用具体 UD 依存关系。

统计显著性检验（配对 bootstrap + Wilcoxon）支持实验变体之间的严格比较。

用法:
    from src.evaluation import (
        normalize, normalize_triplet, load_party_aliases,
        compute_triplet_f1, compute_per_sample_f1,
        compute_all_linguistic_metrics,
        paired_bootstrap, wilcoxon_test,
        classify_errors, generate_error_report, error_distribution_report,
        save_error_cases, generate_error_summary,
    )

    # 归一化文本以便公平比较
    text = normalize("the Seller shall deliver the Goods.")

    # 计算主评估指标
    results = compute_triplet_f1(predictions, gold)

    # 计算补充语言学指标
    ling_metrics = compute_all_linguistic_metrics(predictions, gold, trees)

    # 在实验间运行显著性检验
    sig = paired_bootstrap(baseline_scores, our_scores)

    # 分类并分析错误
    errors = classify_errors(predictions, gold, trees)
    dist = error_distribution_report(errors)
    print(generate_error_summary(errors))
"""

# ---------------------------------------------------------------------------
# 文本归一化
# ---------------------------------------------------------------------------
from src.evaluation.normalization import (
    normalize,
    normalize_triplet,
    load_party_aliases,
    NUMBER_WORDS,
)

# ---------------------------------------------------------------------------
# 加权三元组 F1
# ---------------------------------------------------------------------------
from src.evaluation.triplet_f1 import (
    compute_triplet_f1,
)
from src.evaluation.field_f1 import (
    compute_field_f1,
    compute_per_sample_f1,
    load_f1_weights,
)

# ---------------------------------------------------------------------------
# 语言学指标
# ---------------------------------------------------------------------------
from src.evaluation.dep_path_metrics import (
    compute_dependency_path_legality,
)
from src.evaluation.passive_metrics import (
    compute_passive_recovery_accuracy,
)
from src.evaluation.condition_metrics import (
    compute_condition_iou,
    compute_correction_rate,
)
from src.evaluation.linguistic_metrics import (
    compute_all_linguistic_metrics,
)

# ---------------------------------------------------------------------------
# 统计显著性
# ---------------------------------------------------------------------------
from src.evaluation.significance import (
    paired_bootstrap,
    wilcoxon_test,
    run_all_comparisons,
    stratified_significance,
)

# ---------------------------------------------------------------------------
# 错误分析
# ---------------------------------------------------------------------------
from src.evaluation.error_analyzer import (
    generate_error_report,
    classify_errors,
    save_error_cases,
)
from src.evaluation.error_summary import (
    error_distribution_report,
    generate_error_summary,
)

# =============================================================================
# 公开 API — 调用方应从此模块导入的全部符号
# =============================================================================

__all__ = [
    # 归一化
    "normalize",
    "normalize_triplet",
    "load_party_aliases",
    "NUMBER_WORDS",
    # 三元组 F1
    "compute_triplet_f1",
    "compute_field_f1",
    "compute_per_sample_f1",
    "load_f1_weights",
    # 语言学指标
    "compute_dependency_path_legality",
    "compute_passive_recovery_accuracy",
    "compute_condition_iou",
    "compute_correction_rate",
    "compute_all_linguistic_metrics",
    # 显著性
    "paired_bootstrap",
    "wilcoxon_test",
    "run_all_comparisons",
    "stratified_significance",
    # 错误分析
    "generate_error_report",
    "classify_errors",
    "error_distribution_report",
    "save_error_cases",
    "generate_error_summary",
]
