"""
LexSpec Evaluation Module
=========================

Evaluates legal triplet extraction quality through three complementary dimensions:

1. **Weighted Triplet F1** (primary metric)
   Decomposes extraction quality into 5 field-level scores with configurable
   weights reflecting the relative importance of each component for legal
   contract analysis.

2. **Linguistic Metrics** (supplementary dimension, per PDF requirement)
   Four linguistic-specific metrics that diagnose performance on specific
   syntactic phenomena: dependency path legality, passive voice recovery,
   condition boundary IoU, and validator correction rates.

3. **Error Analysis** (diagnostic dimension, per PDF requirement)
   Two-level error classification (linguistic phenomenon x field error type)
   with bilingual (Chinese + English) linguistic explanations citing specific
   UD dependency relations.

Statistical significance testing (paired bootstrap + Wilcoxon) supports
rigorous comparison between experiment variants.

Usage:
    from src.evaluation import (
        normalize, normalize_triplet, load_party_aliases,
        compute_triplet_f1, compute_per_sample_f1,
        compute_all_linguistic_metrics,
        paired_bootstrap, wilcoxon_test,
        classify_errors, generate_error_report, error_distribution_report,
        save_error_cases, generate_error_summary,
    )

    # Normalize text for fair comparison
    text = normalize("the Seller shall deliver the Goods.")

    # Compute primary evaluation metric
    results = compute_triplet_f1(predictions, gold)

    # Compute supplementary linguistic metrics
    ling_metrics = compute_all_linguistic_metrics(predictions, gold, trees)

    # Run significance testing between experiments
    sig = paired_bootstrap(baseline_scores, our_scores)

    # Classify and analyze errors
    errors = classify_errors(predictions, gold, trees)
    dist = error_distribution_report(errors)
    print(generate_error_summary(errors))
"""

# ---------------------------------------------------------------------------
# Text Normalization
# ---------------------------------------------------------------------------
from src.evaluation.normalization import (
    normalize,
    normalize_triplet,
    load_party_aliases,
    NUMBER_WORDS,
)

# ---------------------------------------------------------------------------
# Weighted Triplet F1
# ---------------------------------------------------------------------------
from src.evaluation.triplet_f1 import (
    compute_triplet_f1,
)
from src.evaluation.field_f1 import (
    compute_field_f1,
    compute_per_sample_f1,
    DEFAULT_WEIGHTS,
)

# ---------------------------------------------------------------------------
# Linguistic Metrics
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
# Statistical Significance
# ---------------------------------------------------------------------------
from src.evaluation.significance import (
    paired_bootstrap,
    wilcoxon_test,
    run_all_comparisons,
    stratified_significance,
)

# ---------------------------------------------------------------------------
# Error Analysis
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
# Public API — everything callers should import from this module
# =============================================================================

__all__ = [
    # Normalization
    "normalize",
    "normalize_triplet",
    "load_party_aliases",
    "NUMBER_WORDS",
    # Triplet F1
    "compute_triplet_f1",
    "compute_field_f1",
    "compute_per_sample_f1",
    "DEFAULT_WEIGHTS",
    # Linguistic metrics
    "compute_dependency_path_legality",
    "compute_passive_recovery_accuracy",
    "compute_condition_iou",
    "compute_correction_rate",
    "compute_all_linguistic_metrics",
    # Significance
    "paired_bootstrap",
    "wilcoxon_test",
    "run_all_comparisons",
    "stratified_significance",
    # Error analysis
    "generate_error_report",
    "classify_errors",
    "error_distribution_report",
    "save_error_cases",
    "generate_error_summary",
]
