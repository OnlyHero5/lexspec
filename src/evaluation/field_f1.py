"""
Field-level F1 computation with token-level matching.

Provides DEFAULT_WEIGHTS, compute_field_f1, token_f1, and compute_per_sample_f1.
These are used by triplet_f1.py for the primary evaluation metric and by
error_analyzer.py for per-field error detection.

Design:
  - All text comparison passes through normalize() from text_normalizer.py.
  - Weights sum to 1.0; this is enforced by normalization in compute_triplet_f1().
  - Per-sample F1 is computable for statistical significance testing.
  - The default weights are aligned with constraints.yaml f1_weights section.
"""

from __future__ import annotations

from typing import Optional, Dict, List, Tuple

from src.extraction.schema import LegalTriplet
from src.evaluation.text_normalizer import normalize
from src.utils.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# Default F1 component weights
# =============================================================================
# These weights reflect the relative importance of each extraction component
# for legal contract analysis. They match the values in configs/constraints.yaml.
# All weights must sum to 1.0.
#
# subject_text:  0.35 — party identification is the most critical task.
# subject_role:  0.10 — role classification has inherent ambiguity.
# predicate:     0.20 — action identification is co-equal with object.
# object:        0.20 — the target of the action.
# condition:     0.15 — condition extraction; not every clause has one.

DEFAULT_WEIGHTS: Dict[str, float] = {
    "subject_text": 0.35,
    "subject_role": 0.10,
    "predicate": 0.20,
    "object": 0.20,
    "condition": 0.15,
}


def compute_field_f1(
    preds: List[str],
    golds: List[str],
    match_type: str = "token",
) -> Tuple[float, float, float]:
    """Compute macro-averaged F1 for a single field across all samples.

    For token-level matching: each (pred, gold) pair is tokenized and
    compared via set overlap. Precision, recall, and F1 are computed per
    sample and then macro-averaged (mean across samples). This gives each
    sample equal weight regardless of text length.

    For exact matching: a binary correct/incorrect per sample, then mean.

    Args:
        preds: List of predicted strings for a single field.
        golds: List of gold-standard strings for the same field.
               Must be the same length as preds.
        match_type: "token" for token-level F1 with partial credit,
                    "exact" for strict string equality after normalization.

    Returns:
        Tuple of (macro_precision, macro_recall, macro_f1).
        macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall).
    """
    n = len(preds)
    if n == 0:
        return (0.0, 0.0, 0.0)

    precisions: List[float] = []
    recalls: List[float] = []

    for pred, gold in zip(preds, golds):
        if match_type == "exact":
            # Binary: 1.0 if strings are identical after normalization,
            # 0.0 otherwise (precision=recall=F1 for exact match).
            match = 1.0 if pred == gold else 0.0
            precisions.append(match)
            recalls.append(match)
        elif match_type == "token":
            p, r, _ = token_f1(pred, gold)
            precisions.append(p)
            recalls.append(r)
        else:
            raise ValueError(f"Unknown match_type: '{match_type}'. Use 'token' or 'exact'.")

    # Macro-average: mean over all samples.
    macro_p = sum(precisions) / n if n > 0 else 0.0
    macro_r = sum(recalls) / n if n > 0 else 0.0

    # Compute F1 from macro-averaged precision and recall.
    denom = macro_p + macro_r
    macro_f1 = 2 * macro_p * macro_r / denom if denom > 0 else 0.0

    return (macro_p, macro_r, macro_f1)


def token_f1(pred: str, gold: str) -> Tuple[float, float, float]:
    """Compute token-level F1 between two strings via set overlap.

    Tokenization: split on whitespace into tokens.
    Empty strings produce empty token sets.

    Precision = |pred_tokens ∩ gold_tokens| / |pred_tokens|
    Recall    = |pred_tokens ∩ gold_tokens| / |gold_tokens|
    F1        = 2 * P * R / (P + R)

    Edge cases:
    - Both empty: perfect match, returns (1.0, 1.0, 1.0).
    - One empty: zero overlap, returns (0.0, 0.0, 0.0).
    - Both non-empty with no overlap: returns (0.0, 0.0, 0.0).

    Args:
        pred: Predicted string (should already be normalized).
        gold: Gold-standard string (should already be normalized).

    Returns:
        Tuple of (precision, recall, f1), each in [0, 1].
    """
    pred_tokens = set(pred.split()) if pred.strip() else set()
    gold_tokens = set(gold.split()) if gold.strip() else set()

    # Both empty: the prediction correctly identified that nothing should
    # be extracted (e.g., condition NONE with empty text).
    if not pred_tokens and not gold_tokens:
        return (1.0, 1.0, 1.0)

    # One empty, one not: complete mismatch.
    if not pred_tokens or not gold_tokens:
        return (0.0, 0.0, 0.0)

    # Compute intersection and derived metrics.
    intersection = pred_tokens & gold_tokens

    precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(intersection) / len(gold_tokens) if gold_tokens else 0.0

    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else 0.0

    return (precision, recall, f1)


def compute_per_sample_f1(
    predictions: List[LegalTriplet],
    gold: List[LegalTriplet],
    weights: Optional[Dict[str, float]] = None,
    party_aliases: Optional[Dict[str, List[str]]] = None,
) -> List[float]:
    """Compute per-sample weighted F1 scores for significance testing.

    This is the entry point for statistical tests (paired bootstrap,
    Wilcoxon). It returns a list of F1 scores, one per sample, that
    can be passed directly to paired_bootstrap() or wilcoxon_test().

    Each sample's F1 is the weighted average of its 5 field scores,
    using the same weight scheme as compute_triplet_f1().

    Args:
        predictions: List of predicted LegalTriplets.
        gold: List of gold-standard LegalTriplets.
        weights: Optional weight override dict.
        party_aliases: Optional party alias mappings for normalization.

    Returns:
        List of float F1 scores, one per sample, in [0, 1].

    Raises:
        ValueError: If predictions and gold have different lengths.
    """
    n_pred = len(predictions)
    n_gold = len(gold)

    if n_pred != n_gold:
        raise ValueError(
            f"Predictions and gold must have the same length for per-sample F1. "
            f"Got {n_pred} predictions and {n_gold} gold."
        )

    n = n_pred
    if n == 0:
        return []

    # Use provided weights or defaults, normalized to sum to 1.0.
    w = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
    weight_sum = sum(w.values())
    if abs(weight_sum - 1.0) > 1e-9:
        w = {k: v / weight_sum for k, v in w.items()}

    per_sample_f1: List[float] = []

    for pred, gold_item in zip(predictions, gold):
        # Normalize subject text for comparison.
        pred_st = normalize(pred.subject.text, party_aliases=party_aliases)
        gold_st = normalize(gold_item.subject.text, party_aliases=party_aliases)

        # Subject text: token-level F1 for this sample.
        _, _, st_f1 = token_f1(pred_st, gold_st)

        # Subject role: binary correct/incorrect.
        role_score = 1.0 if pred.subject.role == gold_item.subject.role else 0.0

        # Predicate: token-level F1.
        pred_pr = normalize(pred.action.predicate, party_aliases=party_aliases)
        gold_pr = normalize(gold_item.action.predicate, party_aliases=party_aliases)
        _, _, pr_f1 = token_f1(pred_pr, gold_pr)

        # Object: token-level F1.
        pred_ob = normalize(pred.action.object, party_aliases=party_aliases)
        gold_ob = normalize(gold_item.action.object, party_aliases=party_aliases)
        _, _, ob_f1 = token_f1(pred_ob, gold_ob)

        # Condition: token-level F1.
        pred_co = normalize(pred.condition.text, party_aliases=party_aliases)
        gold_co = normalize(gold_item.condition.text, party_aliases=party_aliases)
        _, _, co_f1 = token_f1(pred_co, gold_co)

        # Weighted average for this sample (rounded to suppress IEEE 754 artifacts).
        sample_f1 = round(
            w.get("subject_text", 0.35) * st_f1
            + w.get("subject_role", 0.10) * role_score
            + w.get("predicate", 0.20) * pr_f1
            + w.get("object", 0.20) * ob_f1
            + w.get("condition", 0.15) * co_f1,
            10,
        )
        per_sample_f1.append(sample_f1)

    return per_sample_f1
