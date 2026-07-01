"""
Weighted triplet F1 computation — the primary task evaluation metric.

Computes weighted F1 across the 5 triplet fields with configurable weights:
  subject.text:    0.35 — highest; subject attribution is the core error
  subject.role:    0.10 — classification accuracy (exact enum match)
  action.predicate: 0.20 — token-level F1 on lemma
  action.object:    0.20 — token-level span F1
  condition.text:   0.15 — token-level span F1

Theory:
  Legal information extraction is a structured prediction task. A single
  F1 number is insufficient because errors carry different costs (misidentifying
  the obligated party has severe downstream consequences; a fuzzy condition
  boundary is less critical). Weighted F1 provides a principled decomposition.

  Token-level matching gives partial credit for partially correct extractions.
  "the Goods sold" vs "the Goods" gets a precision of 2/3 and recall of 2/2,
  resulting in F1 ≈ 0.80 rather than 0 (exact mismatch).

Design:
  - All text comparison passes through normalize() from text_normalizer.py.
  - Weights sum to 1.0; this is enforced by normalization in compute_triplet_f1().
  - Per-sample F1 is computable for statistical significance testing.
  - The default weights are aligned with constraints.yaml f1_weights section.
"""

from __future__ import annotations

from typing import Optional, Dict, List

from src.extraction.schema import LegalTriplet
from src.evaluation.text_normalizer import normalize
from src.evaluation.field_f1 import DEFAULT_WEIGHTS, compute_field_f1
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_triplet_f1(
    predictions: List[LegalTriplet],
    gold: List[LegalTriplet],
    weights: Optional[Dict[str, float]] = None,
    party_aliases: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, float]:
    """Compute weighted triplet F1 across all 5 fields.

    For each of the 5 fields:
    - subject.text:     Token-level F1 after normalization.
                         Treats subject text as a set of tokens, computes
                         precision/recall/F1 on token overlap. This gives
                         partial credit.
    - subject.role:     Classification accuracy — exact enum match.
                         No partial credit because roles are discrete.
    - action.predicate: Token-level F1 on lemma form. Partial credit for
                         near-matches like "deliver" vs "shall deliver".
    - action.object:    Token-level span F1. The same token-overlap approach.
    - condition.text:   Token-level span F1. Partial credit for boundary errors.

    Overall F1 = weighted average of field-level F1 scores, where weights
    are normalized to sum to 1.0.

    The input lists must be the same length (1:1 alignment). If they differ,
    only the overlapping prefix is evaluated (with a warning).

    Args:
        predictions: List of predicted LegalTriplets from the system.
        gold: List of gold-standard LegalTriplets (same length as predictions).
        weights: Optional weight override dict. Keys must match DEFAULT_WEIGHTS
                 keys. If None, uses DEFAULT_WEIGHTS. Weights are normalized
                 to sum to 1.0 regardless of input values.
        party_aliases: Optional party alias mappings passed to normalize()
                       for entity normalization during text comparison.

    Returns:
        Dict with keys:
        - subject_text_f1, subject_text_precision, subject_text_recall
        - subject_role_acc
        - predicate_f1, predicate_precision, predicate_recall
        - object_f1, object_precision, object_recall
        - condition_f1, condition_precision, condition_recall
        - overall_f1 (weighted average of the 5 field F1 scores)

    Raises:
        ValueError: If predictions and gold have different lengths (after
                    truncation warning) or if both are empty.
    """
    n_pred = len(predictions)
    n_gold = len(gold)

    if n_pred == 0 and n_gold == 0:
        raise ValueError("Cannot compute F1 on empty prediction and gold lists.")

    if n_pred != n_gold:
        min_len = min(n_pred, n_gold)
        logger.warning(
            "Length mismatch: predictions=%d, gold=%d. Evaluating on first %d pairs.",
            n_pred, n_gold, min_len,
        )
        predictions = predictions[:min_len]
        gold = gold[:min_len]

    n = len(predictions)

    # Normalize weights: ensure they sum to 1.0.
    w = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
    weight_sum = sum(w.values())
    if abs(weight_sum - 1.0) > 1e-9:
        logger.debug("Normalizing weights from sum=%.4f to 1.0", weight_sum)
        w = {k: v / weight_sum for k, v in w.items()}

    # -------------------------------------------------------------------
    # Extract and normalize field values for all samples
    # -------------------------------------------------------------------

    # subject.text: normalize both sides.
    pred_subject_texts = [
        normalize(p.subject.text, party_aliases=party_aliases)
        for p in predictions
    ]
    gold_subject_texts = [
        normalize(g.subject.text, party_aliases=party_aliases)
        for g in gold
    ]

    # subject.role: extract enum values as strings for comparison.
    pred_subject_roles = [p.subject.role.value for p in predictions]
    gold_subject_roles = [g.subject.role.value for g in gold]

    # action.predicate: normalize for token-level comparison.
    pred_predicates = [
        normalize(p.action.predicate, party_aliases=party_aliases)
        for p in predictions
    ]
    gold_predicates = [
        normalize(g.action.predicate, party_aliases=party_aliases)
        for g in gold
    ]

    # action.object: normalize.
    pred_objects = [
        normalize(p.action.object, party_aliases=party_aliases)
        for p in predictions
    ]
    gold_objects = [
        normalize(g.action.object, party_aliases=party_aliases)
        for g in gold
    ]

    # condition.text: normalize. Empty strings for NONE conditions.
    pred_conditions = [
        normalize(p.condition.text, party_aliases=party_aliases)
        for p in predictions
    ]
    gold_conditions = [
        normalize(g.condition.text, party_aliases=party_aliases)
        for g in gold
    ]

    # -------------------------------------------------------------------
    # Compute per-field metrics
    # -------------------------------------------------------------------

    # Subject text: token-level F1.
    st_precision, st_recall, st_f1 = compute_field_f1(
        pred_subject_texts, gold_subject_texts, match_type="token"
    )

    # Subject role: classification accuracy.
    role_correct = sum(1 for p, g in zip(pred_subject_roles, gold_subject_roles) if p == g)
    role_acc = role_correct / n if n > 0 else 0.0

    # Predicate: token-level F1.
    pr_precision, pr_recall, pr_f1 = compute_field_f1(
        pred_predicates, gold_predicates, match_type="token"
    )

    # Object: token-level F1.
    ob_precision, ob_recall, ob_f1 = compute_field_f1(
        pred_objects, gold_objects, match_type="token"
    )

    # Condition: token-level F1.
    co_precision, co_recall, co_f1 = compute_field_f1(
        pred_conditions, gold_conditions, match_type="token"
    )

    # -------------------------------------------------------------------
    # Compute weighted overall F1
    # -------------------------------------------------------------------
    # Round to 10 decimal places to suppress floating-point representation
    # artifacts (e.g., 0.35 + 0.10 + 0.20 + 0.20 + 0.15 = 0.9999999999999999
    # in IEEE 754 when all components are 1.0). This is semantically 1.0.
    overall = round(
        w.get("subject_text", 0.35) * st_f1
        + w.get("subject_role", 0.10) * role_acc
        + w.get("predicate", 0.20) * pr_f1
        + w.get("object", 0.20) * ob_f1
        + w.get("condition", 0.15) * co_f1,
        10,
    )

    logger.info(
        "Weighted triplet F1: overall=%.4f (n=%d, st=%.3f, role=%.3f, pred=%.3f, obj=%.3f, cond=%.3f)",
        overall, n, st_f1, role_acc, pr_f1, ob_f1, co_f1,
    )

    return {
        "subject_text_f1": st_f1,
        "subject_text_precision": st_precision,
        "subject_text_recall": st_recall,
        "subject_role_acc": role_acc,
        "predicate_f1": pr_f1,
        "predicate_precision": pr_precision,
        "predicate_recall": pr_recall,
        "object_f1": ob_f1,
        "object_precision": ob_precision,
        "object_recall": ob_recall,
        "condition_f1": co_f1,
        "condition_precision": co_precision,
        "condition_recall": co_recall,
        "overall_f1": overall,
    }
