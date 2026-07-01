"""
Condition Boundary IoU and Linguistic Correction Rate metrics.

  - Condition Boundary IoU: Measures token-level Intersection-over-Union
    between predicted condition spans and UD-derived condition clause spans.
  - Linguistic Correction Rate: Statistics on how often the UD constraint
    validator successfully corrects LLM extraction errors.
"""

from __future__ import annotations

from typing import List, Dict, Set

from src.extraction.schema import (
    LegalTriplet, DependencyTree, ValidationResult,
)
from src.evaluation.text_normalizer import normalize
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_condition_iou(
    predictions: List[LegalTriplet],
    trees: List[DependencyTree],
) -> float:
    """Compute mean token-level IoU between predicted condition spans
    and UD condition clause spans.

    For each clause where both the prediction and UD tree have a condition
    (i.e., advcl relation present in the tree AND the prediction has non-empty
    condition.text), compute:

        IoU = |prediction_tokens ∩ condition_subtree_tokens| /
              |prediction_tokens ∪ condition_subtree_tokens|

    This measures how accurately the system identifies condition clause
    boundaries. Perfect IoU = 1.0 means the predicted span exactly matches
    the UD-derived condition clause span.

    Clauses where neither prediction nor tree has a condition are excluded
    (they trivially agree). Clauses where only one side has a condition
    count as IoU = 0.0 for that clause.

    Args:
        predictions: List of predicted LegalTriplets.
        trees: List of DependencyTree objects (same length).

    Returns:
        Mean IoU across all clause pairs where at least one side has a
        condition. Float in [0, 1].

    Raises:
        ValueError: If input lists have different lengths.
    """
    if len(predictions) != len(trees):
        raise ValueError(
            f"predictions and trees must have the same length. "
            f"Got {len(predictions)} and {len(trees)}."
        )

    ious: List[float] = []
    skipped_no_condition = 0
    skipped_one_side = 0

    for pred, tree in zip(predictions, trees):
        if tree.token_count == 0:
            continue

        # Check if the UD tree has a condition clause (advcl).
        advcl_tokens = tree.find_tokens_by_deprel("advcl")
        has_tree_condition = len(advcl_tokens) > 0
        has_pred_condition = bool(pred.condition.text and pred.condition.text.strip())

        if not has_tree_condition and not has_pred_condition:
            # Both agree there is no condition — skip (not informative for IoU).
            skipped_no_condition += 1
            continue

        # Get prediction condition tokens.
        pred_text = normalize(pred.condition.text, remove_articles=True)
        pred_token_set = set(pred_text.split()) if pred_text else set()

        # Get tree condition subtree tokens (from all advcl spans).
        tree_condition_set: Set[str] = set()
        for advcl_token in advcl_tokens:
            # Get the full subtree of this advcl head.
            subtree_tokens = tree.get_subtree_tokens(advcl_token.index)
            for st in subtree_tokens:
                # Use lemma for robust matching (normalized).
                tree_condition_set.add(st.lemma.lower())

        if not pred_token_set or not tree_condition_set:
            # One side has no tokens — IoU = 0.
            ious.append(0.0)
            skipped_one_side += 1
            continue

        # Compute IoU.
        intersection = pred_token_set & tree_condition_set
        union = pred_token_set | tree_condition_set

        iou = len(intersection) / len(union) if union else 0.0
        ious.append(iou)

    mean_iou = sum(ious) / len(ious) if ious else 0.0

    logger.info(
        "Condition Boundary IoU: %.4f (n=%d, skipped_no_cond=%d, skipped_one_side=%d)",
        mean_iou, len(ious), skipped_no_condition, skipped_one_side,
    )
    return mean_iou


def compute_correction_rate(
    validation_results: List[ValidationResult],
) -> Dict[str, float]:
    """Compute statistics on how often the UD constraint validator
    successfully corrects LLM extraction errors.

    The validator produces one of three statuses per prediction:
      - VALID:              Prediction matches UD evidence; no corrections.
      - CORRECTED:          Minor field errors were automatically fixed.
      - REFLEXION_REQUIRED: Structural errors requiring LLM re-extraction.

    This metric reports the distribution of these statuses and the
    correction success rate: how often the validator can fix an error
    automatically (CORRECTED) vs. needing LLM re-extraction (REFLEXION).

    Args:
        validation_results: List of ValidationResult objects from the
                            UD constraint validator.

    Returns:
        Dict with keys:
        - total_validated: int — total number of validation results.
        - valid_count: int — results with VALID status.
        - valid_rate: float — VALID / total.
        - corrected_count: int — results with CORRECTED status.
        - corrected_rate: float — CORRECTED / total.
        - reflexion_count: int — results with REFLEXION_REQUIRED status.
        - reflexion_rate: float — REFLEXION_REQUIRED / total.
        - correction_success_rate: float — CORRECTED / (CORRECTED + REFLEXION).
          This is the proportion of errors that the validator can fix
          automatically without LLM re-extraction.
    """
    total = len(validation_results)
    if total == 0:
        return {
            "total_validated": 0,
            "valid_count": 0,
            "valid_rate": 0.0,
            "corrected_count": 0,
            "corrected_rate": 0.0,
            "reflexion_count": 0,
            "reflexion_rate": 0.0,
            "correction_success_rate": 0.0,
        }

    from src.extraction.schema import ValidationStatus

    valid_count = sum(1 for r in validation_results if r.status == ValidationStatus.VALID)
    corrected_count = sum(1 for r in validation_results if r.status == ValidationStatus.CORRECTED)
    reflexion_count = sum(1 for r in validation_results if r.status == ValidationStatus.REFLEXION_REQUIRED)

    valid_rate = valid_count / total
    corrected_rate = corrected_count / total
    reflexion_rate = reflexion_count / total

    # Correction success rate: of the errors (corrected + reflexion),
    # how many could be fixed automatically?
    error_total = corrected_count + reflexion_count
    correction_success_rate = corrected_count / error_total if error_total > 0 else 0.0

    logger.info(
        "Correction rate: total=%d, VALID=%d(%.2f), CORRECTED=%d(%.2f), "
        "REFLEXION=%d(%.2f), success=%.4f",
        total, valid_count, valid_rate, corrected_count, corrected_rate,
        reflexion_count, reflexion_rate, correction_success_rate,
    )

    return {
        "total_validated": total,
        "valid_count": valid_count,
        "valid_rate": valid_rate,
        "corrected_count": corrected_count,
        "corrected_rate": corrected_rate,
        "reflexion_count": reflexion_count,
        "reflexion_rate": reflexion_rate,
        "correction_success_rate": correction_success_rate,
    }
