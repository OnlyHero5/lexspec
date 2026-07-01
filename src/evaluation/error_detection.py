"""
Error detection helpers for field-level error identification.

Compares prediction vs gold on each field and determines the secondary
(field-level) error category.
"""

from __future__ import annotations

from typing import List

from src.extraction.schema import (
    LegalTriplet, FieldErrorType,
)
from src.evaluation.text_normalizer import normalize
from src.evaluation.field_f1 import token_f1


def detect_field_errors(
    prediction: LegalTriplet,
    gold: LegalTriplet,
) -> List[str]:
    """Compare each field of prediction vs gold and return a list of error labels.

    Error labels are the field paths with mismatches:
      - "subject.text": subject text token F1 < 1.0
      - "subject.role": roles differ
      - "action.predicate": predicate token F1 < 1.0
      - "action.object": object token F1 < 1.0
      - "condition.text": condition token F1 < 1.0
      - "condition.omission": prediction has no condition, gold has one
      - "condition.overextension": prediction has condition, gold has none OR
        prediction span is substantially larger

    Args:
        prediction: System prediction.
        gold: Gold standard.

    Returns:
        List of field-path strings indicating errors. Empty list if perfect match.
    """
    errors: List[str] = []

    # Subject text: token-level F1 comparison.
    pred_st = normalize(prediction.subject.text)
    gold_st = normalize(gold.subject.text)
    _, _, st_f1 = token_f1(pred_st, gold_st)
    if st_f1 < 1.0:
        errors.append("subject.text")

    # Subject role: exact enum match.
    if prediction.subject.role != gold.subject.role:
        errors.append("subject.role")

    # Predicate: token-level F1.
    pred_pr = normalize(prediction.action.predicate)
    gold_pr = normalize(gold.action.predicate)
    _, _, pr_f1 = token_f1(pred_pr, gold_pr)
    if pr_f1 < 1.0:
        errors.append("action.predicate")

    # Object: token-level F1.
    pred_ob = normalize(prediction.action.object)
    gold_ob = normalize(gold.action.object)
    _, _, ob_f1 = token_f1(pred_ob, gold_ob)
    if ob_f1 < 1.0:
        errors.append("action.object")

    # Condition: token-level F1 + omission/overextension detection.
    pred_co = normalize(prediction.condition.text)
    gold_co = normalize(gold.condition.text)

    has_pred_cond = bool(pred_co.strip())
    has_gold_cond = bool(gold_co.strip())

    if not has_pred_cond and has_gold_cond:
        # Omission: prediction missed a condition that exists in gold.
        errors.append("condition.omission")
    elif has_pred_cond and not has_gold_cond:
        # Over-extension: prediction hallucinated a condition.
        errors.append("condition.overextension")
    elif has_pred_cond and has_gold_cond:
        _, _, co_f1 = token_f1(pred_co, gold_co)
        if co_f1 < 1.0:
            # Both have conditions but they differ.
            # Determine if it's primarily a boundary issue.
            pred_tokens = set(pred_co.split())
            gold_tokens = set(gold_co.split())
            overlap = pred_tokens & gold_tokens
            # If prediction is substantially larger (2x+) with high overlap,
            # classify as overextension. Otherwise generic condition error.
            if len(pred_tokens) > 2 * len(gold_tokens) and len(overlap) >= len(gold_tokens) * 0.5:
                errors.append("condition.overextension")
            elif co_f1 < 0.5:
                errors.append("condition.text")

    return errors


def determine_secondary_category(field_errors: List[str]) -> FieldErrorType:
    """Map detected field errors to the secondary (field-level) error type.

    Priority order for classification when multiple fields are affected:
      1. Subject errors (most critical for legal analysis)
      2. Condition errors (boundary issues)
      3. Predicate/Object errors

    Args:
        field_errors: List of error field paths from detect_field_errors().

    Returns:
        A single FieldErrorType enum value representing the primary field
        affected.
    """
    # Check subject-related errors first (highest priority).
    if any(e.startswith("subject") for e in field_errors):
        if "subject.role" in field_errors and "subject.text" not in field_errors:
            return FieldErrorType.ROLE
        return FieldErrorType.SUBJECT

    # Check condition-related errors.
    if "condition.omission" in field_errors:
        return FieldErrorType.CONDITION_OMISSION
    if "condition.overextension" in field_errors:
        return FieldErrorType.CONDITION_OVEREXTENSION
    if any(e.startswith("condition") for e in field_errors):
        return FieldErrorType.CONDITION_OMISSION  # Default condition error type.

    # Check predicate/object errors.
    if "action.predicate" in field_errors:
        return FieldErrorType.PREDICATE
    if "action.object" in field_errors:
        return FieldErrorType.OBJECT

    # Fallback: if we have errors but none matched (should not happen),
    # default to SUBJECT as the most critical.
    return FieldErrorType.SUBJECT
