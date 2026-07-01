"""
Error Type Determination for Reflexion
=======================================
Analyzes ValidationResult corrections and linguistic evidence to
determine which syntactic phenomena caused extraction errors.

Exported:
  - determine_error_types: Map validation corrections to error hint keys
"""

from __future__ import annotations

from typing import List

from src.extraction.schema import ValidationResult, FieldCorrection, LinguisticEvidence
from src.utils.logging import get_logger

logger = get_logger(__name__)


def determine_error_types(
    validation_result: ValidationResult,
) -> List[str]:
    """Analyze corrections to determine linguistic error type(s).

    Examines the FieldCorrection list and LinguisticEvidence to
    identify which syntactic phenomena caused the extraction errors.
    Returns a priority-ordered list of error hint keys.

    Priority order (most specific/most common first):
      1. Passive voice errors -- checked first because passive is
         the single largest source of extraction errors in legal text.
      2. Condition boundary errors -- condition clauses are frequently
         truncated or over-extended.
      3. Negation role errors -- "shall not" vs "shall" role reversal.
      4. Role mismatch -- modality-role alignment errors.
      5. Long-distance dependency errors -- non-local object attachment.
      6. Default -- generic re-analysis when no specific pattern matches.

    Args:
        validation_result: Validation result with corrections and evidence.

    Returns:
        List of error hint keys (e.g., ["passive_subject", "role_mismatch"]).
        Empty list if status is VALID (no corrections needed).
    """
    corrections: List[FieldCorrection] = validation_result.corrections
    evidence: LinguisticEvidence = validation_result.linguistic_evidence

    # If the status is VALID, there are no corrections to analyze.
    if validation_result.status.value == "VALID":
        return []

    error_types: List[str] = []

    # Collect which fields were corrected for quick membership checks.
    corrected_fields = {c.field for c in corrections}

    # --- Priority 1: Passive Voice Errors ---
    # Passive voice is the most common cause of extraction errors in
    # legal contracts. Check it first to ensure these errors dominate
    # the error type signal.
    if evidence.passive_detected:
        # Determine if the subject or object was misassigned.
        subject_corrected = any(
            f in corrected_fields for f in ("subject.text", "subject.role")
        )
        object_corrected = "action.object" in corrected_fields

        if subject_corrected:
            # The LLM likely used the surface (patient) subject as the
            # actor; it should have used the obl:agent instead.
            error_types.append("passive_subject")
        if object_corrected:
            # The LLM likely failed to recognize the surface subject
            # as the logical object.
            error_types.append("passive_object")

    # --- Priority 2: Condition Boundary Errors ---
    # Check if any condition-related fields were corrected.
    condition_corrected = any(
        f in corrected_fields
        for f in ("condition.text", "condition.type")
    )
    if condition_corrected:
        error_types.append("condition_boundary")

    # --- Priority 3: Negation Role Errors ---
    # If the clause has negative polarity and the role was corrected,
    # this is likely a negation-induced role reversal.
    if evidence.polarity == "negative":
        role_corrected = "subject.role" in corrected_fields
        if role_corrected:
            error_types.append("negation_role")

    # --- Priority 4: Role Mismatch ---
    # If the subject role was corrected but it wasn't already captured
    # by passive or negation, it's a modality-role alignment error.
    if "subject.role" in corrected_fields and "negation_role" not in error_types:
        # Only add role_mismatch if it wasn't already covered by
        # passive_subject (passive with role error is more specific).
        if "passive_subject" not in error_types:
            error_types.append("role_mismatch")

    # --- Priority 5: Long-Distance Dependency ---
    # If the action object or predicate was corrected and wasn't
    # caught by passive_object, check for long-distance dependency.
    action_corrected = any(
        f in corrected_fields
        for f in ("action.object", "action.predicate")
    )
    if action_corrected and "passive_object" not in error_types:
        # Check evidence for dependency distance > 3 as a heuristic
        # for long-distance dependency. Not definitive but useful.
        pred_idx = evidence.predicate_index
        if pred_idx > 0:
            # If predicate index and corrections suggest non-local
            # attachment, flag as long-distance.
            error_types.append("long_distance_object")
        else:
            # Predicate not found -- the predicate itself was wrong,
            # which is a broader error, not specifically long-distance.
            pass

    # --- Fallback: Default ---
    # If no specific error type was identified but corrections exist,
    # use the generic default hint.
    if not error_types and corrections:
        error_types.append("default")

    logger.debug(
        "Error type analysis: corrected_fields=%s, passive=%s, polarity=%s -> %s",
        corrected_fields,
        evidence.passive_detected,
        evidence.polarity,
        error_types,
    )

    return error_types
