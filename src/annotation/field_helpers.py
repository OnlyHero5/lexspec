"""
Field-Level Helpers for Annotation Consensus
=============================================
Utilities for extracting, parsing, and comparing the six fine-grained
fields of a LegalTriplet during annotation consensus operations.

Exported:
  - FIELD_SPEC:                       Definition of the 6 comparison fields
  - _extract_field_values:            Extract field values from a LegalTriplet
  - _parse_role:                      Parse role value into LegalRole enum
  - _parse_condition_type:            Parse condition type into ConditionType enum
  - _classify_disagreement_phenomenon: Map disagreement to linguistic category
  - _triplets_equal:                  Check if two triplets are identical field-wise
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple

from src.extraction.schema import (
    LegalTriplet,
    LegalRole,
    ConditionType,
)
from src.annotation.normalization import normalize_text
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Fields compared at the finest grain (6 fields).
# Each entry is a tuple of (field_name, parent_attribute, child_attribute).
# This drives the field-by-field comparison loop.
# ---------------------------------------------------------------------------

FIELD_SPEC: List[Tuple[str, str, str]] = [
    ("subject.text",       "subject",     "text"),
    ("subject.role",       "subject",     "role"),
    ("action.predicate",   "action",      "predicate"),
    ("action.object",      "action",      "object"),
    ("condition.text",     "condition",   "text"),
    ("condition.type",     "condition",   "type"),
]


def _extract_field_values(triplet: LegalTriplet) -> Dict[str, Any]:
    """Extract the 6 individual field values from a LegalTriplet.

    Args:
        triplet: A LegalTriplet instance.

    Returns:
        Dict mapping field names (e.g., "subject.text") to their values.
    """
    return {
        "subject.text":     triplet.subject.text,
        "subject.role":     triplet.subject.role.value
                            if isinstance(triplet.subject.role, LegalRole)
                            else str(triplet.subject.role),
        "action.predicate": triplet.action.predicate,
        "action.object":    triplet.action.object,
        "condition.text":   triplet.condition.text,
        "condition.type":   triplet.condition.type.value
                            if isinstance(triplet.condition.type, ConditionType)
                            else str(triplet.condition.type),
    }


def _parse_role(value) -> LegalRole:
    """Parse a role value (string or LegalRole enum) into a LegalRole.

    Args:
        value: A string like "obligor" or a LegalRole enum instance.

    Returns:
        LegalRole enum value. Defaults to LegalRole.OTHER if unrecognized.
    """
    if isinstance(value, LegalRole):
        return value
    try:
        return LegalRole(str(value))
    except (ValueError, TypeError):
        logger.debug("Unrecognized role value '%s' -- defaulting to OTHER", value)
        return LegalRole.OTHER


def _parse_condition_type(value) -> ConditionType:
    """Parse a condition type value into a ConditionType enum.

    Args:
        value: A string like "trigger" or a ConditionType enum instance.

    Returns:
        ConditionType enum value. Defaults to ConditionType.NONE if unrecognized.
    """
    if isinstance(value, ConditionType):
        return value
    try:
        return ConditionType(str(value))
    except (ValueError, TypeError):
        logger.debug(
            "Unrecognized condition type '%s' -- defaulting to NONE", value
        )
        return ConditionType.NONE


def _classify_disagreement_phenomenon(
    field: str,
    qwen_anno: LegalTriplet,
    gemma_anno: LegalTriplet,
) -> str:
    """Classify a field-level disagreement into a linguistic phenomenon.

    Maps the field and context of the disagreement to a meaningful
    phenomenon label for reporting and diagnostics.

    Args:
        field: The field name that disagreed (e.g., "subject.role").
        qwen_anno: The Qwen annotation (for context).
        gemma_anno: The Gemma annotation (for context).

    Returns:
        A phenomenon label string (e.g., "role_mismatch", "passive_voice",
        "condition_boundary", "object_identification").
    """
    if field == "subject.role":
        # Role disagreements often stem from modality interpretation
        # or passive/active voice confusion.
        return "role_assignment"

    elif field == "subject.text":
        # Subject text disagreements may indicate different interpretations
        # of who the actor is (e.g., passive voice agent confusion).
        return "subject_identification"

    elif field == "action.predicate":
        # Predicate disagreements indicate different interpretations of
        # which verb is the main legal predicate.
        return "predicate_selection"

    elif field == "action.object":
        # Object disagreements often stem from long-distance dependencies
        # or scope ambiguity.
        return "object_identification"

    elif field in ("condition.text", "condition.type"):
        # Condition disagreements indicate boundary detection issues
        # or condition type classification differences.
        return "condition_detection"

    else:
        return "other"


def _triplets_equal(a: LegalTriplet, b: LegalTriplet) -> bool:
    """Check if two LegalTriplets are identical at the field level.

    Uses the same normalization as field_level_consensus to determine
    equality, so surface-form differences (case, articles) are ignored.

    Args:
        a: First triplet.
        b: Second triplet.

    Returns:
        True if all 6 fields agree after normalization, False otherwise.
    """
    a_vals = _extract_field_values(a)
    b_vals = _extract_field_values(b)

    for field_name, _, _ in FIELD_SPEC:
        a_val = str(a_vals[field_name])
        b_val = str(b_vals[field_name])

        is_text_field = field_name in (
            "subject.text", "action.predicate", "action.object", "condition.text"
        )

        if is_text_field:
            if normalize_text(a_val) != normalize_text(b_val):
                return False
        else:
            if a_val != b_val:
                return False

    return True
