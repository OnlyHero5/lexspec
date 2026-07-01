"""
三元组校验与规范化 —— Pydantic 模型校验 + 枚举值规范化
======================================================

将 LLM 返回的解析后字典转换为经过校验的 ``LegalTriplet`` 实例，
包含枚举值模糊匹配和字段回退逻辑。
"""

from __future__ import annotations

from typing import Dict, Any

from src.extraction.schema import (
    LegalTriplet,
    Subject,
    Action,
    Condition,
    LegalRole,
    ConditionType,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def validate_and_normalize_triplet(
    parsed: Dict[str, Any],
    clause: str,
) -> LegalTriplet:
    """Validate parsed dict against the LegalTriplet schema and normalize.

    This method bridges the gap between whatever the LLM returned and
    the strict ``LegalTriplet`` Pydantic model.  It:

    1. Coerces enum string values (e.g., ``"OBLIGOR"`` → ``LegalRole.OBLIGOR``)
    2. Fills in missing optional fields with sensible defaults
    3. Runs Pydantic validation (``LegalTriplet.model_validate``)
    4. Normalizes the output (whitespace trimming, condition cleanup)

    Args:
        parsed:  The parsed JSON dict from the LLM response.
        clause:  The original clause text (for logging context).

    Returns:
        A validated ``LegalTriplet`` instance.

    Raises:
        ValueError:  If the parsed dict cannot be validated against the
                     ``LegalTriplet`` schema even after coercion.
    """
    # --- Step 1: Extract and coerce nested dicts ---
    # The LLM output is expected to have top-level keys "subject",
    # "action", and "condition", each containing a nested dict.
    # We extract these, providing empty dicts if a key is missing
    # entirely (the Pydantic model will then catch missing fields).
    subject_raw = parsed.get("subject", {})
    action_raw = parsed.get("action", {})
    condition_raw = parsed.get("condition", {})

    # Handle string values for subject/action/condition (edge case where
    # the LLM returns a plain string instead of an object).
    if isinstance(subject_raw, str):
        # The LLM might return just a party name as a string.
        # Convert to a minimal dict so validation doesn't crash.
        logger.debug("subject was a string — converting to dict: %r", subject_raw)
        subject_raw = {"text": subject_raw, "role": "other"}
    if isinstance(action_raw, str):
        logger.debug("action was a string — converting to dict: %r", action_raw)
        action_raw = {"predicate": action_raw, "object": ""}
    if isinstance(condition_raw, str):
        logger.debug("condition was a string — converting to dict: %r", condition_raw)
        condition_raw = {"text": condition_raw, "type": "none"}

    # Ensure we have dicts (not lists or other types).
    if not isinstance(subject_raw, dict):
        subject_raw = {}
    if not isinstance(action_raw, dict):
        action_raw = {}
    if not isinstance(condition_raw, dict):
        condition_raw = {}

    # --- Step 2: Normalize enum string values ---
    # The LLM may return role values in various capitalizations.
    # We map them to the canonical lowercase form that LegalRole expects.
    role_str = subject_raw.get("role", "other")
    role_enum = coerce_legal_role(role_str)

    # Condition type normalization — same principle.
    cond_type_str = condition_raw.get("type", "none")
    cond_type_enum = coerce_condition_type(cond_type_str)

    # --- Step 3: Build the candidate dict ---
    candidate = {
        "subject": {
            "text": str(subject_raw.get("text", "")).strip(),
            "role": role_enum,
        },
        "action": {
            "predicate": str(action_raw.get("predicate", "")).strip(),
            "object": str(action_raw.get("object", "")).strip(),
        },
        "condition": {
            "text": str(condition_raw.get("text", "")).strip(),
            "type": cond_type_enum,
        },
    }

    # --- Step 4: Pydantic validation ---
    # This catches any remaining schema violations (wrong types, missing
    # required fields, etc.) and raises ValidationError if the data
    # cannot be coerced.
    triplet = LegalTriplet.model_validate(candidate)

    # --- Step 5: Post-validation normalization ---
    # Additional cleanup that Pydantic doesn't enforce:
    # - If condition text is empty or whitespace-only, set type to NONE
    #   (an empty condition shouldn't have a non-NONE type).
    # - If condition type is NONE but there is condition text, try to
    #   keep the text (the type assignment was wrong but the text may
    #   still be useful).
    normalized_condition = triplet.condition
    if normalized_condition.text.strip() == "":
        # No meaningful condition text — reset type to NONE for consistency.
        if normalized_condition.type != ConditionType.NONE:
            logger.debug(
                "Resetting condition type from %s to NONE (empty text)",
                normalized_condition.type.value,
            )
        normalized_condition = Condition(text="", type=ConditionType.NONE)

    # Build the final triplet with normalized condition.
    # Since LegalTriplet is frozen, we create a new instance.
    triplet = LegalTriplet(
        subject=triplet.subject,
        action=triplet.action,
        condition=normalized_condition,
    )

    return triplet


# -------------------------------------------------------------------------
# Enum Coercion Helpers
# -------------------------------------------------------------------------


def coerce_legal_role(raw: str) -> LegalRole:
    """Coerce a raw string to a ``LegalRole`` enum value.

    Handles common LLM output variations: case differences, underscores
    vs spaces, partial matches.  Falls back to ``LegalRole.OTHER`` if
    the string cannot be matched.

    Args:
        raw:  Raw role string from the LLM (e.g., ``"OBLIGOR"``,
              ``"right_holder"``, ``"Right Holder"``).

    Returns:
        The matching ``LegalRole`` enum value.
    """
    if not raw:
        return LegalRole.OTHER

    normalized = raw.strip().lower().replace(" ", "_").replace("-", "_")

    # Direct lookup by value.
    try:
        return LegalRole(normalized)
    except ValueError:
        pass

    # Fuzzy match: check if the raw string contains a known role keyword.
    fuzzy_map = {
        "obligor": LegalRole.OBLIGOR,
        "oblig": LegalRole.OBLIGOR,
        "right": LegalRole.RIGHT_HOLDER,
        "holder": LegalRole.RIGHT_HOLDER,
        "prohibit": LegalRole.PROHIBITED_PARTY,
        "indemnif": LegalRole.INDEMNIFYING_PARTY,
    }
    for keyword, role in fuzzy_map.items():
        if keyword in normalized:
            logger.debug(
                "Fuzzy-matched role '%s' -> %s (keyword: '%s')",
                raw,
                role.value,
                keyword,
            )
            return role

    logger.debug("Could not coerce role '%s' — falling back to OTHER", raw)
    return LegalRole.OTHER


def coerce_condition_type(raw: str) -> ConditionType:
    """Coerce a raw string to a ``ConditionType`` enum value.

    Handles case differences, underscores vs spaces, and common
    LLM output variations.  Falls back to ``ConditionType.NONE``
    if the string cannot be matched.

    Args:
        raw:  Raw condition type string (e.g., ``"TEMPORAL"``,
              ``"Trigger"``, ``"exception"``, ``"none"``).

    Returns:
        The matching ``ConditionType`` enum value.
    """
    if not raw:
        return ConditionType.NONE

    normalized = raw.strip().lower().replace(" ", "_").replace("-", "_")

    # Direct lookup by value.
    try:
        return ConditionType(normalized)
    except ValueError:
        pass

    # Fuzzy match for common LLM variations.
    fuzzy_map = {
        "temporal": ConditionType.TEMPORAL,
        "time": ConditionType.TEMPORAL,
        "trigger": ConditionType.TRIGGER,
        "event": ConditionType.TRIGGER,
        "conditional": ConditionType.TRIGGER,
        "except": ConditionType.EXCEPTION,
        "exception": ConditionType.EXCEPTION,
        "carve": ConditionType.EXCEPTION,
        "none": ConditionType.NONE,
        "null": ConditionType.NONE,
        "empty": ConditionType.NONE,
        "no": ConditionType.NONE,
    }
    for keyword, ctype in fuzzy_map.items():
        if keyword in normalized:
            logger.debug(
                "Fuzzy-matched condition type '%s' -> %s (keyword: '%s')",
                raw,
                ctype.value,
                keyword,
            )
            return ctype

    logger.debug(
        "Could not coerce condition type '%s' — falling back to NONE",
        raw,
    )
    return ConditionType.NONE


# -------------------------------------------------------------------------
# Fallback Construction
# -------------------------------------------------------------------------


def build_fallback_triplet(
    clause: str,
    error: str,
) -> LegalTriplet:
    """Build a minimal fallback triplet when extraction fails completely.

    Constructs a ``LegalTriplet`` with empty/default values for all
    fields.  This allows downstream processing (batch extraction,
    evaluation, error analysis) to continue even when the LLM fails
    on a particular clause.

    The fallback triplet preserves the original clause text in the
    log but does not embed it in the triplet (the ``LegalTriplet``
    model in ``schema.py`` does not have a ``clause`` field).  The
    original clause and error context are logged at WARNING level
    for post-hoc debugging.

    Args:
        clause:  The original clause text that failed extraction.
        error:   Human-readable error description for logging.

    Returns:
        A ``LegalTriplet`` with empty Subject, Action, and Condition
        (all fields at their sensible defaults).
    """
    logger.warning(
        "Using fallback triplet for clause (len=%d): %s | Error: %s",
        len(clause),
        clause[:100] + ("..." if len(clause) > 100 else ""),
        error,
    )

    return LegalTriplet(
        subject=Subject(
            text="",
            role=LegalRole.OTHER,
        ),
        action=Action(
            predicate="",
            object="",
        ),
        condition=Condition(
            text="",
            type=ConditionType.NONE,
        ),
    )
