"""
Triplet Coercion Utilities
===========================
Functions for coercing raw LLM JSON output into LegalTriplet-compatible
dictionaries and inferring condition types from text.

These are shared across the annotation pipeline (llm_annotator, reviewer)
and the correction pipeline (reflexion response parser).

Exported:
  - coerce_to_triplet:       Convert raw parsed JSON data to Optional[LegalTriplet]
  - normalize_to_canonical:  Convert annotation-style JSON to LegalTriplet format
  - infer_condition_type:    Infer condition type from condition text (CANONICAL version)
"""

from __future__ import annotations

from typing import Optional

from src.extraction.schema import LegalTriplet
from src.utils.logging import get_logger

logger = get_logger(__name__)


def coerce_to_triplet(data) -> Optional[LegalTriplet]:
    """Coerce parsed JSON data into a LegalTriplet.

    Handles two input formats:
      1. Canonical format: {"subject": {...}, "action": {...}, "condition": {...}}
      2. Annotation format: {"predicate": "...", "subject": {...}, "object": {...}, ...}

    Args:
        data: JSON parse result, can be dict or list.

    Returns:
        LegalTriplet if coercion succeeds, None otherwise.
    """
    if isinstance(data, list):
        if len(data) == 0:
            return None
        data = data[0]

    if not isinstance(data, dict):
        return None

    normalized = normalize_to_canonical(data)

    try:
        return LegalTriplet.model_validate(normalized)
    except Exception as exc:
        logger.debug(
            "LegalTriplet validation failed: %s. Data keys: %s",
            exc, list(data.keys()),
        )
        return None


def normalize_to_canonical(data: dict) -> dict:
    """Normalize annotation-style JSON to LegalTriplet canonical format.

    The annotation prompt may produce:
      {
        "predicate": "<verb>",
        "subject": {"text": "...", "role": "obligor|right_holder|..."},
        "object": {"text": "...", "role": "direct_object|..."},
        "condition": "<text or null>"
      }

    LegalTriplet expects:
      {
        "subject": {"text": "...", "role": "<LegalRole>"},
        "action": {"predicate": "...", "object": "..."},
        "condition": {"text": "...", "type": "<ConditionType>"}
      }

    This method maps between the two formats, handling None condition values,
    string vs dict objects, missing fields, and other edge cases.

    Args:
        data: Annotation-style dict from LLM output.

    Returns:
        Dict compatible with LegalTriplet.model_validate().
    """
    result: dict = {}

    # --- subject ---
    if "subject" in data and isinstance(data["subject"], dict):
        result["subject"] = {
            "text": str(data["subject"].get("text", "")),
            "role": str(data["subject"].get("role", "other")),
        }
    elif "subject" in data and isinstance(data["subject"], str):
        result["subject"] = {"text": data["subject"], "role": "other"}
    else:
        result["subject"] = {"text": "", "role": "other"}

    # --- action ---
    action: dict = {}
    if "action" in data and isinstance(data["action"], dict):
        action["predicate"] = str(data["action"].get("predicate", ""))
        action["object"] = str(data["action"].get("object", ""))
    else:
        action["predicate"] = str(data.get("predicate", ""))
        obj = data.get("object", "")
        if isinstance(obj, dict):
            action["object"] = str(obj.get("text", ""))
        else:
            action["object"] = str(obj)
    result["action"] = action

    # --- condition ---
    condition: dict = {"text": "", "type": "none"}
    raw_condition = data.get("condition")
    if raw_condition is None or raw_condition == "" or raw_condition == "null":
        pass  # No condition -- use defaults.
    elif isinstance(raw_condition, str):
        condition["text"] = raw_condition
        condition["type"] = infer_condition_type(raw_condition)
    elif isinstance(raw_condition, dict):
        condition["text"] = str(raw_condition.get("text", ""))
        condition["type"] = str(raw_condition.get("type", "none"))
    else:
        logger.debug("Unknown condition format: %s", type(raw_condition))
    result["condition"] = condition

    return result


def infer_condition_type(text: str) -> str:
    """Infer condition type from the condition clause text.

    Uses lexical markers (at the start of the condition text) to classify
    the condition as temporal, trigger, or exception. This is a heuristic
    fallback used when the LLM does not return an explicit condition type.

    Args:
        text: Condition clause text.

    Returns:
        One of "temporal", "trigger", "exception", or "none".
    """
    text_lower = text.lower().strip()
    if not text_lower:
        return "none"

    # Temporal markers: time-bound conditions.
    temporal_markers = [
        "within", "after", "before", "upon", "when",
        "during", "until", "on or before", "no later than",
        "as of", "as from", "commencing", "following",
        "from time to time", "at any time",
    ]
    for marker in temporal_markers:
        if text_lower.startswith(marker):
            return "temporal"

    # Exception markers: carve-outs from obligations.
    exception_markers = [
        "unless", "except", "other than", "save as",
        "save for", "but for", "with the exception of",
    ]
    for marker in exception_markers:
        if text_lower.startswith(marker):
            return "exception"

    # Trigger markers: event-based conditions.
    trigger_markers = [
        "if", "in the event that", "in the event of",
        "in case", "in case of", "should", "provided that",
        "on condition that", "so long as", "as long as",
        "subject to", "conditioned upon",
    ]
    for marker in trigger_markers:
        if text_lower.startswith(marker):
            return "trigger"

    # No marker matched but text is non-empty -- default to trigger
    # (most condition clauses in legal text are event-triggered).
    return "trigger"
