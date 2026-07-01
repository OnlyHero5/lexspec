"""
Reflexion Response Parser
==========================
Parses LLM re-extraction responses during Reflexion correction into
LegalTriplet objects. Normalizes annotation-style output to the
canonical format expected by the schema.

Exported:
  - parse_llm_response:   Parse raw LLM response into Optional[LegalTriplet]
  - _validate_and_return: Validate parsed dict against LegalTriplet schema
  - _normalize_annotation_format: Map annotation-style JSON to canonical format
"""

from __future__ import annotations

import json
import re
from typing import Optional

from src.extraction.schema import LegalTriplet
from src.annotation.triplet_coercion import infer_condition_type
from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_llm_response(response: str) -> Optional[LegalTriplet]:
    """Parse the LLM's text response into a LegalTriplet.

    Handles several common LLM output formats:
      1. Pure JSON:         {"subject": {...}, "action": {...}, ...}
      2. Markdown-fenced:   ```json ... ```
      3. Array-wrapped:     [{"subject": {...}, ...}] -- takes first element
      4. With prefix text:  extracts the first JSON object found

    Uses Pydantic's model_validate for type-safe deserialization.

    Args:
        response: Raw string response from the LLM.

    Returns:
        Parsed LegalTriplet, or None if no valid JSON object was found
        or if the parsed data fails Pydantic validation.
    """
    if not response or not response.strip():
        logger.warning("Empty LLM response during Reflexion parsing")
        return None

    # Attempt 1: Strip markdown code fences (```json ... ```).
    cleaned = response.strip()
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    fence_match = re.search(fence_pattern, cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
        logger.debug("Extracted JSON from markdown code fence")

    # Attempt 2: Try direct JSON parsing of the cleaned string.
    try:
        data = json.loads(cleaned)
        return _validate_and_return(data)
    except json.JSONDecodeError:
        pass

    # Attempt 3: Search for the first JSON object in the response.
    json_object_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    for match in re.finditer(json_object_pattern, cleaned):
        try:
            data = json.loads(match.group())
            result = _validate_and_return(data)
            if result is not None:
                logger.debug("Found valid JSON object via regex extraction")
                return result
        except json.JSONDecodeError:
            continue

    # Attempt 4: Try parsing as a JSON array and take the first element.
    try:
        data = json.loads(cleaned)
        if isinstance(data, list) and len(data) > 0:
            return _validate_and_return(data[0])
    except json.JSONDecodeError:
        pass

    # All parsing attempts failed.
    logger.warning(
        "Could not parse LLM Reflexion response. First 200 chars: %.200s",
        cleaned,
    )
    return None


def _validate_and_return(data) -> Optional[LegalTriplet]:
    """Validate a parsed dict against the LegalTriplet schema.

    Handles both the canonical LegalTriplet format and a few common
    alternative structures that LLMs sometimes produce.

    Args:
        data: A dict parsed from JSON, expected to match LegalTriplet
              structure.

    Returns:
        LegalTriplet if validation succeeds, None otherwise.
    """
    if not isinstance(data, dict):
        return None

    # Normalize: if the LLM used "predicate" instead of "action",
    # restructure the data to match the LegalTriplet schema.
    if "predicate" in data and "action" not in data:
        data = _normalize_annotation_format(data)
    elif "object" in data and "action" not in data:
        data = _normalize_annotation_format(data)

    try:
        return LegalTriplet.model_validate(data)
    except Exception as exc:
        logger.debug("LegalTriplet validation failed: %s", exc)
        return None


def _normalize_annotation_format(data: dict) -> dict:
    """Normalize annotation-style JSON to LegalTriplet format.

    The annotation prompt format (from configs/prompts.yaml) produces:
      {
        "predicate": "<verb>",
        "subject": {"text": "...", "role": "..."},
        "object": {"text": "...", "role": "..."},
        "condition": "<text or null>"
      }

    The LegalTriplet schema expects:
      {
        "subject": {"text": "...", "role": "..."},
        "action": {"predicate": "...", "object": "..."},
        "condition": {"text": "...", "type": "..."}
      }

    This method maps between the two formats.
    Uses the canonical infer_condition_type from src.annotation.triplet_coercion.

    Args:
        data: Annotation-style dict.

    Returns:
        Dict compatible with LegalTriplet schema.
    """
    result: dict = {}

    # Map subject: pass through (same structure).
    if "subject" in data and isinstance(data["subject"], dict):
        result["subject"] = data["subject"]

    # Map action from separate predicate + object keys.
    action: dict = {}
    if "predicate" in data:
        action["predicate"] = str(data["predicate"])
    if "object" in data:
        # The object might be a dict {"text": "...", "role": "..."}
        # or a plain string. Extract the text portion.
        if isinstance(data["object"], dict):
            action["object"] = str(data["object"].get("text", ""))
        else:
            action["object"] = str(data["object"])
    result["action"] = action

    # Map condition: might be a string, None, or a dict.
    condition: dict = {"text": "", "type": "none"}
    raw_condition = data.get("condition")
    if raw_condition is None or raw_condition == "":
        pass  # Keep defaults (empty text, type=none)
    elif isinstance(raw_condition, str):
        condition["text"] = raw_condition
        # Use the canonical infer_condition_type from annotation module.
        condition["type"] = infer_condition_type(raw_condition)
    elif isinstance(raw_condition, dict):
        condition["text"] = str(raw_condition.get("text", ""))
        condition["type"] = str(raw_condition.get("type", "none"))
    result["condition"] = condition

    return result
