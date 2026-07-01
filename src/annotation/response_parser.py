"""
LLM Response Parser for Annotation
===================================
Parses raw LLM text responses into LegalTriplet objects.
Handles multiple output formats (pure JSON, markdown-fenced, etc.)
and cleans up common model artifacts.

Exported:
  - strip_gemma_artifacts: Clean up common Gemma model output artifacts
  - parse_llm_response:    Parse raw LLM response string into Optional[LegalTriplet]
"""

from __future__ import annotations

import json
import re
from typing import Optional, List

from src.extraction.schema import LegalTriplet
from src.annotation.triplet_coercion import coerce_to_triplet
from src.utils.logging import get_logger

logger = get_logger(__name__)


def strip_gemma_artifacts(text: str) -> str:
    """Strip common non-JSON artifacts from Gemma model output.

    Gemma (especially the 31B variant) sometimes outputs markdown-formatted
    content even when the prompt explicitly asks for JSON only. This function
    cleans up as much of that as possible before JSON extraction is attempted.

    Cleanup rules:
      - Skip blank lines
      - Strip bullet markers (*, -, * (bullet), 1.)
      - Strip "Sentence:" / "Input:" echo lines
      - Skip long prose lines that contain no JSON characters

    Args:
        text: Raw LLM response text.

    Returns:
        Cleaned-up text.
    """
    lines = text.split("\n")
    cleaned_lines: List[str] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        # Strip bullet markers: "* text", "- text", "* (bullet) text", "1. text"
        bullet_match = re.match(
            r"^(?:\*|\-|•|\d+[.\)]\s*)(.*)$", stripped
        )
        if bullet_match:
            stripped = bullet_match.group(1).strip()

        # Strip "Sentence:" / "Input:" / "Output:" echo lines.
        echo_match = re.match(
            r'^(?:Sentence|Input|Output|Clause)\s*:?\s*["\']?(.*)["\']?\s*$',
            stripped,
            re.IGNORECASE,
        )
        if echo_match:
            inner = echo_match.group(1).strip()
            if inner.startswith("{"):
                # Echo content happens to be JSON -- keep the JSON part.
                stripped = inner
            else:
                # Echoed the input sentence -- skip this line.
                continue

        # Skip long prose lines that contain no JSON characters.
        if "{" not in stripped and "}" not in stripped:
            if len(stripped) > 120 and '"' not in stripped:
                continue

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def parse_llm_response(response: str) -> Optional[LegalTriplet]:
    """Parse an LLM text response into a LegalTriplet.

    Attempts four parsing strategies in priority order:
      1. Pure JSON object:    {"subject": {...}, "action": {...}, ...}
      2. Markdown code block: ```json ... ```
      3. Regex JSON extract:  locate the first balanced {} object in text
      4. JSON array:          [{"subject": {...}}, ...] -> take first element

    Pre-processes the response with strip_gemma_artifacts() before parsing.

    Args:
        response: Raw LLM response text.

    Returns:
        LegalTriplet if parsing succeeds, None otherwise.
    """
    if not response or not response.strip():
        logger.warning("Empty LLM response during annotation parsing")
        return None

    cleaned = response.strip()

    # --- Preprocessing: strip common Gemma artifacts ---
    cleaned = strip_gemma_artifacts(cleaned)

    # Strategy 1: Strip markdown code fences, then parse directly.
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    fence_match = re.search(fence_pattern, cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
        logger.debug("Extracted JSON from markdown code fence for annotation")

    # Strategy 2: Direct JSON parse.
    try:
        data = json.loads(cleaned)
        result = coerce_to_triplet(data)
        if result is not None:
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 3: Regex match for the first balanced-brace JSON object.
    # Supports up to 3 levels of nesting, sufficient for LegalTriplet.
    json_pattern = r"\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}"
    for match in re.finditer(json_pattern, cleaned):
        try:
            data = json.loads(match.group())
            result = coerce_to_triplet(data)
            if result is not None:
                logger.debug("Found valid JSON object via regex extraction")
                return result
        except json.JSONDecodeError:
            continue

    # Strategy 4: Try parsing as a JSON array and take the first element.
    try:
        data = json.loads(cleaned)
        if isinstance(data, list) and len(data) > 0:
            return coerce_to_triplet(data[0])
    except json.JSONDecodeError:
        pass

    logger.warning("Could not parse LLM annotation response into LegalTriplet")
    return None
