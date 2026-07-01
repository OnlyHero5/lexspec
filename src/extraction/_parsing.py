"""
LLM 响应解析 —— 多策略 JSON 提取
================================

从大语言模型返回的原始文本中提取 JSON 对象，
使用多种回退策略处理不完美的模型输出。
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import json
import re

from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse the LLM JSON response with robust error handling.

    The LLM may produce output that is not strictly valid JSON —
    extra text before/after the JSON block, truncated output,
    markdown code fences, trailing commas, etc.  This method
    attempts multiple strategies in order of reliability:

    1. **Direct parse** — ``json.loads`` on the raw response.  Works
       when the model produces clean JSON.

    2. **Regex extraction** — Search for a ``{...}`` block in the
       response using a greedy brace matcher.  Handles cases where
       the model wraps JSON in explanatory text or markdown fences.

    3. **Empty dict** — Return an empty dict on complete failure.
       The caller (``extract()``) detects this and builds a fallback
       triplet.

    Args:
        response:  Raw text response from the LLM.

    Returns:
        Parsed dict (may be empty on complete failure).  The caller
        is responsible for validating the dict against the
        ``LegalTriplet`` schema.
    """
    if not response or not response.strip():
        logger.warning("LLM returned empty response")
        return {}

    # --- Strategy 1: Direct JSON parse ---
    # Strip whitespace and try parsing the entire response as JSON.
    cleaned = response.strip()
    try:
        result = json.loads(cleaned)
        # The LLM may return a JSON array (as the prompts.yaml format
        # requests).  If so, take the first element — the extractor
        # operates on single triplets.
        if isinstance(result, list):
            if len(result) > 0:
                logger.debug(
                    "LLM returned JSON array of %d elements — using first element",
                    len(result),
                )
                result = result[0]
            else:
                logger.warning("LLM returned an empty JSON array")
                return {}
        if isinstance(result, dict):
            logger.debug("Direct JSON parse succeeded")
            return result
        # If it's neither a dict nor a list, the output is unusable.
        logger.warning(
            "Parsed JSON is not a dict or list (type=%s)",
            type(result).__name__,
        )
        return {}
    except (json.JSONDecodeError, ValueError) as exc:
        logger.debug("Direct JSON parse failed: %s — trying regex extraction", exc)

    # --- Strategy 2: Regex extraction of {...} block ---
    # The model sometimes wraps JSON in markdown code fences or
    # prepends/appends explanatory text.  We use a greedy regex to
    # find the outermost matched brace pair.
    try:
        # Find the first '{' and then track brace depth to find the
        # matching '}'.  This handles nested objects within the JSON.
        result = _extract_json_object(cleaned)
        if result is not None:
            logger.debug("Regex JSON extraction succeeded")
            return result
    except (json.JSONDecodeError, ValueError) as exc:
        logger.debug("Regex JSON extraction failed: %s", exc)

    # --- Strategy 3: Markdown code fence extraction ---
    # Check for ```json ... ``` or ``` ... ``` blocks.  Some models
    # (especially instruction-tuned ones) default to wrapping JSON
    # in markdown fences even when told not to.
    try:
        fence_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?```",
            cleaned,
            re.DOTALL,
        )
        if fence_match:
            fence_content = fence_match.group(1).strip()
            try:
                result = json.loads(fence_content)
                if isinstance(result, list) and len(result) > 0:
                    result = result[0]
                if isinstance(result, dict):
                    logger.debug("JSON extracted from markdown code fence")
                    return result
            except (json.JSONDecodeError, ValueError):
                pass  # Fall through to Strategy 4
    except Exception:
        pass  # Safety net — regex shouldn't raise but be defensive

    # --- Strategy 4: Complete failure ---
    logger.warning("All JSON parsing strategies failed for response")
    return {}


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract a valid JSON object ``{...}`` from text using brace matching.

    Scans the text character by character, tracking brace depth, to
    find the first complete, outermost ``{...}`` pair.  Returns the
    parsed dict if successful, or ``None`` if no valid JSON object
    could be extracted.

    This is more robust than a simple regex because it correctly
    handles nested braces (e.g., JSON strings containing escaped
    braces, or nested objects).

    Args:
        text:  Text that may contain a JSON object somewhere within it.

    Returns:
        Parsed dict, or None if extraction fails.
    """
    # Find the opening brace.
    start_idx = text.find("{")
    if start_idx == -1:
        return None

    # Track brace depth.  Start at 1 because we found the first '{'.
    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start_idx:], start=start_idx):
        if escape_next:
            # The previous character was a backslash — this character
            # is escaped and should not be interpreted as a structural
            # character (e.g., \" inside a JSON string).
            escape_next = False
            continue

        if ch == "\\" and in_string:
            # Backslash inside a string — the next character is escaped.
            escape_next = True
            continue

        if ch == '"' and not escape_next:
            # Toggle string state.  Unescaped quotes delimit JSON strings.
            in_string = not in_string
            continue

        if in_string:
            # Inside a string — structural braces are not counted.
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # We found the matching closing brace.
                json_str = text[start_idx:i + 1]
                return json.loads(json_str)

    # If we exit the loop without depth hitting 0, the braces are
    # unbalanced — the JSON is malformed or truncated.
    logger.debug("Brace extraction failed: unbalanced braces (depth=%d)", depth)
    return None
