"""
LLM 请求重试 —— 指数退避与错误处理
===================================

封装 OpenAI SDK 调用的重试逻辑，支持指数退避、
瞬时错误恢复和 token 用量日志记录。
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import time

from openai import (
    APIError,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
    APIStatusError,
)

from src.utils.logging import get_logger

logger = get_logger(__name__)


def send_request_with_retry(
    config: Any,
    client: Any,
    messages: list[dict[str, str]],
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    """Dispatch a chat completion request with retry + backoff.

    This is the single code path for all completion requests.  It
    encapsulates the retry loop so that ``complete()`` and
    ``complete_structured()`` only differ in how they build the
    ``response_format`` argument.

    Retry strategy:
    - Catch ``APIConnectionError``, ``APITimeoutError``,
      ``RateLimitError``, and generic ``APIError`` / ``APIStatusError``.
    - On each failure, sleep for ``2^i`` seconds where ``i`` is the
      attempt number (0-indexed), giving backoff intervals of 1s, 2s,
      4s, ... up to ``2^(max_retries-1)``.
    - Log a warning on each retry attempt so operators can detect
      persistent connectivity issues.
    - After exhausting ``max_retries``, log an error and raise
      ``RuntimeError`` with the chain of original exceptions preserved.

    Args:
        config:           ``ClientConfig`` with model, generation params,
                          and reliability settings.
        client:           An ``openai.OpenAI`` instance.
        messages:         List of message dicts with 'role' and 'content'.
        response_format:  Optional response_format dict for structured output.

    Returns:
        The text content of the first choice.

    Raises:
        RuntimeError:  After all retries are exhausted.
    """
    last_exception: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            # --- Build the API call parameters ---
            # All generation parameters come from ClientConfig so every
            # call uses the same settings.  We use keyword arguments to
            # make it clear which parameter is which.
            kwargs: Dict[str, Any] = {
                "model": config.model,
                "messages": messages,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "seed": config.seed,
            }

            # Only include response_format if specified — the base
            # complete() method does not set it.
            if response_format is not None:
                kwargs["response_format"] = response_format

            # Include extra_body for provider-specific options (e.g.,
            # llama.cpp server settings like cache_prompt, top_k, etc.).
            if config.extra_body is not None:
                kwargs["extra_body"] = config.extra_body

            logger.debug(
                "Sending request (attempt %d/%d): model=%s, messages_len=%d",
                attempt + 1,
                config.max_retries + 1,
                config.model,
                len(messages),
            )

            # --- Execute the API call ---
            response = client.chat.completions.create(**kwargs)

            # --- Extract the response text ---
            # With temperature=0.0 there is exactly one choice.
            msg = response.choices[0].message
            content = msg.content or ""

            # Some models (e.g. Gemma 4 with thinking) may leave content
            # empty while putting text in reasoning_content.
            if not content.strip():
                reasoning = getattr(msg, "reasoning_content", None)
                if reasoning:
                    logger.debug(
                        "Using reasoning_content fallback (content was empty)"
                    )
                    content = reasoning

            # The API may return None for content in edge cases (e.g.,
            # content filter triggered).  Treat this as an empty string
            # rather than crashing so the caller's JSON parser handles it.
            if not content:
                logger.warning(
                    "LLM returned empty content on attempt %d",
                    attempt + 1,
                )

            # Log token usage for cost/performance tracking.
            if response.usage is not None:
                logger.debug(
                    "Request complete: prompt_tokens=%d, completion_tokens=%d, "
                    "total_tokens=%d",
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response.usage.total_tokens,
                )

            return content

        except (APIConnectionError, APITimeoutError) as exc:
            # --- Transient network errors ---
            # Connection refused, DNS failure, or request timeout.
            # These are likely temporary — retry with backoff.
            last_exception = exc
            logger.warning(
                "Network error on attempt %d/%d: %s",
                attempt + 1,
                config.max_retries + 1,
                exc,
            )
            if attempt < config.max_retries:
                backoff = 2 ** attempt  # 1s, 2s, 4s, ...
                logger.info("Retrying in %ds (exponential backoff)...", backoff)
                time.sleep(backoff)

        except RateLimitError as exc:
            # --- Server-side rate limiting ---
            # llama.cpp may apply rate limits.  Wait and retry.
            last_exception = exc
            logger.warning(
                "Rate limit hit on attempt %d/%d: %s",
                attempt + 1,
                config.max_retries + 1,
                exc,
            )
            if attempt < config.max_retries:
                backoff = 2 ** attempt
                logger.info("Retrying in %ds (exponential backoff)...", backoff)
                time.sleep(backoff)

        except (APIError, APIStatusError) as exc:
            # --- Server errors (5xx) or client errors (4xx) ---
            # Retry on 5xx (server fault).  For 4xx (bad request), the
            # retry is unlikely to help but we attempt once in case the
            # issue is a transient server state misclassified as 4xx.
            last_exception = exc
            logger.warning(
                "API error on attempt %d/%d (status=%s): %s",
                attempt + 1,
                config.max_retries + 1,
                getattr(exc, "status_code", "?"),
                exc,
            )
            if attempt < config.max_retries:
                backoff = 2 ** attempt
                logger.info("Retrying in %ds (exponential backoff)...", backoff)
                time.sleep(backoff)

    # --- All retries exhausted ---
    # We preserve the original exception chain for debugging.
    logger.error(
        "All %d retry attempts exhausted for model=%s. Last error: %s",
        config.max_retries + 1,
        config.model,
        last_exception,
    )
    raise RuntimeError(
        f"LLM request failed after {config.max_retries + 1} attempts. "
        f"Last error: {last_exception}"
    ) from last_exception
