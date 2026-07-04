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
    temperature: Optional[float] = None,
) -> str:
    """以重试与退避策略发送聊天补全请求。

    所有补全请求的单一代码路径。封装重试循环，使 ``complete()`` 与
    ``complete_structured()`` 仅在 ``response_format`` 参数的构建方式上不同。

    重试策略:
    - 捕获 ``APIConnectionError``、``APITimeoutError``、
      ``RateLimitError`` 及通用 ``APIError`` / ``APIStatusError``。
    - 每次失败后休眠 ``2^i`` 秒，``i`` 为尝试序号（从 0 起），
      退避间隔为 1s、2s、4s……直至 ``2^(max_retries-1)``。
    - 每次重试记录警告日志，便于运维人员发现持续性连接问题。
    - 耗尽 ``max_retries`` 后记录错误并抛出 ``RuntimeError``，
      保留原始异常链。

    参数:
        config:           含模型、生成参数与可靠性设置的 ``ClientConfig``。
        client:           ``openai.OpenAI`` 实例。
        messages:         含 'role' 与 'content' 的消息字典列表。
        response_format:  结构化输出的可选 response_format 字典。

    返回:
        第一个选项的文本内容。

    异常:
        RuntimeError:  所有重试耗尽后抛出。
    """
    last_exception: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            # --- 构建 API 调用参数 ---
            # 所有生成参数来自 ClientConfig，确保每次调用使用相同设置。
            # 使用关键字参数以明确各参数含义。
            kwargs: Dict[str, Any] = {
                "model": config.model,
                "messages": messages,
                "temperature": config.temperature if temperature is None else temperature,
                "max_tokens": config.max_tokens,
                "seed": config.seed,
            }

            # 仅在指定时包含 response_format——基础 complete() 方法不设置此项。
            if response_format is not None:
                kwargs["response_format"] = response_format

            # 包含 extra_body 以支持提供商特定选项（如 llama.cpp 的
            # cache_prompt、top_k 等服务器设置）。
            if config.extra_body is not None:
                kwargs["extra_body"] = config.extra_body

            logger.debug(
                "Sending request (attempt %d/%d): model=%s, messages_len=%d",
                attempt + 1,
                config.max_retries + 1,
                config.model,
                len(messages),
            )

            # --- 执行 API 调用 ---
            response = client.chat.completions.create(**kwargs)

            # --- 提取响应文本 ---
            # temperature=0.0 时恰好有一个选项。
            msg = response.choices[0].message
            content = msg.content or ""

            # 部分模型（如带 thinking 的 Gemma 4）可能 content 为空，
            # 而将文本放在 reasoning_content 中。
            if not content.strip():
                reasoning = getattr(msg, "reasoning_content", None)
                if reasoning:
                    logger.debug(
                        "Using reasoning_content fallback (content was empty)"
                    )
                    content = reasoning

            # API 在边缘情况下可能返回 None 内容（如内容过滤触发）。
            # 视为空字符串而非崩溃，由调用方的 JSON 解析器处理。
            if not content:
                logger.warning(
                    "LLM returned empty content on attempt %d",
                    attempt + 1,
                )

            # 记录 token 用量以供成本/性能追踪。
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
            # --- 瞬时网络错误 ---
            # 连接拒绝、DNS 失败或请求超时。可能为暂时性问题，退避后重试。
            last_exception = exc
            logger.warning(
                "Network error on attempt %d/%d: %s",
                attempt + 1,
                config.max_retries + 1,
                exc,
            )
            if attempt < config.max_retries:
                backoff = 2 ** attempt  # 1s、2s、4s……
                logger.info("Retrying in %ds (exponential backoff)...", backoff)
                time.sleep(backoff)

        except RateLimitError as exc:
            # --- 服务端速率限制 ---
            # llama.cpp 可能施加速率限制。等待后重试。
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
            # --- 服务器错误（5xx）或客户端错误（4xx）---
            # 对 5xx（服务端故障）重试。对 4xx（错误请求）重试通常无效，
            # 但若问题为被误分类为 4xx 的瞬时服务端状态，仍尝试一次。
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

    # --- 所有重试耗尽 ---
    # 保留原始异常链以便调试。
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
