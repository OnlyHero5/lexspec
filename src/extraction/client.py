"""
大语言模型客户端封装 —— 兼容 OpenAI 的 HTTP API
================================================

封装 OpenAI SDK 调用，连接到远程 llama.cpp 服务器。处理指数退避重试、
超时控制和结构化 JSON 输出约束。

设计说明:
  - 使用 OpenAI Python SDK（openai>=1.0.0）指向 llama.cpp 的 /v1 端点
  - 所有生成参数集中在 ClientConfig 中，确保每次实验运行使用相同设置
  - 重试循环捕获瞬时故障（连接错误、速率限制、服务器错误）并采用指数退避
  - complete_structured 支持 json_object 模式和 JSON Schema 严格模式

使用示例::

    config = ClientConfig(base_url="http://server:8080/v1", model="qwen3.5-9b")
    client = LLMClient(config)
    response = client.complete("你是助手。", "说你好。")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from openai import OpenAI

from src.utils.logging import get_logger

from ._client_retry import send_request_with_retry

logger = get_logger(__name__)


# =============================================================================
# Client Configuration
# =============================================================================


@dataclass
class ClientConfig:
    """大语言模型客户端配置 —— 封装 llama.cpp 服务器的连接与生成参数。

    所有生成参数固定以保证确定性、可复现的输出。每个字段均注明其在
    推理流水线中的作用。

    默认值与 LexSpec configs/model.yaml 中 server 和 generation 段一致。
    可通过关键字参数覆盖以进行临时实验或标注模型切换。
    """

    # --- Connection ---
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "not-needed"

    # --- Model ---
    model: str = "qwen3.5-9b"

    # --- Generation (reproducibility-critical) ---
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int = 42

    # --- Reliability ---
    timeout: int = 60
    max_retries: int = 3

    # --- Structured output ---
    response_format: Optional[Dict[str, str]] = field(
        default_factory=lambda: {"type": "json_object"}
    )

    # --- Extension point for provider-specific options ---
    extra_body: Optional[Dict[str, Any]] = field(default=None)


# =============================================================================
# LLM Client
# =============================================================================


class LLMClient:
    """兼容 OpenAI 的大语言模型客户端 —— 用于 llama.cpp 推理。

    封装 OpenAI Python SDK 与本地部署的 llama.cpp 服务器通信。
    每个实例通过 ClientConfig 绑定一个模型（实验模型或标注模型）。

    功能:
      - 系统+用户消息的聊天补全请求
      - 通过 response_format 或 JSON Schema 的结构化 JSON 输出
      - 瞬时错误重试 + 指数退避
      - 超时控制
    """

    def __init__(self, config: ClientConfig):
        """Initialize the OpenAI client pointing at the llama.cpp endpoint.

        Creates an ``openai.OpenAI`` instance with the configured base URL
        and API key, then stores a copy of the config for reuse in each
        completion call.

        Args:
            config:  ``ClientConfig`` with endpoint, model identity, and all
                     generation parameters.
        """
        # --- Store configuration for reuse in complete() calls ---
        # We keep a reference so that generate-time parameters (temperature,
        # max_tokens, seed, etc.) are always drawn from the config that was
        # passed at construction time.
        self.config = config

        # --- Create the OpenAI SDK client ---
        # The SDK is pointed at llama.cpp's /v1 endpoint.  The api_key value
        # is arbitrary — llama.cpp ignores it, but the SDK requires a non-empty
        # string to avoid a validation error.
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            max_retries=0,  # We handle retries ourselves for backoff control
        )

        logger.info(
            "LLMClient initialized: model=%s, base_url=%s, temperature=%.1f, "
            "max_tokens=%d, seed=%d, timeout=%ds, max_retries=%d",
            self.config.model,
            self.config.base_url,
            self.config.temperature,
            self.config.max_tokens,
            self.config.seed,
            self.config.timeout,
            self.config.max_retries,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat completion request and return the response text.

        Builds a two-message conversation (system + user), calls the
        chat completions endpoint with parameters from ``ClientConfig``,
        and returns the model's text response.

        On transient errors (connection refused, 5xx, rate limit) the
        request is retried with exponential backoff.  After
        ``max_retries`` failures a ``RuntimeError`` is raised.

        Args:
            system_prompt:  System message defining the model's role
                            and task instructions.
            user_prompt:    User message carrying the actual task input
                            (e.g., the clause text to extract from).

        Returns:
            The text content of the first choice from the model's response.
            With ``temperature=0.0`` this is the single highest-probability
            generation.

        Raises:
            RuntimeError:  If all retry attempts are exhausted.  The
                           original exception chain is preserved via
                           ``raise ... from last_exception``.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return send_request_with_retry(
            config=self.config,
            client=self.client,
            messages=messages,
        )

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send a request with structured JSON output constraints.

        Like ``complete()`` but adds ``response_format`` to the API call
        so the model is steered toward producing valid JSON.

        Two modes are supported:

        1. **json_object mode** (default): Uses ``response_format={"type":
           "json_object"}``.  The model is instructed to output valid JSON
           but no schema is enforced at the API level.  This is the most
           broadly compatible option.

        2. **json_schema mode**: When ``json_schema`` is provided, it is
           passed as ``response_format={"type": "json_schema", "json_schema":
           {...}}``.  This tells the API to constrain generation to match
           the given JSON Schema.  Requires server-side support (available
           in newer llama.cpp builds with grammar-based sampling).

        Args:
            system_prompt:  System message with role and instructions.
            user_prompt:    User message with the task input.
            json_schema:    Optional JSON Schema dict for strict mode.
                            When provided, must contain ``name`` and
                            ``schema`` keys per the OpenAI structured
                            output specification.

        Returns:
            JSON string from the model's response.

        Raises:
            RuntimeError:  If all retry attempts are exhausted.
        """
        # --- Build response_format based on whether a schema is provided ---
        if json_schema is not None:
            # Use the JSON Schema mode for stricter generation constraints.
            # The schema dict should contain 'name' (a short identifier) and
            # 'schema' (the actual JSON Schema object).
            response_format = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
            logger.debug(
                "Using JSON Schema mode with schema name=%s",
                json_schema.get("name", "unnamed"),
            )
        else:
            # Fall back to basic json_object mode — the model is asked to
            # output JSON but no grammar-level enforcement is applied.
            response_format = self.config.response_format
            logger.debug("Using json_object mode (no schema enforcement)")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return send_request_with_retry(
            config=self.config,
            client=self.client,
            messages=messages,
            response_format=response_format,
        )
