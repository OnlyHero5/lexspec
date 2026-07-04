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
# 客户端配置
# =============================================================================


@dataclass
class ClientConfig:
    """大语言模型客户端配置 —— 封装 llama.cpp 服务器的连接与生成参数。

    字段:
        base_url: llama.cpp OpenAI 兼容 API 根地址（如 ``http://host:8080/v1``）。
        api_key: API 密钥占位符；llama.cpp 不校验，但 SDK 要求非空。
        model: 服务端模型名（``--model`` 参数对应值）。
        temperature: 采样温度；``0.0`` 为贪心解码，保证可复现。
        max_tokens: 单次生成的最大 token 数。
        seed: 随机种子，与 ``temperature=0`` 配合保证确定性。
        timeout: 单次 HTTP 请求超时秒数。
        max_retries: 瞬时故障时的最大重试次数。
        response_format: 默认结构化输出格式（通常为 ``json_object``）。
        extra_body: 传给服务端的额外 JSON 字段（提供商扩展用）。

    默认值与 ``configs/model.yaml`` 中 ``server``、``generation`` 段一致。
    """

    # --- 连接 ---
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "not-needed"

    # --- 模型 ---
    model: str = "qwen3.5-9b"

    # --- 生成参数（可复现性关键）---
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int = 42

    # --- 可靠性 ---
    timeout: int = 60
    max_retries: int = 3

    # --- 结构化输出 ---
    response_format: Optional[Dict[str, str]] = field(
        default_factory=lambda: {"type": "json_object"}
    )

    # --- 提供商特定选项的扩展点 ---
    extra_body: Optional[Dict[str, Any]] = field(default=None)


# =============================================================================
# 大语言模型客户端
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
        """初始化指向 llama.cpp 端点的 OpenAI 客户端。

        使用配置的 base URL 与 API 密钥创建 ``openai.OpenAI`` 实例，
        并保存配置副本供每次补全调用复用。

        参数:
            config:  包含端点、模型标识及全部生成参数的 ``ClientConfig``。
        """
        # --- 保存配置供 complete() 调用复用 ---
        # 保留引用，使生成时参数（temperature、max_tokens、seed 等）
        # 始终来自构造时传入的配置。
        self.config = config

        # --- 创建 OpenAI SDK 客户端 ---
        # SDK 指向 llama.cpp 的 /v1 端点。api_key 值可任意填写——
        # llama.cpp 会忽略，但 SDK 要求非空字符串以避免校验错误。
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            max_retries=0,  # 自行处理重试以控制退避策略
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
    # 公开 API
    # -------------------------------------------------------------------------

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
    ) -> str:
        """发送聊天补全请求并返回响应文本。

        构建双消息对话（系统 + 用户），使用 ``ClientConfig`` 中的参数
        调用聊天补全端点，并返回模型的文本响应。

        对瞬时错误（连接拒绝、5xx、速率限制）采用指数退避重试。
        在 ``max_retries`` 次失败后抛出 ``RuntimeError``。

        参数:
            system_prompt:  定义模型角色与任务指令的系统消息。
            user_prompt:    承载实际任务输入的用户消息
                            （例如待抽取的条款文本）。

        返回:
            模型响应中第一个选项的文本内容。
            在 ``temperature=0.0`` 时为概率最高的单一生成结果。

        异常:
            RuntimeError:  所有重试尝试耗尽时抛出，原始异常链通过
                           ``raise ... from last_exception`` 保留。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return send_request_with_retry(
            config=self.config,
            client=self.client,
            messages=messages,
            temperature=temperature,
        )

    def complete_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[Dict[str, Any]] = None,
        *,
        temperature: Optional[float] = None,
    ) -> str:
        """发送带结构化 JSON 输出约束的请求。

        与 ``complete()`` 类似，但向 API 调用添加 ``response_format``，
        引导模型生成有效 JSON。

        支持两种模式：

        1. **json_object 模式**（默认）：使用 ``response_format={"type":
           "json_object"}``。模型被指示输出有效 JSON，但 API 层不强制模式。
           兼容性最广。

        2. **json_schema 模式**：提供 ``json_schema`` 时，传递
           ``response_format={"type": "json_schema", "json_schema": {...}}``。
           告知 API 将生成约束为符合给定 JSON Schema。需要服务端支持
           （新版 llama.cpp 支持基于语法的采样）。

        参数:
            system_prompt:  含角色与指令的系统消息。
            user_prompt:    含任务输入的用户消息。
            json_schema:    严格模式的可选 JSON Schema 字典。
                            提供时需包含 OpenAI 结构化输出规范要求的
                            ``name`` 与 ``schema`` 键。

        返回:
            模型响应中的 JSON 字符串。

        异常:
            RuntimeError:  所有重试尝试耗尽时抛出。
        """
        # --- 根据是否提供 schema 构建 response_format ---
        if json_schema is not None:
            # 使用 JSON Schema 模式以获得更严格的生成约束。
            # schema 字典应包含 'name'（短标识符）与 'schema'（实际 JSON Schema 对象）。
            response_format = {
                "type": "json_schema",
                "json_schema": json_schema,
            }
            logger.debug(
                "Using JSON Schema mode with schema name=%s",
                json_schema.get("name", "unnamed"),
            )
        else:
            # 回退到基本 json_object 模式——要求模型输出 JSON，
            # 但不进行语法级强制。
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
            temperature=temperature,
        )
