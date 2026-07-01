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
    """以鲁棒错误处理解析大语言模型 JSON 响应。

    大语言模型输出可能并非严格有效 JSON——JSON 块前后有多余文本、
    输出截断、Markdown 代码围栏、尾随逗号等。本方法按可靠性顺序
    尝试多种策略：

    1. **直接解析** —— 对原始响应执行 ``json.loads``。模型输出干净 JSON 时有效。

    2. **正则提取** —— 用贪婪花括号匹配器在响应中搜索 ``{...}`` 块。
       处理模型将 JSON 包裹在说明文字或 Markdown 围栏中的情况。

    3. **空字典** —— 完全失败时返回空字典。
       调用方（``extract()``）检测此情况并构建回退三元组。

    参数:
        response:  大语言模型的原始文本响应。

    返回:
        解析后的字典（完全失败时可能为空）。调用方负责对照
        ``LegalTriplet`` 模式校验字典。
    """
    if not response or not response.strip():
        logger.warning("LLM returned empty response")
        return {}

    # --- 策略 1：直接 JSON 解析 ---
    # 修剪空白后尝试将整个响应解析为 JSON。
    cleaned = response.strip()
    try:
        result = json.loads(cleaned)
        # 大语言模型可能返回 JSON 数组（prompts.yaml 格式所要求）。
        # 若是数组则取第一个元素——抽取器处理单个三元组。
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
        # 若既非字典也非列表，输出不可用。
        logger.warning(
            "Parsed JSON is not a dict or list (type=%s)",
            type(result).__name__,
        )
        return {}
    except (json.JSONDecodeError, ValueError) as exc:
        logger.debug("Direct JSON parse failed: %s — trying regex extraction", exc)

    # --- 策略 2：正则提取 {...} 块 ---
    # 模型有时将 JSON 包裹在 Markdown 代码围栏中，或在前后附加说明文字。
    # 使用贪婪正则查找最外层匹配的花括号对。
    try:
        # 找到第一个 '{'，再追踪花括号深度以定位匹配的 '}'。
        # 可处理 JSON 内的嵌套对象。
        result = _extract_json_object(cleaned)
        if result is not None:
            logger.debug("Regex JSON extraction succeeded")
            return result
    except (json.JSONDecodeError, ValueError) as exc:
        logger.debug("Regex JSON extraction failed: %s", exc)

    # --- 策略 3：Markdown 代码围栏提取 ---
    # 检查 ```json ... ``` 或 ``` ... ``` 块。部分模型
    # （尤其指令微调模型）即使被告知不要，仍默认用 Markdown 围栏包裹 JSON。
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
                pass  # 继续策略 4
    except Exception:
        pass  # 安全网——正则不应抛出，但保持防御性

    # --- 策略 4：完全失败 ---
    logger.warning("All JSON parsing strategies failed for response")
    return {}


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """用花括号匹配从文本中提取有效 JSON 对象 ``{...}``。

    逐字符扫描文本并追踪花括号深度，定位第一个完整的最外层 ``{...}`` 对。
    成功则返回解析后的字典，否则返回 ``None``。

    比简单正则更鲁棒，可正确处理嵌套花括号
    （如 JSON 字符串中的转义花括号或嵌套对象）。

    参数:
        text:  可能内含 JSON 对象的文本。

    返回:
        解析后的字典，提取失败则返回 None。
    """
    # 查找开括号。
    start_idx = text.find("{")
    if start_idx == -1:
        return None

    # 追踪花括号深度。从 1 开始，因已找到第一个 '{'。
    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start_idx:], start=start_idx):
        if escape_next:
            # 前一字符为反斜杠——当前字符被转义，
            # 不应解释为结构字符（如 JSON 字符串内的 \"）。
            escape_next = False
            continue

        if ch == "\\" and in_string:
            # 字符串内的反斜杠——下一字符被转义。
            escape_next = True
            continue

        if ch == '"' and not escape_next:
            # 切换字符串状态。未转义引号界定 JSON 字符串。
            in_string = not in_string
            continue

        if in_string:
            # 字符串内——结构花括号不计入深度。
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                # 找到匹配的闭括号。
                json_str = text[start_idx:i + 1]
                return json.loads(json_str)

    # 若循环结束而深度未归零，花括号不平衡——JSON 格式错误或截断。
    logger.debug("Brace extraction failed: unbalanced braces (depth=%d)", depth)
    return None
