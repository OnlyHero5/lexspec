"""
标注用的 LLM 响应解析器
=======================
将原始 LLM 文本响应解析为 LegalTriplet 对象。
处理多种输出格式（纯 JSON、Markdown 代码块等），
并清理常见模型产物。

导出：
  - strip_gemma_artifacts: 清理 Gemma 模型常见输出产物
  - parse_llm_response:      将原始 LLM 响应解析为 Optional[LegalTriplet]
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
    """去除 Gemma 模型输出中的常见非 JSON 产物。

    Gemma（尤其 31B）有时即使提示要求仅 JSON，
    仍会输出 Markdown 格式内容。本函数在尝试 JSON 提取前
    尽可能清理这些内容。

    清理规则：
      - 跳过空行
      - 去除项目符号（*、-、•、1.）
      - 去除 "Sentence:" / "Input:" 回显行
      - 跳过不含 JSON 字符的长散文行

    参数：
        text: 原始 LLM 响应文本。

    返回：
        清理后的文本。
    """
    lines = text.split("\n")
    cleaned_lines: List[str] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        # 去除项目符号："* text"、"- text"、"* (bullet) text"、"1. text"
        bullet_match = re.match(
            r"^(?:\*|\-|•|\d+[.\)]\s*)(.*)$", stripped
        )
        if bullet_match:
            stripped = bullet_match.group(1).strip()

        # 去除 "Sentence:" / "Input:" / "Output:" 回显行。
        echo_match = re.match(
            r'^(?:Sentence|Input|Output|Clause)\s*:?\s*["\']?(.*)["\']?\s*$',
            stripped,
            re.IGNORECASE,
        )
        if echo_match:
            inner = echo_match.group(1).strip()
            if inner.startswith("{"):
                # 回显内容恰为 JSON — 保留 JSON 部分。
                stripped = inner
            else:
                # 回显了输入句子 — 跳过该行。
                continue

        # 跳过不含 JSON 字符的长散文行。
        if "{" not in stripped and "}" not in stripped:
            if len(stripped) > 120 and '"' not in stripped:
                continue

        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def parse_llm_response(response: str) -> Optional[LegalTriplet]:
    """将 LLM 文本响应解析为 LegalTriplet。

    按优先级依次尝试四种解析策略：
      1. 纯 JSON 对象：    {"subject": {...}, "action": {...}, ...}
      2. Markdown 代码块： ```json ... ```
      3. 正则提取 JSON：  在文本中定位首个平衡的 {} 对象
      4. JSON 数组：      [{"subject": {...}}, ...] -> 取首元素

    解析前用 strip_gemma_artifacts() 预处理响应。

    参数：
        response: 原始 LLM 响应文本。

    返回：
        解析成功返回 LegalTriplet，否则 None。
    """
    if not response or not response.strip():
        logger.warning("Empty LLM response during annotation parsing")
        return None

    cleaned = response.strip()

    # --- 预处理：去除 Gemma 常见产物 ---
    cleaned = strip_gemma_artifacts(cleaned)

    # 策略 1：去除 Markdown 代码围栏后直接解析。
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    fence_match = re.search(fence_pattern, cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
        logger.debug("Extracted JSON from markdown code fence for annotation")

    # 策略 2：直接 JSON 解析。
    try:
        data = json.loads(cleaned)
        result = coerce_to_triplet(data)
        if result is not None:
            return result
    except json.JSONDecodeError:
        pass

    # 策略 3：正则匹配首个平衡花括号的 JSON 对象。
    # 支持最多 3 层嵌套，足以覆盖 LegalTriplet。
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

    # 策略 4：尝试解析为 JSON 数组并取首元素。
    try:
        data = json.loads(cleaned)
        if isinstance(data, list) and len(data) > 0:
            return coerce_to_triplet(data[0])
    except json.JSONDecodeError:
        pass

    logger.warning("Could not parse LLM annotation response into LegalTriplet")
    return None
