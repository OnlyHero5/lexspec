"""
Reflexion 响应解析器
====================
在 Reflexion 纠错期间将 LLM 重新抽取的响应解析为
LegalTriplet 对象。将标注式输出规范化为
模式所期望的规范格式。

导出：
  - parse_llm_response:   将原始 LLM 响应解析为 Optional[LegalTriplet]
  - _validate_and_return: 将解析后的 dict 对照 LegalTriplet 模式校验
  - _normalize_annotation_format: 将标注式 JSON 映射为规范格式
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
    """将 LLM 文本响应解析为 LegalTriplet。

    处理多种常见 LLM 输出格式：
      1. 纯 JSON：         {"subject": {...}, "action": {...}, ...}
      2. Markdown 围栏：   ```json ... ```
      3. 数组包装：        [{"subject": {...}, ...}] — 取首元素
      4. 带前缀文本：      提取首个 JSON 对象

    使用 Pydantic 的 model_validate 做类型安全反序列化。

    参数：
        response: LLM 返回的原始字符串。

    返回：
        解析得到的 LegalTriplet；未找到有效 JSON 对象或
        Pydantic 校验失败时为 None。
    """
    if not response or not response.strip():
        logger.warning("Empty LLM response during Reflexion parsing")
        return None

    # 尝试 1：去除 Markdown 代码围栏（```json ... ```）。
    cleaned = response.strip()
    fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    fence_match = re.search(fence_pattern, cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
        logger.debug("Extracted JSON from markdown code fence")

    # 尝试 2：对清理后的字符串直接 JSON 解析。
    try:
        data = json.loads(cleaned)
        return _validate_and_return(data)
    except json.JSONDecodeError:
        pass

    # 尝试 3：在响应中搜索首个 JSON 对象。
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

    # 尝试 4：尝试解析为 JSON 数组并取首元素。
    try:
        data = json.loads(cleaned)
        if isinstance(data, list) and len(data) > 0:
            return _validate_and_return(data[0])
    except json.JSONDecodeError:
        pass

    # 全部解析尝试失败。
    logger.warning(
        "Could not parse LLM Reflexion response. First 200 chars: %.200s",
        cleaned,
    )
    return None


def _validate_and_return(data) -> Optional[LegalTriplet]:
    """将解析后的 dict 对照 LegalTriplet 模式校验。

    同时处理规范 LegalTriplet 格式与 LLM 有时产出的
    几种常见替代结构。

    参数：
        data: 自 JSON 解析的 dict，预期匹配 LegalTriplet
              结构。

    返回：
        校验成功返回 LegalTriplet，否则 None。
    """
    if not isinstance(data, dict):
        return None

    # 规范化：若 LLM 使用 "predicate" 而非 "action"，
    # 重组数据以匹配 LegalTriplet 模式。
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
    """将标注式 JSON 规范化为 LegalTriplet 格式。

    标注提示格式（来自 configs/prompts.yaml）产生：
      {
        "predicate": "<动词>",
        "subject": {"text": "...", "role": "..."},
        "object": {"text": "...", "role": "..."},
        "condition": "<文本或 null>"
      }

    LegalTriplet 模式期望：
      {
        "subject": {"text": "...", "role": "..."},
        "action": {"predicate": "...", "object": "..."},
        "condition": {"text": "...", "type": "..."}
      }

    本方法在两种格式间映射。
    条件类型推断使用 src.annotation.triplet_coercion 中的规范 infer_condition_type。

    参数：
        data: 标注式 dict。

    返回：
        与 LegalTriplet 模式兼容的 dict。
    """
    result: dict = {}

    # 映射 subject：结构相同，直接传递。
    if "subject" in data and isinstance(data["subject"], dict):
        result["subject"] = data["subject"]

    # 由独立的 predicate + object 键映射 action。
    action: dict = {}
    if "predicate" in data:
        action["predicate"] = str(data["predicate"])
    if "object" in data:
        # object 可能是 dict {"text": "...", "role": "..."}
        # 或纯字符串。提取 text 部分。
        if isinstance(data["object"], dict):
            action["object"] = str(data["object"].get("text", ""))
        else:
            action["object"] = str(data["object"])
    result["action"] = action

    # 映射 condition：可能是字符串、None 或 dict。
    condition: dict = {"text": "", "type": "none"}
    raw_condition = data.get("condition")
    if raw_condition is None or raw_condition == "":
        pass  # 保留默认（空文本，type=none）
    elif isinstance(raw_condition, str):
        condition["text"] = raw_condition
        # 使用标注模块中的规范 infer_condition_type。
        condition["type"] = infer_condition_type(raw_condition)
    elif isinstance(raw_condition, dict):
        condition["text"] = str(raw_condition.get("text", ""))
        condition["type"] = str(raw_condition.get("type", "none"))
    result["condition"] = condition

    return result
