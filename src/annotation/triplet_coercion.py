"""
三元组强制转换工具
==================
将原始 LLM JSON 输出强制转换为 LegalTriplet 兼容字典，
并从文本推断条件类型。

供标注流水线（llm_annotator、reviewer）
与纠错流水线（reflexion 响应解析器）共用。

导出：
  - coerce_to_triplet:       将原始解析 JSON 转为 Optional[LegalTriplet]
  - normalize_to_canonical:  将标注式 JSON 转为 LegalTriplet 格式
  - infer_condition_type:    从条件文本推断条件类型（规范版本）
"""

from __future__ import annotations

from typing import Optional

from src.extraction.schema import LegalTriplet
from src.utils.logging import get_logger

logger = get_logger(__name__)


def coerce_to_triplet(data) -> Optional[LegalTriplet]:
    """将解析后的 JSON 数据强制转换为 LegalTriplet。

    支持两种输入格式：
      1. 规范格式：{"subject": {...}, "action": {...}, "condition": {...}}
      2. 标注格式：{"predicate": "...", "subject": {...}, "object": {...}, ...}

    参数：
        data: JSON 解析结果，可为 dict 或 list。

    返回：
        强制转换成功返回 LegalTriplet，否则 None。
    """
    if isinstance(data, list):
        if len(data) == 0:
            return None
        data = data[0]

    if not isinstance(data, dict):
        return None

    normalized = normalize_to_canonical(data)

    try:
        return LegalTriplet.model_validate(normalized)
    except Exception as exc:
        logger.debug(
            "LegalTriplet validation failed: %s. Data keys: %s",
            exc, list(data.keys()),
        )
        return None


def normalize_to_canonical(data: dict) -> dict:
    """将标注式 JSON 规范化为 LegalTriplet 标准格式。

    标注提示可能产生：
      {
        "predicate": "<动词>",
        "subject": {"text": "...", "role": "obligor|right_holder|..."},
        "object": {"text": "...", "role": "direct_object|..."},
        "condition": "<文本或 null>"
      }

    LegalTriplet 期望：
      {
        "subject": {"text": "...", "role": "<LegalRole>"},
        "action": {"predicate": "...", "object": "..."},
        "condition": {"text": "...", "type": "<ConditionType>"}
      }

    本方法在两种格式间映射，处理 None 条件、
    字符串与 dict 宾语、缺失字段等边界情况。

    参数：
        data: LLM 输出的标注式 dict。

    返回：
        与 LegalTriplet.model_validate() 兼容的 dict。
    """
    result: dict = {}

    # --- 主语 ---
    if "subject" in data and isinstance(data["subject"], dict):
        result["subject"] = {
            "text": str(data["subject"].get("text", "")),
            "role": str(data["subject"].get("role", "other")),
        }
    elif "subject" in data and isinstance(data["subject"], str):
        result["subject"] = {"text": data["subject"], "role": "other"}
    else:
        result["subject"] = {"text": "", "role": "other"}

    # --- 动作 ---
    action: dict = {}
    if "action" in data and isinstance(data["action"], dict):
        action["predicate"] = str(data["action"].get("predicate", ""))
        action["object"] = str(data["action"].get("object", ""))
    else:
        action["predicate"] = str(data.get("predicate", ""))
        obj = data.get("object", "")
        if isinstance(obj, dict):
            action["object"] = str(obj.get("text", ""))
        else:
            action["object"] = str(obj)
    result["action"] = action

    # --- 条件 ---
    condition: dict = {"text": "", "type": "none"}
    raw_condition = data.get("condition")
    if raw_condition is None or raw_condition == "" or raw_condition == "null":
        pass  # 无条件 — 使用默认值。
    elif isinstance(raw_condition, str):
        condition["text"] = raw_condition
        condition["type"] = infer_condition_type(raw_condition)
    elif isinstance(raw_condition, dict):
        condition["text"] = str(raw_condition.get("text", ""))
        condition["type"] = str(raw_condition.get("type", "none"))
    else:
        logger.debug("Unknown condition format: %s", type(raw_condition))
    result["condition"] = condition

    return result


def infer_condition_type(text: str) -> str:
    """从条件从句文本推断条件类型。

    使用条件文本开头的词汇标记，将条件分类为
    时间型、触发型或例外型。当 LLM 未返回显式条件类型时
    用作启发式回退。

    参数：
        text: 条件从句文本。

    返回：
        "temporal"、"trigger"、"exception" 或 "none" 之一。
    """
    text_lower = text.lower().strip()
    if not text_lower:
        return "none"

    # 时间标记：有时间界限的条件。
    temporal_markers = [
        "within", "after", "before", "upon", "when",
        "during", "until", "on or before", "no later than",
        "as of", "as from", "commencing", "following",
        "from time to time", "at any time",
    ]
    for marker in temporal_markers:
        if text_lower.startswith(marker):
            return "temporal"

    # 例外标记：义务范围的除外条款。
    exception_markers = [
        "unless", "except", "other than", "save as",
        "save for", "but for", "with the exception of",
    ]
    for marker in exception_markers:
        if text_lower.startswith(marker):
            return "exception"

    # 触发标记：事件型条件。
    trigger_markers = [
        "if", "in the event that", "in the event of",
        "in case", "in case of", "should", "provided that",
        "on condition that", "so long as", "as long as",
        "subject to", "conditioned upon",
    ]
    for marker in trigger_markers:
        if text_lower.startswith(marker):
            return "trigger"

    # 未匹配标记但文本非空 — 默认为触发型
    # （法律文本中多数条件从句为事件触发）。
    return "trigger"
