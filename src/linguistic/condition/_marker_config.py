"""
标记词配置：YAML 加载与标记节解析。

从约束 YAML 加载条件标记词分类体系。
无硬编码回退 —— 缺失或格式错误的配置视为错误。
"""

from __future__ import annotations

from typing import Dict, List
from pathlib import Path

import yaml

from src.extraction.schema import ConditionType
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_markers(self, constraints_path: str) -> None:
    """从 YAML 配置加载条件标记词分类体系。

    解析 constraints.yaml 的 condition_markers 节。
    各类别（trigger、temporal、exception）含 ``mark_words`` 列表。

    所有标记词小写并索引到扁平字典，供提取时 O(1) 查找。

    参数：
        constraints_path: YAML 配置文件路径。

    抛出：
        FileNotFoundError: 配置文件不存在时。
        yaml.YAMLError: YAML 格式错误时。
        KeyError: 缺少或为空 ``condition_markers`` 时。
    """
    config_path = Path(constraints_path)
    if not config_path.is_absolute():
        resolved = config_path.resolve()
        if not resolved.exists():
            alt_paths = [
                Path.cwd() / "configs" / "constraints.yaml",
                Path(__file__).parent.parent.parent.parent / "configs" / "constraints.yaml",
            ]
            for alt in alt_paths:
                if alt.exists():
                    config_path = alt
                    break

    if not config_path.exists():
        raise FileNotFoundError(
            f"Constraints config not found at '{constraints_path}'. "
            "Create it from configs/constraints.yaml or check your working directory."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    if not config_data or "condition_markers" not in config_data:
        raise KeyError(
            f"No 'condition_markers' section in '{config_path}'. "
            "The constraints YAML must define trigger/temporal/exception marker lists."
        )

    markers_section = config_data["condition_markers"]
    self._markers = _parse_markers_section(markers_section)

    if not self._markers:
        raise ValueError(
            f"Condition marker taxonomy is empty in '{config_path}'. "
            "At least one marker must be defined per category."
        )

    self._marker_list = list(self._markers.keys())

    logger.info(
        "Condition marker taxonomy loaded: %d markers total "
        "(trigger=%d, temporal=%d, exception=%d)",
        len(self._markers),
        sum(1 for v in self._markers.values() if v == ConditionType.TRIGGER),
        sum(1 for v in self._markers.values() if v == ConditionType.TEMPORAL),
        sum(1 for v in self._markers.values() if v == ConditionType.EXCEPTION),
    )


def _parse_markers_section(section: dict) -> Dict[str, ConditionType]:
    """将 condition_markers YAML 节解析为查找字典。

    参数：
        section: YAML 中的 condition_markers 字典。

    返回：
        小写标记文本 -> ConditionType 的字典。

    抛出：
        KeyError: 缺少必需类别时。
    """
    result: Dict[str, ConditionType] = {}

    category_map = {
        "trigger": ConditionType.TRIGGER,
        "temporal": ConditionType.TEMPORAL,
        "exception": ConditionType.EXCEPTION,
    }

    for category_name, condition_type in category_map.items():
        if category_name not in section:
            raise KeyError(
                f"Category '{category_name}' not found in condition_markers config. "
                "All three categories (trigger, temporal, exception) are required."
            )

        category_data = section[category_name]

        if isinstance(category_data, dict):
            words = category_data.get("mark_words", [])
        elif isinstance(category_data, list):
            words = category_data
        else:
            raise TypeError(
                f"Unexpected format for category '{category_name}': "
                f"expected dict or list, got {type(category_data).__name__}."
            )

        for word in words:
            word_lower = word.lower().strip()
            if word_lower and word_lower not in result:
                result[word_lower] = condition_type

    return result
