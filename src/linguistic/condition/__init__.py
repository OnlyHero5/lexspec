"""
条件从句边界提取
=====================================
通过识别 advcl+mark 模式并从完整子树跨度提取，
从 UD 依存树中提取条件/例外/时间从句。

子模块：
  _marker_config:   回退分类体系、YAML 配置加载、标记词列表
  _extractor:       主提取接口（extract、extract_all、classify）
  _overlap:         条件跨度重叠计算
"""

from __future__ import annotations

from src.linguistic.condition._marker_config import (
    _load_markers as _load_markers_fn,
    _parse_markers_section as _parse_markers_section_fn,
)
from src.linguistic.condition._extractor import (
    extract as _extract_fn,
    extract_all as _extract_all_fn,
    _classify_condition as _classify_condition_fn,
)
from src.linguistic.condition._overlap import (
    compute_condition_overlap as _compute_condition_overlap_fn,
    is_condition_in_main_clause as _is_condition_in_main_clause_fn,
)


class ConditionExtractor:
    """从 UD 依存树提取并分类条件从句。

    从 configs/constraints.yaml 加载条件标记词分类体系，
    使用 advcl+mark UD 模式识别条件从句边界。
    根据标记词与法律领域分类体系将条件分为
    TRIGGER、TEMPORAL 或 EXCEPTION。

    校验器（步骤 5）使用本模块以：
      1. 检查大语言模型是否正确识别条件的存在/缺失。
      2. 校验条件从句边界（跨度准确性）。
      3. 分类错误类型：条件遗漏、过度抽取、边界错误。

    使用示例::

        extractor = ConditionExtractor("configs/constraints.yaml")
        spans = extractor.extract_all(tree, predicate_idx)
        if spans:
            print(f"Found {spans[0].condition_type} condition: {spans[0].text}")
    """

    # 从子模块导入 —— 绑定为实例/静态方法：
    _load_markers = _load_markers_fn
    _parse_markers_section = staticmethod(_parse_markers_section_fn)
    extract = _extract_fn
    extract_all = _extract_all_fn
    _classify_condition = _classify_condition_fn
    compute_condition_overlap = staticmethod(_compute_condition_overlap_fn)
    is_condition_in_main_clause = staticmethod(_is_condition_in_main_clause_fn)

    def __init__(self, constraints_path: str = "configs/constraints.yaml"):
        """从约束 YAML 加载条件标记词并初始化提取器。

        参数:
            constraints_path: 约束配置文件路径，须含 ``condition_markers`` 节。

        异常:
            FileNotFoundError: 配置文件不存在。
            KeyError: 缺少 ``condition_markers`` 或必需子键。
        """
        self._markers: dict = {}
        self._marker_list: list = []
        self._load_markers(constraints_path)

    @property
    def marker_list(self) -> list:
        """返回全部已知条件标记词的扁平列表。

        返回:
            字符串列表，供 ``find_advcl_with_mark()`` 识别 advcl 条件从句。
        """
        return list(self._marker_list)
