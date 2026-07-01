"""
约束配置访问器 —— 从 constraints.yaml 读取的唯一真实来源。

``configs/constraints.yaml`` 中的所有实验参数必须通过本模块读取。
配置缺失或无效时立即抛出异常；禁止静默回退到硬编码默认值。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)

F1_WEIGHT_KEYS = ("subject_text", "subject_role", "predicate", "object", "condition")

PHENOMENON_QUOTA_KEYS = {
    "passive": "passive",
    "conditional": "conditional",
    "relative": "relative_clause",
    "long_distance": "long_distance",
    "negation": "negation",
}

REQUIRED_VALIDATION_KEYS = (
    "condition_overlap",
    "subject_match",
    "object_match",
    "long_distance_tokens",
    "long_distance_mdd",
)


@dataclass(frozen=True)
class NormalizationConfig:
    """文本规范化开关集合（来自 ``constraints.yaml`` 的 ``normalization`` 节）。

    字段:
        remove_articles: 是否去除名词短语前导冠词（the/a/an）。
        lemmatize: 是否将词形还原为词典形式后再比较。
        number_normalization: 是否统一数字写法（如 five ↔ 5）。
        use_party_aliases: 是否应用 ``party_alias_mappings`` 中的当事方别名。
    """

    remove_articles: bool
    lemmatize: bool
    number_normalization: bool
    use_party_aliases: bool


def load_constraints_config(config_path: str = "configs/constraints.yaml") -> Dict[str, Any]:
    """加载完整的约束配置 YAML 并返回字典。

    参数:
        config_path: ``constraints.yaml`` 文件路径。

    返回:
        含 ``f1_weights``、``validation``、``normalization``、
        ``phenomenon_thresholds`` 等全部配置节的字典。

    异常:
        FileNotFoundError: 配置文件不存在。
        ValueError: 文件为空或 YAML 解析结果为空。
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"约束配置文件未找到: '{config_path}'。请确认 configs/constraints.yaml 存在。"
        )
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if config is None:
        raise ValueError(f"约束配置文件 '{config_path}' 为空。")
    return config


def _require_section(config: Mapping[str, Any], key: str, config_path: str) -> Dict[str, Any]:
    section = config.get(key)
    if not isinstance(section, dict):
        raise KeyError(
            f"Missing or invalid '{key}' section in '{config_path}'. "
            "All constraint parameters must be defined in constraints.yaml."
        )
    return section


def get_f1_weights(
    config: Mapping[str, Any],
    config_path: str = "configs/constraints.yaml",
) -> Dict[str, float]:
    """读取加权 F1 各字段权重并校验其和为 1.0。

    参数:
        config: 已由 ``load_constraints_config`` 加载的配置字典。
        config_path: 仅用于错误消息中的路径展示。

    返回:
        键为 ``subject_text``、``subject_role``、``predicate``、
        ``object``、``condition`` 的浮点权重字典。

    异常:
        KeyError: ``f1_weights`` 节缺失或缺少必需键。
        ValueError: 各权重之和不等于 1.0。
    """
    section = _require_section(config, "f1_weights", config_path)
    missing = [key for key in F1_WEIGHT_KEYS if key not in section]
    if missing:
        raise KeyError(
            f"Incomplete 'f1_weights' in '{config_path}'. Missing keys: {missing}"
        )

    weights = {key: float(section[key]) for key in F1_WEIGHT_KEYS}
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"'f1_weights' in '{config_path}' must sum to 1.0; got {total:.6f}"
        )
    logger.info("Loaded F1 weights from %s", config_path)
    return weights


def get_normalization_config(
    config: Mapping[str, Any],
    config_path: str = "configs/constraints.yaml",
) -> NormalizationConfig:
    """从配置字典构建文本规范化开关对象。

    参数:
        config: 已加载的约束配置字典。
        config_path: 仅用于错误消息中的路径展示。

    返回:
        含四个布尔开关的 ``NormalizationConfig`` 实例。

    异常:
        KeyError: ``normalization`` 节缺失或缺少必需键。
    """
    section = _require_section(config, "normalization", config_path)
    required = ("remove_articles", "lemmatize", "number_normalization", "use_party_aliases")
    missing = [key for key in required if key not in section]
    if missing:
        raise KeyError(
            f"Incomplete 'normalization' in '{config_path}'. Missing keys: {missing}"
        )
    return NormalizationConfig(
        remove_articles=bool(section["remove_articles"]),
        lemmatize=bool(section["lemmatize"]),
        number_normalization=bool(section["number_normalization"]),
        use_party_aliases=bool(section["use_party_aliases"]),
    )


def get_party_alias_mappings(
    config: Mapping[str, Any],
    config_path: str = "configs/constraints.yaml",
) -> Dict[str, List[str]]:
    """读取当事方规范名到表面形式别名的映射。

    参数:
        config: 已加载的约束配置字典。
        config_path: 仅用于错误消息中的路径展示。

    返回:
        字典，键为规范当事方名（如 ``"Seller"``），值为别名列表
        （如 ``["the seller", "vendor"]``）。

    异常:
        KeyError: 缺少 ``party_alias_mappings`` 节。
        TypeError: 某当事方的别名列表格式非法。
    """
    section = config.get("party_alias_mappings")
    if section is None:
        raise KeyError(
            f"Missing 'party_alias_mappings' section in '{config_path}'. "
            "Define canonical party names and alias lists (may be empty dict entries)."
        )
    if not isinstance(section, dict):
        raise TypeError(
            f"'party_alias_mappings' in '{config_path}' must be a mapping, got {type(section)}"
        )

    result: Dict[str, List[str]] = {}
    for canonical, aliases in section.items():
        if isinstance(aliases, list):
            result[str(canonical)] = [str(alias) for alias in aliases]
        elif isinstance(aliases, str):
            result[str(canonical)] = [aliases]
        else:
            raise TypeError(
                f"Invalid alias list for '{canonical}' in '{config_path}': {aliases!r}"
            )
    return result


def get_validation_thresholds(
    config: Mapping[str, Any],
    config_path: str = "configs/constraints.yaml",
) -> Dict[str, float]:
    """读取校验器与错误分析使用的距离/相似度阈值。

    参数:
        config: 已加载的约束配置字典。
        config_path: 仅用于错误消息中的路径展示。

    返回:
        含以下键的字典：
          - ``condition_overlap``: 条件 IoU/Jaccard 最低接受值；
          - ``subject_match`` / ``object_match``: 主语/宾语模糊匹配阈值；
          - ``long_distance_tokens``: 长距离依存判定的最小词元距离；
          - ``long_distance_mdd``: 语料现象检测用的平均依存距离阈值。

    异常:
        KeyError: ``validation`` 节缺失或缺少必需键。
    """
    section = _require_section(config, "validation", config_path)
    missing = [key for key in REQUIRED_VALIDATION_KEYS if key not in section]
    if missing:
        raise KeyError(
            f"Incomplete 'validation' in '{config_path}'. Missing keys: {missing}"
        )
    return {
        "condition_overlap": float(section["condition_overlap"]),
        "subject_match": float(section["subject_match"]),
        "object_match": float(section["object_match"]),
        "long_distance_tokens": float(section["long_distance_tokens"]),
        "long_distance_mdd": float(section["long_distance_mdd"]),
    }


def get_corpus_sampling_config(
    config: Mapping[str, Any],
    config_path: str = "configs/constraints.yaml",
) -> Dict[str, Any]:
    """读取语料构建的默认采样参数。

    参数:
        config: 已加载的约束配置字典。
        config_path: 仅用于错误消息中的路径展示。

    返回:
        含 ``target_count_default``（目标测试集规模）与
        ``random_seed``（随机种子）的字典。

    异常:
        KeyError: ``corpus_sampling`` 节或必需键缺失。
    """
    section = _require_section(config, "corpus_sampling", config_path)
    if "target_count_default" not in section:
        raise KeyError(
            f"Missing 'corpus_sampling.target_count_default' in '{config_path}'."
        )
    if "random_seed" not in section:
        raise KeyError(f"Missing 'corpus_sampling.random_seed' in '{config_path}'.")
    return {
        "target_count_default": int(section["target_count_default"]),
        "random_seed": int(section["random_seed"]),
    }


def get_phenomenon_quotas(
    config: Mapping[str, Any],
    target_count: int,
    config_path: str = "configs/constraints.yaml",
) -> Dict[str, int]:
    """将现象比例阈值转换为各现象类别的最低条款数量。

    参数:
        config: 已加载的约束配置字典。
        target_count: 目标测试集总条数；各现象配额为
            ``ceil(比例 × target_count)``，且至少为 1。
        config_path: 仅用于错误消息中的路径展示。

    返回:
        现象池名 → 最低条数 的字典，键包括 ``passive``、
        ``conditional``、``relative_clause``、``long_distance``、
        ``negation``。

    异常:
        ValueError: ``target_count`` 非正数。
        KeyError: ``phenomenon_thresholds`` 节缺失或缺少必需键。
    """
    if target_count <= 0:
        raise ValueError("target_count must be positive for stratified sampling")

    section = _require_section(config, "phenomenon_thresholds", config_path)
    missing = [key for key in PHENOMENON_QUOTA_KEYS if key not in section]
    if missing:
        raise KeyError(
            f"Incomplete 'phenomenon_thresholds' in '{config_path}'. Missing keys: {missing}"
        )

    quotas = {
        pool_key: max(1, ceil(float(section[yaml_key]) * target_count))
        for yaml_key, pool_key in PHENOMENON_QUOTA_KEYS.items()
    }
    logger.info(
        "Phenomenon quotas for target_count=%d from %s: %s",
        target_count,
        config_path,
        quotas,
    )
    return quotas


def normalize_for_comparison(
    text: str,
    config_path: str = "configs/constraints.yaml",
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> str:
    """按 YAML 规范化规则对文本做评估/共识比较前的预处理。

    参数:
        text: 待规范化的原始文本片段。
        config_path: 当 ``config`` 为 None 时用于加载约束 YAML 的路径。
        config: 可选的已加载配置字典；提供时可避免重复读文件。

    返回:
        经去冠词、词形还原、数字统一及（可选）当事方别名替换后的
        规范化字符串，供 F1 计算与标注共识使用。
    """
    from src.evaluation.text_normalizer import normalize

    cfg = config if config is not None else load_constraints_config(config_path)
    norm_cfg = get_normalization_config(cfg, config_path)
    aliases: Optional[Dict[str, List[str]]] = None
    if norm_cfg.use_party_aliases:
        aliases = get_party_alias_mappings(cfg, config_path)

    return normalize(
        text,
        remove_articles=norm_cfg.remove_articles,
        lemmatize=norm_cfg.lemmatize,
        number_normalize=norm_cfg.number_normalization,
        party_aliases=aliases,
    )
