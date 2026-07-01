"""
PolarityDetector 的配置加载：YAML 解析与查找表构建。

从约束 YAML 加载情态规则。
无硬编码回退 —— 缺失或格式错误的配置视为错误。
"""

from __future__ import annotations

from typing import Dict
from pathlib import Path

import yaml

from src.extraction.schema import LegalRole
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_rules(self, constraints_path: str) -> None:
    """从 YAML 配置加载情态规则。

    解析 modality_rules 节，定义：
      obligor:        {aux_verbs: [shall, must, will], negated: false}
      right_holder:   {aux_verbs: [may, can], negated: false}
      prohibited_party: {aux_verbs: [shall, may, must], negated: true}

    构建两个内部结构：
      1. _modality_rules: 供内省的完整规则字典
      2. _lookup: 快速 (aux_lemma, is_negated) -> LegalRole 映射

    参数：
        constraints_path: 约束 YAML 文件路径。

    抛出：
        FileNotFoundError: 配置文件不存在时。
        yaml.YAMLError: YAML 格式错误时。
        KeyError: 缺少或不完整 ``modality_rules`` 时。
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

    if not config_data or "modality_rules" not in config_data:
        raise KeyError(
            f"No 'modality_rules' section in '{config_path}'. "
            "The constraints YAML must define obligor / right_holder / prohibited_party rules."
        )

    self._modality_rules = config_data["modality_rules"]

    required_roles = ("obligor", "right_holder", "prohibited_party")
    for role_key in required_roles:
        if role_key not in self._modality_rules:
            raise KeyError(
                f"Required modality rule '{role_key}' not found in '{config_path}'. "
                "All three roles (obligor, right_holder, prohibited_party) are required."
            )
        rule = self._modality_rules[role_key]
        if not rule.get("aux_verbs"):
            raise ValueError(
                f"Modality rule '{role_key}' in '{config_path}' has empty aux_verbs list."
            )

    logger.info("Loaded modality rules from %s", config_path)

    _build_lookup(self)


def _build_lookup(self) -> None:
    """构建 (aux_lemma, is_negated) -> LegalRole 查找表。

    将助动词与否定状态的每种组合映射到
    对应法律角色，供 O(1) 分类。
    """
    role_map = {
        "obligor": LegalRole.OBLIGOR,
        "right_holder": LegalRole.RIGHT_HOLDER,
        "prohibited_party": LegalRole.PROHIBITED_PARTY,
    }

    self._lookup.clear()

    for role_key, role_enum in role_map.items():
        rule = self._modality_rules[role_key]
        aux_verbs = rule.get("aux_verbs", [])
        negated = rule.get("negated", False)

        for aux in aux_verbs:
            aux_lower = aux.lower().strip()
            if aux_lower:
                self._lookup[(aux_lower, negated)] = role_enum

    logger.debug(
        "Built modality lookup: %d (aux, negated) -> role mappings",
        len(self._lookup),
    )
