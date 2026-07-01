"""
三元组级归一化与当事方别名加载。

所有设置均从 configs/constraints.yaml 加载 — 无硬编码默认值。
"""

from __future__ import annotations

from typing import Optional, Dict, List

from src.extraction.schema import LegalTriplet
from src.evaluation.text_normalizer import normalize, NUMBER_WORDS  # 再导出
from src.utils.constraints import (
    get_normalization_config,
    get_party_alias_mappings,
    load_constraints_config,
    normalize_for_comparison,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_party_aliases(constraints_path: str = "configs/constraints.yaml") -> Dict[str, List[str]]:
    """在 normalization.use_party_aliases 启用时加载当事方别名映射。"""
    config = load_constraints_config(constraints_path)
    norm_cfg = get_normalization_config(config, constraints_path)
    if not norm_cfg.use_party_aliases:
        return {}
    return get_party_alias_mappings(config, constraints_path)


def normalize_triplet(
    triplet: LegalTriplet,
    party_aliases: Optional[Dict[str, List[str]]] = None,
    constraints_path: str = "configs/constraints.yaml",
) -> LegalTriplet:
    """归一化 LegalTriplet 中全部文本字段以便公平比较。"""
    from src.extraction.schema import Subject, Action, Condition

    config = load_constraints_config(constraints_path)
    norm_cfg = get_normalization_config(config, constraints_path)
    aliases = party_aliases
    if aliases is None and norm_cfg.use_party_aliases:
        aliases = get_party_alias_mappings(config, constraints_path)

    def norm_text(text: str) -> str:
        return normalize(
            text,
            remove_articles=norm_cfg.remove_articles,
            lemmatize=norm_cfg.lemmatize,
            number_normalize=norm_cfg.number_normalization,
            party_aliases=aliases,
        )

    return LegalTriplet(
        subject=Subject(text=norm_text(triplet.subject.text), role=triplet.subject.role),
        action=Action(
            predicate=norm_text(triplet.action.predicate),
            object=norm_text(triplet.action.object),
        ),
        condition=Condition(
            text=norm_text(triplet.condition.text),
            type=triplet.condition.type,
        ),
    )


__all__ = [
    "normalize",
    "normalize_for_comparison",
    "normalize_triplet",
    "load_party_aliases",
    "NUMBER_WORDS",
]
