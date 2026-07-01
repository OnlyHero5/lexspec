"""
标注统计生成器
==============
在三元组层面（完全一致 vs 部分/完全不一致）与
字段层面（各字段一致率）计算标注者间一致性指标。
并按语言现象统计分歧分布，用于诊断。
"""

from __future__ import annotations

from typing import List, Dict, Any

from src.extraction.schema import LegalTriplet
from src.utils.logging import get_logger

from src.annotation.consensus import field_level_consensus
from src.annotation.field_helpers import (
    FIELD_SPEC,
    _classify_disagreement_phenomenon,
    _triplets_equal,
)

logger = get_logger(__name__)


def generate_annotation_stats(
    annotations_qwen: List[LegalTriplet],
    annotations_gemma: List[LegalTriplet],
    gold: List[LegalTriplet],
) -> Dict[str, Any]:
    """生成标注统计报告。

    在三元组层面（完全一致 vs 部分/完全不一致）与
    字段层面（各字段一致率）计算标注者间一致性。
    并按语言现象统计分歧分布，用于诊断。

    参数：
        annotations_qwen: Qwen 模型标注列表。
        annotations_gemma: Gemma 模型标注列表。
        gold: 最终金标准三元组列表。

    返回：
        统计 dict，键包括：
          - "total_clauses": int，已标注条款总数
          - "full_agreement_count": int，Qwen 与 Gemma
            产出完全相同三元组的条款数（6 字段均一致）
          - "full_agreement_rate": float，full_agreement_count / total_clauses
          - "field_agreement": dict[str, float]，6 个字段各自的
            一致率
          - "disagreement_by_phenomenon": dict[str, int]，按语言现象
            统计的分歧类型计数（passive_voice、
            condition_boundary、negation、role_mismatch 等）
          - "qwen_gold_agreement_rate": float，Qwen 标注与
            最终金标准的完全一致率
          - "gemma_gold_agreement_rate": float，Gemma 标注与
            最终金标准的完全一致率
    """
    # 校验输入长度一致。
    total = len(annotations_qwen)
    if len(annotations_gemma) != total or len(gold) != total:
        logger.error(
            "Mismatched annotation list lengths: qwen=%d, gemma=%d, gold=%d",
            len(annotations_qwen), len(annotations_gemma), len(gold),
        )
        raise ValueError(
            "Annotation lists must have the same length. "
            f"Got qwen={len(annotations_qwen)}, gemma={len(annotations_gemma)}, "
            f"gold={len(gold)}"
        )

    if total == 0:
        logger.warning("generate_annotation_stats called with empty lists")
        return {
            "total_clauses": 0,
            "full_agreement_count": 0,
            "full_agreement_rate": 0.0,
            "field_agreement": {f: 0.0 for f, _, _ in FIELD_SPEC},
            "disagreement_by_phenomenon": {},
            "qwen_gold_agreement_rate": 0.0,
            "gemma_gold_agreement_rate": 0.0,
        }

    # --- 完全一致（三元组层面）---
    # 仅当 6 个字段全部一致时视为三元组完全一致。
    full_agreement_count = 0
    # 各字段一致计数。
    field_agreement_counts = {field_name: 0 for field_name, _, _ in FIELD_SPEC}
    # 按现象统计的分歧计数。
    phenomenon_counts: Dict[str, int] = {}
    # 模型-金标准一致计数。
    qwen_gold_agreement = 0
    gemma_gold_agreement = 0

    for i in range(total):
        qwen_anno = annotations_qwen[i]
        gemma_anno = annotations_gemma[i]
        gold_triplet = gold[i]

        # 逐字段比较 Qwen 与 Gemma。
        consensus, disagreements = field_level_consensus(qwen_anno, gemma_anno)

        # 统计完全一致。
        if len(disagreements) == 0:
            full_agreement_count += 1

        # 统计各字段一致（每条款 6 个字段）。
        # 共识调用已给出不一致字段；其余视为一致。
        disagreed_field_names = {d["field"] for d in disagreements}
        for field_name, _, _ in FIELD_SPEC:
            if field_name not in disagreed_field_names:
                field_agreement_counts[field_name] += 1

        # 按现象统计分歧模式。
        for d in disagreements:
            phenomenon = _classify_disagreement_phenomenon(
                d["field"],
                qwen_anno,
                gemma_anno,
            )
            phenomenon_counts[phenomenon] = phenomenon_counts.get(phenomenon, 0) + 1

        # 统计模型-金标准一致（三元组层面）。
        if _triplets_equal(qwen_anno, gold_triplet):
            qwen_gold_agreement += 1
        if _triplets_equal(gemma_anno, gold_triplet):
            gemma_gold_agreement += 1

    # 由计数计算比率。
    field_agreement_rates: Dict[str, float] = {}
    for field_name, _, _ in FIELD_SPEC:
        field_agreement_rates[field_name] = (
            field_agreement_counts[field_name] / total if total > 0 else 0.0
        )

    stats = {
        "total_clauses": total,
        "full_agreement_count": full_agreement_count,
        "full_agreement_rate": full_agreement_count / total if total > 0 else 0.0,
        "field_agreement": field_agreement_rates,
        "disagreement_by_phenomenon": phenomenon_counts,
        "qwen_gold_agreement_rate": qwen_gold_agreement / total if total > 0 else 0.0,
        "gemma_gold_agreement_rate": gemma_gold_agreement / total if total > 0 else 0.0,
    }

    logger.info(
        "Annotation stats: %d clauses, full_agreement=%.1f%%, "
        "qwen_gold=%.1f%%, gemma_gold=%.1f%%",
        total,
        stats["full_agreement_rate"] * 100,
        stats["qwen_gold_agreement_rate"] * 100,
        stats["gemma_gold_agreement_rate"] * 100,
    )

    return stats
