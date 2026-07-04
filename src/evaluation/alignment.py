"""
按 clause_id 对齐金标、预测与测试集记录。

评估与错误分析须以金标 clause_id 顺序为准，避免仅靠行序对齐导致
静默错位评分。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.extraction.schema import LegalTriplet
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ClauseAlignmentError(ValueError):
    """clause_id 对齐失败。"""


def index_records_by_clause_id(
    records: List[Dict],
    *,
    label: str,
) -> Dict[str, Dict]:
    """将含 clause_id 的记录列表索引为 dict。"""
    index: Dict[str, Dict] = {}
    for rec in records:
        clause_id = rec.get("clause_id")
        if not clause_id:
            raise ClauseAlignmentError(f"{label}: record missing clause_id")
        if clause_id in index:
            raise ClauseAlignmentError(
                f"{label}: duplicate clause_id '{clause_id}'"
            )
        index[str(clause_id)] = rec
    return index


def align_to_gold_order(
    gold_records: List[Dict],
    other_records: List[Dict],
    *,
    other_label: str,
    strict: bool = True,
) -> List[Dict]:
    """按 gold_records 的 clause_id 顺序对齐 other_records。"""
    if not gold_records:
        return []

    other_index = index_records_by_clause_id(other_records, label=other_label)
    gold_ids = [str(r["clause_id"]) for r in gold_records]

    missing = [cid for cid in gold_ids if cid not in other_index]
    if missing:
        msg = (
            f"{other_label} missing {len(missing)} clause_id(s) present in gold "
            f"(first: {missing[:5]})"
        )
        if strict:
            raise ClauseAlignmentError(msg)
        logger.warning("%s — skipping missing rows", msg)

    extra = sorted(set(other_index) - set(gold_ids))
    if extra and strict:
        raise ClauseAlignmentError(
            f"{other_label} has {len(extra)} extra clause_id(s) not in gold "
            f"(first: {extra[:5]})"
        )

    aligned: List[Dict] = []
    for clause_id in gold_ids:
        if clause_id in other_index:
            aligned.append(other_index[clause_id])
    return aligned


def align_predictions_to_gold(
    gold_records: List[Dict],
    prediction_records: List[Dict],
    *,
    strict: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """按金标顺序对齐预测，返回 (aligned_predictions, gold_records)。"""
    aligned_preds = align_to_gold_order(
        gold_records,
        prediction_records,
        other_label="predictions",
        strict=strict,
    )
    if strict and len(aligned_preds) != len(gold_records):
        raise ClauseAlignmentError(
            f"Aligned predictions ({len(aligned_preds)}) != gold ({len(gold_records)})"
        )
    return aligned_preds, gold_records


def records_to_triplets(records: List[Dict]) -> List[LegalTriplet]:
    """从 JSONL 记录中提取 triplet 字段并校验为 LegalTriplet。"""
    triplets: List[LegalTriplet] = []
    for record in records:
        triplet_data = record.get("triplet", record)
        try:
            triplets.append(LegalTriplet.model_validate(triplet_data))
        except Exception as exc:
            logger.debug(
                "Failed to validate triplet for clause %s: %s",
                record.get("clause_id", "?"),
                exc,
            )
            triplets.append(LegalTriplet())
    return triplets
