"""
Annotation Statistics Generator
================================
Computes inter-annotator agreement metrics at both the triplet level
(full agreement vs. partial/full disagreement) and the field level
(per-field agreement rates). Also computes disagreement distribution
by linguistic phenomenon for diagnostic purposes.
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
    """Generate annotation statistics report.

    Computes inter-annotator agreement metrics at both the triplet level
    (full agreement vs. partial/full disagreement) and the field level
    (per-field agreement rates). Also computes disagreement distribution
    by linguistic phenomenon for diagnostic purposes.

    Args:
        annotations_qwen: List of annotations from the Qwen model.
        annotations_gemma: List of annotations from the Gemma model.
        gold: List of final gold-standard triplets.

    Returns:
        Statistics dict with the following keys:
          - "total_clauses": int, total number of annotated clauses
          - "full_agreement_count": int, number of clauses where Qwen
            and Gemma produced identical triplets (all 6 fields agree)
          - "full_agreement_rate": float, full_agreement_count / total_clauses
          - "field_agreement": dict[str, float], per-field agreement rates
            for each of the 6 fields
          - "disagreement_by_phenomenon": dict[str, int], counts of
            disagreement types by linguistic phenomenon (passive_voice,
            condition_boundary, negation, role_mismatch, etc.)
          - "qwen_gold_agreement_rate": float, full agreement rate between
            Qwen annotations and the final gold
          - "gemma_gold_agreement_rate": float, full agreement rate between
            Gemma annotations and the final gold
    """
    # Validate input lengths match.
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

    # --- Full Agreement (triplet-level) ---
    # A triplet is in full agreement only if all 6 fields agree.
    full_agreement_count = 0
    # Per-field agreement counters.
    field_agreement_counts = {field_name: 0 for field_name, _, _ in FIELD_SPEC}
    # Per-phenomenon disagreement counters.
    phenomenon_counts: Dict[str, int] = {}
    # Model-gold agreement counters.
    qwen_gold_agreement = 0
    gemma_gold_agreement = 0

    for i in range(total):
        qwen_anno = annotations_qwen[i]
        gemma_anno = annotations_gemma[i]
        gold_triplet = gold[i]

        # Compare Qwen vs Gemma field by field.
        consensus, disagreements = field_level_consensus(qwen_anno, gemma_anno)

        # Track full agreement.
        if len(disagreements) == 0:
            full_agreement_count += 1

        # Track per-field agreement (6 fields per clause).
        # We already know which fields disagreed from the consensus call.
        # All fields except those in disagreements are agreed.
        disagreed_field_names = {d["field"] for d in disagreements}
        for field_name, _, _ in FIELD_SPEC:
            if field_name not in disagreed_field_names:
                field_agreement_counts[field_name] += 1

        # Track phenomenon-level disagreement patterns.
        for d in disagreements:
            phenomenon = _classify_disagreement_phenomenon(
                d["field"],
                qwen_anno,
                gemma_anno,
            )
            phenomenon_counts[phenomenon] = phenomenon_counts.get(phenomenon, 0) + 1

        # Track model-gold agreement (triplet-level).
        if _triplets_equal(qwen_anno, gold_triplet):
            qwen_gold_agreement += 1
        if _triplets_equal(gemma_anno, gold_triplet):
            gemma_gold_agreement += 1

    # Compute rates from counts.
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
