"""
Combined linguistic metrics entry point.

Aggregates the four linguistic-specific evaluation metrics:
  1. Dependency Path Legality Rate (dep_path_metrics.py)
  2. Passive Voice Recovery Accuracy (passive_metrics.py)
  3. Condition Boundary IoU (condition_metrics.py)
  4. Linguistic Correction Rate (condition_metrics.py)

These metrics serve as diagnostic tools: they help identify which linguistic
phenomena are causing extraction failures, guiding targeted improvements.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

from src.extraction.schema import (
    LegalTriplet, DependencyTree, ValidationResult,
)
from src.evaluation.dep_path_metrics import compute_dependency_path_legality
from src.evaluation.passive_metrics import compute_passive_recovery_accuracy
from src.evaluation.condition_metrics import compute_condition_iou, compute_correction_rate
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_all_linguistic_metrics(
    predictions: List[LegalTriplet],
    gold: List[LegalTriplet],
    trees: List[DependencyTree],
    validation_results: Optional[List[ValidationResult]] = None,
) -> Dict[str, Any]:
    """Compute all four linguistic metrics in a single call.

    This is the primary entry point for the linguistic evaluation dimension.
    It returns a comprehensive dict with all metric groups that can be
    serialized directly to JSON for the evaluation report.

    Args:
        predictions: List of predicted LegalTriplets.
        gold: List of gold-standard LegalTriplets.
        trees: List of DependencyTree objects from UD parsing.
        validation_results: Optional list of ValidationResult objects.
                            If None, correction rate metrics are omitted.

    Returns:
        Dict with the following top-level keys:
        - dependency_path_legality: float
        - passive_recovery: Dict[str, float] (passive voice metrics)
        - condition_iou: float
        - correction_analysis: Dict[str, float] or None
        - summary: Dict with aggregated highlights for report inclusion

    Raises:
        ValueError: If the core inputs (predictions, gold, trees) have
                    mismatched lengths.
    """
    n = len(predictions)
    if n != len(gold) or n != len(trees):
        raise ValueError(
            f"Core inputs must have the same length. "
            f"Got predictions={len(predictions)}, gold={len(gold)}, trees={len(trees)}."
        )

    # Metric 1: Dependency path legality.
    legality = compute_dependency_path_legality(predictions, trees)

    # Metric 2: Passive voice recovery accuracy.
    passive_metrics = compute_passive_recovery_accuracy(predictions, trees, gold)

    # Metric 3: Condition boundary IoU.
    condition_iou = compute_condition_iou(predictions, trees)

    # Metric 4: Correction rate (only if validation results are provided).
    correction_metrics = None
    if validation_results is not None:
        correction_metrics = compute_correction_rate(validation_results)

    # Build a summary for the evaluation report.
    summary = {
        "linguistic_quality_indicators": {
            "dependency_legality": legality,
            "condition_boundary_iou": condition_iou,
        },
        "passive_voice_handling": {
            "passive_count": passive_metrics["passive_count"],
            "recovery_accuracy": passive_metrics["recovery_accuracy"],
            "false_agent_rate": passive_metrics["false_agent_rate"],
        },
    }
    if correction_metrics is not None:
        summary["validator_performance"] = {
            "valid_rate": correction_metrics["valid_rate"],
            "correction_success_rate": correction_metrics["correction_success_rate"],
            "reflexion_required_rate": correction_metrics["reflexion_rate"],
        }

    return {
        "dependency_path_legality": legality,
        "passive_recovery": passive_metrics,
        "condition_iou": condition_iou,
        "correction_analysis": correction_metrics,
        "summary": summary,
    }
