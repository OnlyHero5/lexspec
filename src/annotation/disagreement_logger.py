"""
Disagreement Logging for Annotation Quality Tracking
======================================================
Records and persists annotation disagreements between the two
annotation models (Qwen3.6 27B and Gemma4 31B) during the
gold-standard test set construction pipeline.

Each disagreement event is captured as an AnnotationDisagreement
Pydantic model (defined in src/extraction/schema.py) and serialized
to JSONL for downstream analysis, reporting, and quality monitoring.

The disagreement log serves several purposes:
  1. **Audit trail**: Every disagreement and its resolution is recorded,
     enabling full traceability from raw annotations to final gold labels.
  2. **Quality monitoring**: Disagreement patterns reveal systematic
     weaknesses in annotation models (e.g., one model consistently
     misidentifies passive voice subjects).
  3. **Human review queue**: Unresolved disagreements are the work list
     for human annotators to adjudicate.
  4. **Inter-annotator agreement reporting**: The log is the primary data
     source for computing Cohen's kappa, Krippendorff's alpha, and other
     agreement metrics (see src/evaluation/).

Usage:
    from src.annotation.disagreement_logger import log_disagreement
    from src.annotation.disagreement_io import save_disagreement_log

    record = log_disagreement(
        clause_id="LEXSPEC-001",
        text="Seller shall deliver the Goods.",
        anno_a=qwen_triplet,
        anno_b=gemma_triplet,
        disagreements=disagreement_list,
        resolution="Human chose anno_a for subject.role",
    )

    save_disagreement_log([record], "data/processed/annotation_log.jsonl")
"""

from __future__ import annotations

from typing import List, Optional

from src.extraction.schema import (
    LegalTriplet,
    Subject,
    Action,
    Condition,
    LegalRole,
    ConditionType,
    AnnotationDisagreement,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def log_disagreement(
    clause_id: str,
    text: str,
    anno_a: LegalTriplet,
    anno_b: LegalTriplet,
    disagreements: List[dict],
    resolution: Optional[str] = None,
) -> AnnotationDisagreement:
    """Create an AnnotationDisagreement record from a disagreement event.

    Captures the full context of an annotation disagreement between the
    two models: the original clause text, both models' annotations, the
    specific fields where they disagreed, and any resolution that was
    applied (human or automatic).

    The returned AnnotationDisagreement is a Pydantic model -- it carries
    full schema validation and can be serialized directly via
    model_dump(mode="json") for JSONL persistence.

    Args:
        clause_id: Unique clause identifier (e.g., "LEXSPEC-001").
                   Links the disagreement back to the source document.
        text: The full clause text that was annotated. Preserved for
              reference when reviewing disagreements.
        anno_a: First model's annotation (typically Qwen3.6 27B).
        anno_b: Second model's annotation (typically Gemma4 31B).
        disagreements: List of field-level disagreement records from
                       field_level_consensus(). Each dict should contain
                       at minimum: field, anno_a_value, anno_b_value.
                       May also contain: resolved, resolved_by, resolution.
        resolution: Optional overall resolution note describing how
                    disagreements were resolved (e.g., "all resolved by
                    human review", "auto-resolved for condition fields").

    Returns:
        A fully validated AnnotationDisagreement Pydantic model instance.
    """
    # Validate inputs minimally -- Pydantic validation will catch schema
    # issues when constructing the AnnotationDisagreement.
    if not clause_id:
        logger.warning("log_disagreement called with empty clause_id")
    if not text:
        logger.warning("log_disagreement called with empty clause text")
    if not disagreements:
        logger.debug(
            "log_disagreement called with empty disagreements list for clause '%s'",
            clause_id,
        )

    # Normalize the disagreement records to ensure all expected keys
    # are present. Some callers may provide partial records.
    normalized_disagreements: List[dict] = []
    for i, d in enumerate(disagreements):
        if not isinstance(d, dict):
            logger.warning(
                "Disagreement item %d is not a dict (type=%s) -- skipping",
                i, type(d).__name__,
            )
            continue
        normalized = {
            "field": d.get("field", f"unknown_{i}"),
            "anno_a_value": str(d.get("anno_a_value", "")),
            "anno_b_value": str(d.get("anno_b_value", "")),
            "resolved": bool(d.get("resolved", False)),
            "resolved_by": str(d.get("resolved_by", "")),
            "resolution_text": str(d.get("resolution", d.get("resolution_text", ""))),
        }
        normalized_disagreements.append(normalized)

    # Build the final gold triplet for this disagreement record.
    final_gold = _build_tentative_gold(anno_a, normalized_disagreements)

    # Construct the Pydantic model.
    try:
        record = AnnotationDisagreement(
            clause_id=clause_id,
            text=text,
            qwen_annotation=anno_a,
            gemma_annotation=anno_b,
            disagreement_fields=normalized_disagreements,
            final_gold=final_gold,
        )
    except Exception as exc:
        logger.error(
            "Failed to construct AnnotationDisagreement for clause '%s': %s",
            clause_id, exc,
        )
        raise ValueError(
            f"Invalid AnnotationDisagreement data for clause '{clause_id}'"
        ) from exc

    # Log the disagreement event at an appropriate level.
    unresolved_count = sum(
        1 for d in normalized_disagreements if not d["resolved"]
    )
    resolved_count = len(normalized_disagreements) - unresolved_count

    if unresolved_count > 0:
        logger.info(
            "Disagreement logged for '%s': %d fields, %d resolved, %d unresolved%s",
            clause_id,
            len(normalized_disagreements),
            resolved_count,
            unresolved_count,
            f" -- resolution note: {resolution}" if resolution else "",
        )
    else:
        logger.debug(
            "Disagreement logged for '%s': all %d fields resolved",
            clause_id, len(normalized_disagreements),
        )

    return record


def _build_tentative_gold(
    anno_a: LegalTriplet,
    disagreements: List[dict],
) -> LegalTriplet:
    """Build a tentative gold triplet from anno_a with any resolutions applied.

    Starts with anno_a's values as the default (since Qwen is the primary
    model). For each resolved disagreement where the resolution matches
    anno_b's value, overrides the corresponding field with anno_b's value.

    This produces the best-effort gold triplet given the current state
    of disagreement resolutions.

    Args:
        anno_a: The primary model's annotation (used as the base).
        disagreements: List of normalized disagreement records.

    Returns:
        A LegalTriplet representing the best-effort gold given current
        resolution state. For unresolved fields, anno_a's value is kept.
    """
    # Start with anno_a's values as the default.
    gold_subject_text = anno_a.subject.text
    gold_subject_role = anno_a.subject.role
    gold_action_predicate = anno_a.action.predicate
    gold_action_object = anno_a.action.object
    gold_condition_text = anno_a.condition.text
    gold_condition_type = anno_a.condition.type

    for d in disagreements:
        if not d.get("resolved", False):
            # Unresolved -- keep anno_a's value (already the default).
            continue

        field = d.get("field", "")
        resolution = d.get("resolution_text", d.get("resolution", ""))

        if not resolution:
            # Resolution is set but empty -- skip.
            logger.debug(
                "Disagreement for '%s' marked resolved but has empty resolution",
                field,
            )
            continue

        # Apply the resolved value to the appropriate field.
        if field == "subject.text":
            gold_subject_text = resolution
        elif field == "subject.role":
            # Parse the role string into a LegalRole enum.
            try:
                gold_subject_role = LegalRole(resolution)
            except ValueError:
                logger.warning(
                    "Invalid role resolution '%s' for subject.role -- keeping anno_a value",
                    resolution,
                )
        elif field == "action.predicate":
            gold_action_predicate = resolution
        elif field == "action.object":
            gold_action_object = resolution
        elif field == "condition.text":
            gold_condition_text = resolution
        elif field == "condition.type":
            try:
                gold_condition_type = ConditionType(resolution)
            except ValueError:
                logger.warning(
                    "Invalid condition type resolution '%s' -- keeping anno_a value",
                    resolution,
                )
        else:
            logger.warning(
                "Unknown field '%s' in disagreement -- cannot apply resolution",
                field,
            )

    return LegalTriplet(
        subject=Subject(text=gold_subject_text, role=gold_subject_role),
        action=Action(
            predicate=gold_action_predicate,
            object=gold_action_object,
        ),
        condition=Condition(
            text=gold_condition_text,
            type=gold_condition_type,
        ),
    )
