"""
Field-Level Voting Consensus for Dual-Model Annotation
========================================================
When two annotation models (Qwen3.6 27B and Gemma4 31B) produce
different outputs, this module resolves disagreements field by field,
identifying which fields need human review.

Consensus approach:
  - Fields are compared at the finest granularity (6 fields):
      1. subject.text
      2. subject.role
      3. action.predicate
      4. action.object
      5. condition.text
      6. condition.type
  - Text fields use normalized comparison (lowercase, strip articles,
    lemmatize where possible) to avoid surface-form disagreements.
  - Role/type fields use exact enum value match.
  - When both models agree -> adopt the agreed value.
  - When models disagree -> mark for human review, use anno_a (Qwen)
    as the tentative value (Qwen3.6 27B is the larger, primary model).

Output:
  - A consensus LegalTriplet (best-effort merge)
  - A list of disagreement records for downstream human review and
    logging (see src/annotation/disagreement_logger.py)
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple

from src.extraction.schema import (
    LegalTriplet,
    Subject,
    Action,
    Condition,
)
from src.utils.logging import get_logger

from src.annotation.normalization import normalize_text
from src.annotation.field_helpers import (
    FIELD_SPEC,
    _extract_field_values,
    _parse_role,
    _parse_condition_type,
)

logger = get_logger(__name__)


# =============================================================================
# Public API
# =============================================================================


def field_level_consensus(
    anno_a: LegalTriplet,
    anno_b: LegalTriplet,
) -> Tuple[LegalTriplet, List[Dict[str, Any]]]:
    """Compare two annotations field by field and produce a consensus.

    Compares 6 fields at the finest granularity:
      - subject.text, subject.role
      - action.predicate, action.object
      - condition.text, condition.type

    Matching rules:
      - Text fields: normalized comparison (lowercase, strip articles,
        basic lemmatization). Two texts that differ only by articles,
        case, or minor inflection are considered AGREED.
      - Role/type fields: exact string match on the enum value.

    Consensus logic:
      - Both models agree -> adopt the agreed value (the value itself
        is taken from anno_a, since they agree).
      - Models disagree -> mark the field as "needs_human_review",
        using anno_a (Qwen, the primary model) as the tentative value.

    Args:
        anno_a: First model's annotation (typically Qwen, the primary).
        anno_b: Second model's annotation (typically Gemma, the secondary).

    Returns:
        A tuple of (consensus_triplet, disagreements_list):
        - consensus_triplet: Best-effort merged LegalTriplet. For agreed
          fields, the value is adopted. For disagreed fields, anno_a's
          value is used as a tentative placeholder.
        - disagreements: List of disagreement records, each a dict with:
            {
                "field": str,          # e.g. "subject.text"
                "anno_a_value": str,   # Qwen's value
                "anno_b_value": str,   # Gemma's value
                "resolved": bool,      # Initially False; set to True after human review
                "resolved_by": str,    # "" initially; set to "human" or "auto"
                "resolution": str,     # "" initially; set to the resolved value
            }
    """
    # Build lookup maps for each annotation's field values.
    a_values = _extract_field_values(anno_a)
    b_values = _extract_field_values(anno_b)

    disagreements: List[Dict[str, Any]] = []

    # We'll build the consensus triplet field-by-field.
    # Start with copies of anno_a's values as defaults.
    consensus_subject_text = anno_a.subject.text
    consensus_subject_role = anno_a.subject.role
    consensus_action_predicate = anno_a.action.predicate
    consensus_action_object = anno_a.action.object
    consensus_condition_text = anno_a.condition.text
    consensus_condition_type = anno_a.condition.type

    for field_name, _, _ in FIELD_SPEC:
        a_val = a_values[field_name]
        b_val = b_values[field_name]

        # Determine whether the two values agree using the appropriate
        # comparison strategy for this field type.
        is_text_field = field_name in (
            "subject.text", "action.predicate", "action.object", "condition.text"
        )

        if is_text_field:
            # Normalized comparison for text fields.
            agreed = normalize_text(str(a_val)) == normalize_text(str(b_val))
        else:
            # Exact comparison for role/type enum fields.
            agreed = str(a_val) == str(b_val)

        if agreed:
            # Both models agree -- no action needed.
            logger.debug("Field '%s': AGREED (a='%s', b='%s')", field_name, a_val, b_val)
        else:
            # Models disagree -- record for human review.
            disagreement_record: Dict[str, Any] = {
                "field": field_name,
                "anno_a_value": str(a_val),
                "anno_b_value": str(b_val),
                "resolved": False,
                "resolved_by": "",
                "resolution": "",
            }
            disagreements.append(disagreement_record)
            logger.info(
                "Field '%s': DISAGREE (qwen='%s', gemma='%s')",
                field_name, a_val, b_val,
            )

    # Build the consensus triplet from the (possibly partially overridden)
    # field values.
    consensus = LegalTriplet(
        subject=Subject(
            text=consensus_subject_text,
            role=_parse_role(consensus_subject_role),
        ),
        action=Action(
            predicate=consensus_action_predicate,
            object=consensus_action_object,
        ),
        condition=Condition(
            text=consensus_condition_text,
            type=_parse_condition_type(consensus_condition_type),
        ),
    )

    total_fields = len(FIELD_SPEC)
    agreed_count = total_fields - len(disagreements)
    logger.info(
        "Consensus: %d/%d fields agreed, %d disagreements",
        agreed_count, total_fields, len(disagreements),
    )

    return consensus, disagreements


def resolve_disagreement(
    disagreement: Dict[str, Any],
    human_choice: str,
) -> Dict[str, Any]:
    """Apply human resolution to a disagreement record.

    When a human reviewer decides which model's annotation is correct
    for a disputed field, this function records that resolution. The
    resolved disagreement can then be used by build_gold_from_consensus()
    to produce the final gold triplet.

    Args:
        disagreement: A disagreement dict from field_level_consensus(),
                      containing at minimum the keys "field",
                      "anno_a_value", "anno_b_value".
        human_choice: The human's chosen value. This must be one of:
                      - anno_a_value (the human agrees with Qwen)
                      - anno_b_value (the human agrees with Gemma)
                      - A custom string (the human provides a different value)

    Returns:
        The updated disagreement dict with resolution fields filled:
          - "resolved": True
          - "resolved_by": "human"
          - "resolution": the chosen value
    """
    if not isinstance(disagreement, dict):
        logger.error("resolve_disagreement called with non-dict argument")
        return {"error": "Invalid disagreement record"}

    field = disagreement.get("field", "unknown")
    a_val = str(disagreement.get("anno_a_value", ""))
    b_val = str(disagreement.get("anno_b_value", ""))

    if human_choice == a_val:
        logger.info(
            "Resolution for '%s': human chose ANNO_A value '%s'", field, human_choice
        )
    elif human_choice == b_val:
        logger.info(
            "Resolution for '%s': human chose ANNO_B value '%s'", field, human_choice
        )
    else:
        logger.info(
            "Resolution for '%s': human provided custom value '%s' "
            "(anno_a='%s', anno_b='%s')",
            field, human_choice, a_val, b_val,
        )

    disagreement["resolved"] = True
    disagreement["resolved_by"] = "human"
    disagreement["resolution"] = human_choice

    return disagreement


def build_gold_from_consensus(
    clause_id: str,
    text: str,
    consensus: LegalTriplet,
    disagreements: List[Dict[str, Any]],
) -> LegalTriplet:
    """Build the final gold triplet from consensus, applying human resolutions.

    For each disputed field:
      - If the disagreement has been resolved (by human review), the
        resolved value replaces the tentative (anno_a) value.
      - If still unresolved, the anno_a value is kept as tentative.

    Args:
        clause_id: Clause identifier (e.g., "LEXSPEC-001"). Used for
                   logging only -- not stored in the triplet itself.
        text: Original clause text. Used for logging only.
        consensus: The consensus triplet from field_level_consensus(),
                   with anno_a values for disputed fields.
        disagreements: List of disagreement records, some of which may
                       have been resolved via resolve_disagreement().

    Returns:
        Final gold-standard LegalTriplet with all resolved values applied.
    """
    # Start with the consensus values (which are anno_a's for disputed fields).
    gold_subject_text = consensus.subject.text
    gold_subject_role = consensus.subject.role
    gold_action_predicate = consensus.action.predicate
    gold_action_object = consensus.action.object
    gold_condition_text = consensus.condition.text
    gold_condition_type = consensus.condition.type

    resolved_count = 0
    unresolved_count = 0

    for disagreement in disagreements:
        field = disagreement.get("field", "")
        is_resolved = disagreement.get("resolved", False)
        resolution = disagreement.get("resolution", "")

        if is_resolved and resolution:
            if field == "subject.text":
                gold_subject_text = resolution
            elif field == "subject.role":
                gold_subject_role = _parse_role(resolution)
            elif field == "action.predicate":
                gold_action_predicate = resolution
            elif field == "action.object":
                gold_action_object = resolution
            elif field == "condition.text":
                gold_condition_text = resolution
            elif field == "condition.type":
                gold_condition_type = _parse_condition_type(resolution)
            else:
                logger.warning(
                    "Unknown field '%s' in disagreement -- skipping resolution", field
                )
                continue

            resolved_count += 1
            logger.debug(
                "Applied resolution for '%s': '%s'", field, resolution
            )
        else:
            unresolved_count += 1
            logger.debug(
                "Field '%s' remains unresolved -- keeping anno_a value", field
            )

    logger.info(
        "Gold construction for clause '%s': %d resolved, %d unresolved (tentative)",
        clause_id, resolved_count, unresolved_count,
    )

    gold = LegalTriplet(
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

    return gold
