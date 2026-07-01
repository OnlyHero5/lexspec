"""
Result-building utilities: LinguisticEvidence, feedback, and correction application.

These functions construct the ValidationResult sub-objects from the
accumulated validation state.
"""

from __future__ import annotations

from typing import Optional, List

from src.extraction.schema import (
    DependencyTree,
    Token,
    LegalTriplet,
    Subject,
    Action,
    Condition,
    LegalRole,
    ConditionType,
    LinguisticEvidence,
    FieldCorrection,
    ConditionSpan,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_linguistic_evidence(
    tree: DependencyTree,
    predicate_idx: int,
    ud_subject: Optional[Token],
    ud_object: Optional[Token],
    condition_span: Optional[ConditionSpan],
    is_passive_detected: bool,
    modality_aux: str,
    polarity: str,
    legal_role: LegalRole,
) -> LinguisticEvidence:
    """Build the LinguisticEvidence object from validation state.

    This evidence object captures everything the validator derived
    from the UD parse, providing the ground-truth reference against
    which the LLM prediction was compared. It is included in the
    final ValidationResult for transparency and is used by the
    error analyzer for generating linguistic explanations.
    """
    predicate_token = tree.get_token(predicate_idx)

    evidence = LinguisticEvidence(
        predicate=predicate_token.lemma if predicate_token else "",
        predicate_index=predicate_idx,
        ud_subject=ud_subject.text if ud_subject else "",
        ud_object=ud_object.text if ud_object else "",
        condition_span=condition_span.text if condition_span else "",
        condition_type=(
            condition_span.condition_type if condition_span
            else ConditionType.NONE
        ),
        passive_detected=is_passive_detected,
        modality_aux=modality_aux,
        polarity=polarity,
        legal_role=legal_role,
    )

    return evidence


def build_feedback(feedback_parts: List[str]) -> str:
    """Assemble human-readable feedback string.

    Used in two contexts:
      1. Reflexion: The feedback is included in the Reflexion prompt
         sent back to the LLM, telling it what went wrong and how to
         re-analyze the clause.
      2. Error analysis: The feedback is logged in error case records
         for downstream diagnostic reporting.

    The feedback uses natural language and cites specific UD relations
    to help the LLM understand what syntactic patterns to look for.

    Args:
        feedback_parts: List of individual feedback strings from
                        each validation step.

    Returns:
        Concatenated feedback string, or empty string if no issues.
    """
    if not feedback_parts:
        return ""

    # Number each feedback item for clarity.
    numbered = [
        f"{i}. {part}" for i, part in enumerate(feedback_parts, 1)
    ]

    preamble = (
        "The UD syntactic analysis identified the following issues with "
        "the extracted triplet:"
    )

    return preamble + "\n" + "\n".join(numbered)


def apply_corrections(
    triplet: LegalTriplet,
    corrections: List[FieldCorrection],
) -> LegalTriplet:
    """Apply a list of field corrections to produce a corrected triplet.

    Iterates over all field corrections and updates the corresponding
    fields in the triplet. The corrected triplet is a new LegalTriplet
    object — the original is never mutated.

    Field paths handled:
      - "subject.text" -> triplet.subject.text
      - "subject.role" -> triplet.subject.role (converted from string)
      - "action.object" -> triplet.action.object
      - "action.predicate" -> triplet.action.predicate
      - "condition.text" -> triplet.condition.text
      - "condition.type" -> triplet.condition.type (converted from string)

    Args:
        triplet: The original LLM triplet (not mutated).
        corrections: List of FieldCorrection objects.

    Returns:
        A new LegalTriplet with corrections applied.
    """
    # Start with a copy of the original triplet.
    # We build new Subject, Action, Condition objects as needed.
    new_subject_text = triplet.subject.text
    new_subject_role = triplet.subject.role
    new_predicate = triplet.action.predicate
    new_object = triplet.action.object
    new_condition_text = triplet.condition.text
    new_condition_type = triplet.condition.type

    for correction in corrections:
        field = correction.field
        corrected_value = correction.corrected

        if field == "subject.text":
            new_subject_text = corrected_value
            logger.debug(
                "Applied correction: subject.text '%s' -> '%s'",
                correction.original, corrected_value,
            )
        elif field == "subject.role":
            # Convert string role back to LegalRole enum.
            try:
                new_subject_role = LegalRole(corrected_value)
                logger.debug(
                    "Applied correction: subject.role '%s' -> '%s'",
                    correction.original, corrected_value,
                )
            except ValueError:
                logger.warning(
                    "Invalid role correction value '%s' — keeping original",
                    corrected_value,
                )
        elif field == "action.object":
            new_object = corrected_value
            logger.debug(
                "Applied correction: action.object '%s' -> '%s'",
                correction.original, corrected_value,
            )
        elif field == "action.predicate":
            new_predicate = corrected_value
            logger.debug(
                "Applied correction: action.predicate '%s' -> '%s'",
                correction.original, corrected_value,
            )
        elif field == "condition.text":
            new_condition_text = corrected_value
            logger.debug(
                "Applied correction: condition.text '%s' -> '%s'",
                correction.original[:40], corrected_value[:40],
            )
        elif field == "condition.type":
            try:
                new_condition_type = ConditionType(corrected_value)
                logger.debug(
                    "Applied correction: condition.type '%s' -> '%s'",
                    correction.original, corrected_value,
                )
            except ValueError:
                logger.warning(
                    "Invalid condition type correction '%s' — keeping original",
                    corrected_value,
                )
        else:
            logger.warning("Unknown correction field: '%s'", field)

    return LegalTriplet(
        subject=Subject(text=new_subject_text, role=new_subject_role),
        action=Action(predicate=new_predicate, object=new_object),
        condition=Condition(text=new_condition_text, type=new_condition_type),
    )
