"""
7-Step Validation Algorithm — Standalone Implementation
========================================================
Extracted from ConstraintValidator.validate() to keep file size manageable.

This module contains the core orchestration logic of the 7-step algorithm.
The ConstraintValidator class delegates to this function, keeping the class
itself focused on component lifecycle and configuration.
"""

from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING

from src.extraction.schema import (
    DependencyTree,
    Token,
    LegalTriplet,
    ValidationStatus,
    ValidationResult,
    LinguisticEvidence,
    FieldCorrection,
    LegalRole,
    ConditionSpan,
)
from src.utils.logging import get_logger

from src.linguistic.validator._steps_1_2 import step1_find_predicate, step2_detect_voice
from src.linguistic.validator._steps_3_4 import step3_validate_subject, step4_validate_object
from src.linguistic.validator._step_5 import step5_validate_condition
from src.linguistic.validator._step_6 import step6_validate_role
from src.linguistic.validator._step_7 import step7_determine_status
from src.linguistic.validator._results import (
    build_linguistic_evidence,
    build_feedback,
    apply_corrections,
)

if TYPE_CHECKING:
    from src.linguistic.validator.validator import ConstraintValidator

logger = get_logger(__name__)


def run_validation(
    validator: "ConstraintValidator",
    triplet: LegalTriplet,
    text: str,
    tree: Optional[DependencyTree] = None,
) -> ValidationResult:
    """Run the complete 7-step constraint validation algorithm.

    This is the main entry point for all validation. It orchestrates
    the 7-step pipeline and produces a ValidationResult with status,
    evidence, corrections, and feedback.

    If `tree` is None, the text is parsed internally via StanzaParser.
    For batch processing, pre-parse the text and pass the tree to
    avoid redundant Stanza calls.

    Args:
        validator: ConstraintValidator instance providing parser access
                   and component dependencies.
        triplet: LLM-extracted legal triplet to validate.
        text: Original contract clause text (exact input sent to LLM).
        tree: Pre-parsed dependency tree. Parsed from text if None.

    Returns:
        ValidationResult with full validation details.

    Raises:
        ValueError: If parsing fails (empty text, no sentences produced).
        RuntimeError: If Stanza is not initialized and parser is None.
    """
    # Ensure we have a parse tree.
    if tree is None:
        logger.debug("No pre-parsed tree provided — parsing text")
        tree = validator.parser.parse(text)

    if tree.token_count == 0:
        raise ValueError(
            "Cannot validate: dependency tree has zero tokens. "
            "The input text may be empty or unparseable."
        )

    # Accumulators for the 7-step algorithm.
    corrections: List[FieldCorrection] = []
    feedback_parts: List[str] = []
    ud_subject: Optional[Token] = None
    ud_object: Optional[Token] = None
    condition_span: Optional[ConditionSpan] = None
    is_passive_detected: bool = False
    modality_aux: str = ""
    polarity: str = "positive"
    legal_role: LegalRole = LegalRole.OTHER
    predicate_token: Optional[Token] = None

    logger.info(
        "Validating triplet for sentence (%d tokens): '%s'",
        tree.token_count,
        text[:100] + "..." if len(text) > 100 else text,
    )

    # ---- Step 1: Locate the root predicate ----
    predicate_token = step1_find_predicate(tree)
    if predicate_token is None:
        # Cannot validate without a root predicate.
        # The sentence may be fragmentary or the parse failed.
        logger.warning("No root predicate found in tree — cannot validate")
        return ValidationResult(
            status=ValidationStatus.REFLEXION_REQUIRED,
            original_prediction=triplet,
            corrected_prediction=None,
            linguistic_evidence=LinguisticEvidence(),
            corrections=[],
            feedback=(
                "Could not locate the main clause predicate in the dependency "
                "parse. The sentence may be fragmentary, ungrammatical, or "
                "the parser may have produced a malformed tree. Please verify "
                "the input text and re-parse."
            ),
        )

    predicate_idx = predicate_token.index
    predicate_lemma = predicate_token.lemma
    logger.info(
        "Step 1: Root predicate='%s' (lemma='%s') at index %d",
        predicate_token.text, predicate_lemma, predicate_idx,
    )

    # ---- Step 2: Detect voice; restore semantic arguments ----
    is_passive_detected, raw_ud_subject, raw_ud_object = step2_detect_voice(
        tree, predicate_idx
    )

    # In passive voice, raw_ud_subject = obl:agent (semantic agent),
    # raw_ud_object = nsubj:pass (semantic patient).
    # In active voice, raw_ud_subject = nsubj, raw_ud_object = obj.
    ud_subject = raw_ud_subject
    ud_object = raw_ud_object

    logger.info(
        "Step 2: Passive=%s, UD subject='%s', UD object='%s'",
        is_passive_detected,
        ud_subject.text if ud_subject else "None",
        ud_object.text if ud_object else "None",
    )

    # ---- Step 3: Validate subject ----
    subject_valid = step3_validate_subject(
        triplet.subject.text, ud_subject, corrections, feedback_parts
    )
    if subject_valid:
        logger.info("Step 3: Subject VALID (%s)", triplet.subject.text)
    else:
        logger.info("Step 3: Subject needs correction")

    # ---- Step 4: Validate object ----
    object_valid = step4_validate_object(
        triplet.action.object, ud_object, corrections, feedback_parts
    )
    if object_valid:
        logger.info("Step 4: Object VALID (%s)", triplet.action.object)
    else:
        logger.info("Step 4: Object needs correction")

    # ---- Step 5: Validate condition ----
    condition_span = step5_validate_condition(
        triplet, tree, predicate_idx,
        validator.condition_extractor, validator._condition_overlap,
        corrections, feedback_parts,
    )
    if condition_span is not None:
        logger.info(
            "Step 5: UD condition='%s' (type=%s)",
            condition_span.text[:80],
            condition_span.condition_type.value,
        )
    else:
        logger.info("Step 5: No UD condition detected")

    # ---- Step 6: Validate modality/role ----
    role_valid = step6_validate_role(
        triplet.subject.role, tree, predicate_idx,
        predicate_lemma,
        validator.polarity_detector, corrections, feedback_parts,
    )

    # Extract modality for evidence (even if role is valid).
    modality_aux, polarity = validator.polarity_detector.detect_modality(
        tree, predicate_idx
    )
    legal_role, _ = validator.polarity_detector.detect(
        tree, predicate_idx, predicate_lemma
    )

    if role_valid:
        logger.info("Step 6: Role VALID (%s)", triplet.subject.role.value)
    else:
        logger.info("Step 6: Role needs correction")

    # ---- Step 7: Determine output status ----
    status = step7_determine_status(corrections)
    logger.info("Step 7: Final status = %s", status.value)

    # ---- Build the ValidationResult ----
    evidence = build_linguistic_evidence(
        tree=tree,
        predicate_idx=predicate_idx,
        ud_subject=ud_subject,
        ud_object=ud_object,
        condition_span=condition_span,
        is_passive_detected=is_passive_detected,
        modality_aux=modality_aux,
        polarity=polarity,
        legal_role=legal_role,
    )

    feedback = build_feedback(feedback_parts)

    corrected_triplet = None
    if status == ValidationStatus.CORRECTED and corrections:
        corrected_triplet = apply_corrections(triplet, corrections)

    return ValidationResult(
        status=status,
        original_prediction=triplet,
        corrected_prediction=corrected_triplet,
        linguistic_evidence=evidence,
        corrections=corrections,
        feedback=feedback,
    )
