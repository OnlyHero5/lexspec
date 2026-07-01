"""
Validation Step 7: Status Determination.

Decides whether the triplet is VALID, CORRECTED, or REFLEXION_REQUIRED.
"""

from __future__ import annotations

from typing import List

from src.extraction.schema import FieldCorrection, ValidationStatus
from src.utils.logging import get_logger

logger = get_logger(__name__)


def step7_determine_status(
    corrections: List[FieldCorrection],
) -> ValidationStatus:
    """Step 7: Determine the output status.

    Decision logic:

    1. No corrections -> VALID.
       The LLM output is syntactically consistent with the UD parse.
       No changes needed. This is the ideal outcome.

    2. Has corrections AND UD has candidates for all corrected fields ->
       CORRECTED.
       We can fix the LLM output automatically using the UD evidence.
       The corrected_prediction in ValidationResult contains the
       auto-fixed triplet.

       Conditions for CORRECTED:
         - All subject corrections have a non-None ud_subject.
         - All object corrections have a non-None ud_object.
         (Condition corrections always have UD evidence by construction,
          since we only add them when ud_spans is non-empty.)

    3. Has corrections AND UD is missing evidence for some corrected
       fields -> REFLEXION_REQUIRED.
       The LLM needs to re-analyze with linguistic hints because we
       cannot confidently auto-correct.

       Triggers for REFLEXION_REQUIRED:
         - Agentless passive: UD has patient but no agent.
           The LLM may have correctly inferred the agent from context,
           but we cannot verify it syntactically.
         - Missing UD subject: nsubj is None and obl:agent is None.
         - Missing UD object: obj is None and nsubj:pass is None.
           (Intransitive verb or parse error.)

    Args:
        corrections: List of identified corrections.

    Returns:
        ValidationStatus enum value.
    """
    if not corrections:
        return ValidationStatus.VALID

    # Check if any correction targets a field where UD lacks evidence.
    # We can only auto-correct if UD has a candidate for every
    # corrected field.
    needs_reflexion = False

    for correction in corrections:
        field = correction.field

        if field == "subject.text" and correction.corrected == "":
            # UD has no subject to offer as a correction.
            needs_reflexion = True
            logger.debug(
                "Reflexion needed: subject correction but UD has no subject"
            )

        if field == "action.object" and correction.corrected == "":
            # UD has no object to offer as a correction.
            needs_reflexion = True
            logger.debug(
                "Reflexion needed: object correction but UD has no object"
            )

        if field == "condition.text" and correction.corrected == "":
            # UD found no condition — but the LLM may have extracted
            # a valid condition that Stanza missed.
            needs_reflexion = True
            logger.debug(
                "Reflexion needed: condition removal but possible parse gap"
            )

    # Additional check: if either ud_subject or ud_object is None
    # and a correction targets the corresponding field, we need Reflexion.
    # (The correction.corrected check above catches explicit empty
    # corrections, but we also want to catch cases where the correction
    # text is non-empty but the UD token itself is missing — this
    # shouldn't happen in normal operation but is a safety check.)
    # Actually, this is already covered: if ud_subject is None and we
    # have a subject correction, the correction would have corrected=""
    # because there's no UD token to use. So the check above covers it.

    if needs_reflexion:
        logger.info("Status: REFLEXION_REQUIRED (%d corrections with gaps)", len(corrections))
        return ValidationStatus.REFLEXION_REQUIRED

    logger.info("Status: CORRECTED (%d auto-correctable corrections)", len(corrections))
    return ValidationStatus.CORRECTED
