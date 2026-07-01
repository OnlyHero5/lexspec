"""
Validation Step 6: Role Validation.

Validates subject.role against UD-derived legal role from modality/polarity analysis.
"""

from __future__ import annotations

from typing import List

from src.extraction.schema import (
    DependencyTree,
    FieldCorrection,
    LegalRole,
)
from src.linguistic.polarity_detector import PolarityDetector
from src.utils.logging import get_logger

logger = get_logger(__name__)


def step6_validate_role(
    llm_role: LegalRole,
    tree: DependencyTree,
    predicate_idx: int,
    predicate_lemma: str,
    polarity_detector: PolarityDetector,
    corrections: List[FieldCorrection],
    feedback_parts: List[str],
) -> bool:
    """Step 6: Validate subject.role against UD-derived role.

    The legal role is determined by modal auxiliary (shall/may/must)
    and negation (not/no/never) patterns. This is the PolarityDetector's
    primary function.

    Role validation is generally the most reliable step because the
    rules are deterministic: given an aux verb and negation status,
    the role is uniquely determined (with the exception of "will"
    which can be ambiguous between obligation and future tense).

    Args:
        llm_role: The LLM-assigned role.
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate.
        predicate_lemma: Lemma of the predicate (for lexical overrides).
        polarity_detector: PolarityDetector instance.
        corrections: List to append FieldCorrection objects to.
        feedback_parts: List to append feedback strings to.

    Returns:
        True if the role matches.
    """
    ud_role, polarity = polarity_detector.detect(
        tree, predicate_idx, predicate_lemma
    )

    if ud_role == LegalRole.OTHER:
        # UD cannot determine the role (no modal, no clear pattern).
        # We accept whatever the LLM assigned — we have no evidence
        # to contradict it.
        logger.debug("UD role is OTHER — accepting LLM role %s", llm_role.value)
        return True

    if llm_role == ud_role:
        return True

    # Role mismatch: UD has a clear role, LLM assigned something different.
    # This is almost always a correctable error because the role rules
    # are deterministic.
    corrections.append(FieldCorrection(
        field="subject.role",
        original=llm_role.value,
        corrected=ud_role.value,
        reason=(
            f"Legal role derived from UD parse: "
            f"modal='{polarity_detector.detect_modality(tree, predicate_idx)[0]}', "
            f"polarity={polarity}, predicate='{predicate_lemma}' -> "
            f"{ud_role.value}. The LLM assigned {llm_role.value} which "
            f"conflicts with the syntactic evidence."
        ),
    ))
    feedback_parts.append(
        f"Role mismatch: LLM assigned {llm_role.value} but UD modality "
        f"analysis indicates {ud_role.value} (polarity={polarity})."
    )
    return False
