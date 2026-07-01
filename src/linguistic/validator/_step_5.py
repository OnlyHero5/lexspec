"""
Validation Step 5: Condition Validation.

Compares LLM-extracted condition against UD condition spans using IoU overlap.
"""

from __future__ import annotations

from typing import Optional, List

from src.extraction.schema import (
    LegalTriplet,
    DependencyTree,
    FieldCorrection,
    ConditionType,
    ConditionSpan,
)
from src.linguistic.condition_extractor import ConditionExtractor
from src.utils.logging import get_logger

logger = get_logger(__name__)


def step5_validate_condition(
    triplet: LegalTriplet,
    tree: DependencyTree,
    predicate_idx: int,
    condition_extractor: ConditionExtractor,
    condition_overlap: float,
    corrections: List[FieldCorrection],
    feedback_parts: List[str],
) -> Optional[ConditionSpan]:
    """Step 5: Validate the condition field.

    Condition validation is the most complex step because it involves
    span boundary comparison, not just token matching. The key issues:

    1. Condition Omission: UD finds a condition clause but the LLM
       marked condition.type = NONE. The LLM missed the condition.

    2. Condition Over-extraction: The LLM marked a condition where
       UD finds none. The LLM hallucinated a condition or extracted
       main clause content as a condition.

    3. Condition Boundary Error: Both UD and LLM find a condition
       but their spans differ significantly (IoU below threshold).
       The LLM may have included main clause tokens or truncated
       the condition clause.

    4. Condition Type Error: Both find a condition but the LLM
       classified it incorrectly (e.g., TEMPORAL vs TRIGGER).

    Strategy:
      - Extract UD condition spans via ConditionExtractor.
      - Compare with LLM condition text using token overlap (IoU).
      - If IoU >= threshold: accept (minor boundary differences OK).
      - If IoU < threshold: add correction or trigger Reflexion.

    Args:
        triplet: The LLM-extracted triplet.
        tree: Dependency tree.
        predicate_idx: 1-based index of the predicate.
        condition_extractor: ConditionExtractor instance.
        condition_overlap: Minimum IoU threshold for condition span match.
        corrections: List to append FieldCorrection objects to.
        feedback_parts: List to append feedback strings to.

    Returns:
        The UD ConditionSpan, or None if no condition was detected.
    """
    llm_has_condition = (
        triplet.condition.text
        and triplet.condition.text.strip()
        and triplet.condition.type != ConditionType.NONE
    )

    # Extract all UD condition spans for this predicate.
    ud_spans = condition_extractor.extract_all(tree, predicate_idx)

    # Case 1: Neither LLM nor UD has a condition.
    if not ud_spans and not llm_has_condition:
        return None

    # Case 2: UD finds a condition but LLM missed it (omission).
    if ud_spans and not llm_has_condition:
        primary_span = ud_spans[0]
        feedback_parts.append(
            f"Condition omitted: UD parse identifies a "
            f"{primary_span.condition_type.value} condition clause: "
            f"'{primary_span.text}'. The LLM marked this clause as "
            f"having no condition."
        )
        corrections.append(FieldCorrection(
            field="condition.text",
            original="",
            corrected=primary_span.text,
            reason=(
                f"UD parse identifies a {primary_span.condition_type.value} "
                f"condition clause via advcl relation with mark='{primary_span.mark_text}': "
                f"'{primary_span.text}'. The LLM omitted this condition entirely."
            ),
        ))
        corrections.append(FieldCorrection(
            field="condition.type",
            original=ConditionType.NONE.value,
            corrected=primary_span.condition_type.value,
            reason=(
                f"Condition type derived from mark word "
                f"'{primary_span.mark_text}' -> {primary_span.condition_type.value}."
            ),
        ))
        # Return the UD span so the evidence object can be populated.
        # The status will be CORRECTED since we have UD evidence for
        # both the condition text and type.
        return primary_span

    # Case 3: LLM has a condition but UD finds none (over-extraction).
    if not ud_spans and llm_has_condition:
        feedback_parts.append(
            f"Condition over-extraction: the LLM extracted condition "
            f"'{triplet.condition.text}' but no condition clause was "
            f"found in the UD parse. The LLM may have extracted main "
            f"clause content as a condition."
        )
        # Check if the LLM condition text overlaps with the main clause.
        # This helps distinguish hallucination from extraction of
        # a temporal adverb phrase that UD didn't tag as advcl.
        correction_applied = False
        for token_idx in range(1, tree.token_count + 1):
            token = tree.get_token(token_idx)
            if token is None:
                continue
            if token.text.lower() in triplet.condition.text.lower():
                # The condition text contains main clause tokens.
                # Likely over-extraction — remove the condition.
                correction_applied = True
                break

        if correction_applied:
            corrections.append(FieldCorrection(
                field="condition.text",
                original=triplet.condition.text,
                corrected="",
                reason=(
                    "No condition clause found in UD parse. The LLM-extracted "
                    "condition text includes main clause tokens, suggesting "
                    "over-extraction. Condition removed."
                ),
            ))
            corrections.append(FieldCorrection(
                field="condition.type",
                original=triplet.condition.type.value,
                corrected=ConditionType.NONE.value,
                reason="No condition clause present in UD parse.",
            ))
        # If no main clause overlap, the LLM may have extracted
        # a genuine condition that UD failed to parse. We can't
        # confidently correct this — it's a parse error.
        # Still return None to signal "no UD condition found."
        return None

    # Case 4: Both LLM and UD have conditions — compare spans.
    if ud_spans and llm_has_condition:
        primary_span = ud_spans[0]

        # Compute overlap between LLM condition text and UD span.
        overlap = ConditionExtractor.compute_condition_overlap(
            triplet.condition.text, primary_span, tree
        )

        if overlap >= condition_overlap:
            # Overlap is sufficient. Check if condition type matches.
            if triplet.condition.type != primary_span.condition_type:
                # Type mismatch — correct the type.
                corrections.append(FieldCorrection(
                    field="condition.type",
                    original=triplet.condition.type.value,
                    corrected=primary_span.condition_type.value,
                    reason=(
                        f"Condition type derived from mark word "
                        f"'{primary_span.mark_text}' should be "
                        f"{primary_span.condition_type.value}, not "
                        f"{triplet.condition.type.value}."
                    ),
                ))
            # Condition span is acceptable.
            return primary_span

        # Overlap below threshold — boundary error.
        feedback_parts.append(
            f"Condition boundary error: LLM condition span "
            f"'{triplet.condition.text}' has low overlap (IoU={overlap:.2f}) "
            f"with UD condition span '{primary_span.text}'. "
            f"The LLM may have truncated or extended the condition clause."
        )
        corrections.append(FieldCorrection(
            field="condition.text",
            original=triplet.condition.text,
            corrected=primary_span.text,
            reason=(
                f"Condition boundary mismatch (IoU={overlap:.2f} < "
                f"threshold={condition_overlap}). UD parse identifies "
                f"the condition via advcl relation: '{primary_span.text}'."
            ),
        ))
        if triplet.condition.type != primary_span.condition_type:
            corrections.append(FieldCorrection(
                field="condition.type",
                original=triplet.condition.type.value,
                corrected=primary_span.condition_type.value,
                reason=(
                    f"Condition type corrected to "
                    f"{primary_span.condition_type.value} based on mark word "
                    f"'{primary_span.mark_text}'."
                ),
            ))
        return primary_span

    return None
