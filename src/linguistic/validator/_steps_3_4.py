"""
Validation Steps 3 & 4: Subject and Object Validation.

Step 3: Validate the subject field against UD-derived semantic agent.
Step 4: Validate the object field against UD-derived semantic patient.
"""

from __future__ import annotations

from typing import Optional, List

from src.extraction.schema import Token, FieldCorrection
from src.linguistic.ud_features import UDFeatureExtractor
from src.linguistic.text_utils import match_text, normalize_text
from src.utils.logging import get_logger

logger = get_logger(__name__)


def step3_validate_subject(
    llm_subject: str,
    ud_subject: Optional[Token],
    corrections: List[FieldCorrection],
    feedback_parts: List[str],
) -> bool:
    """Step 3: Validate the subject field.

    Compares the LLM-extracted subject.text against the UD-derived
    semantic agent token.

    Match criteria (in priority order):
      1. Exact match after normalization: "Seller" == "Seller".
      2. Substring match: "the Seller" contains "Seller".
      3. Token overlap: both strings share key content tokens.
      4. Coordination match: UD subject is a conjunct and
         LLM extracted the full coordination or a single conjunct.

    If the LLM subject does NOT match the UD subject:
      - If UD has a clear subject -> Add FieldCorrection (CORRECTED path).
      - If UD has no subject (agentless passive, missing nsubj) ->
        Add feedback and trigger REFLEXION.

    Args:
        llm_subject: The LLM-extracted subject text.
        ud_subject: The UD-derived semantic agent token (or None).
        corrections: List to append FieldCorrection objects to.
        feedback_parts: List to append feedback strings to.

    Returns:
        True if the subject is valid (no correction needed).
    """
    # Case A: LLM extracted a subject but UD found none.
    # This could mean:
    #   - Agentless passive: the LLM inferred an agent from context.
    #     This is actually GOOD LLM behavior, but we flag it for
    #     manual review since the UD parse cannot confirm.
    #   - The LLM hallucinated a subject.
    if ud_subject is None:
        if llm_subject and llm_subject.strip():
            # LLM extracted a subject where UD sees none.
            feedback_parts.append(
                f"Subject '{llm_subject}' was extracted by the LLM, but "
                f"no syntactic subject was found in the UD parse. This may "
                f"be an agentless passive where the LLM correctly inferred "
                f"the agent from discourse context. Manual review recommended."
            )
            # No correction applied (we cannot determine correctness).
            return True  # Treat as "valid" since we can't disprove it.
        else:
            # Both LLM and UD agree: no subject.
            return True

    # Case B: UD found a subject but LLM didn't extract one.
    if not llm_subject or not llm_subject.strip():
        feedback_parts.append(
            f"Subject is missing from the LLM extraction but UD parse "
            f"identifies '{ud_subject.text}' as the semantic agent "
            f"(deprel={ud_subject.deprel})."
        )
        corrections.append(FieldCorrection(
            field="subject.text",
            original="",
            corrected=ud_subject.text,
            reason=(
                f"UD parse identifies '{ud_subject.text}' as the "
                f"semantic agent via {ud_subject.deprel} relation. "
                f"The LLM omitted the subject entirely."
            ),
        ))
        return False

    # Case C: Both have subjects — compare them.
    if match_text(llm_subject, ud_subject):
        return True  # Match found.

    # Subject does not match.
    # If the LLM subject contains the UD subject text as a substring,
    # it's likely a coordination expansion — acceptable.
    if normalize_text(ud_subject.text) in normalize_text(llm_subject):
        logger.debug(
            "LLM subject '%s' contains UD subject '%s' — accepting as "
            "coordination expansion.", llm_subject, ud_subject.text,
        )
        return True

    # Mismatch: add correction.
    corrections.append(FieldCorrection(
        field="subject.text",
        original=llm_subject,
        corrected=ud_subject.text,
        reason=(
            f"UD parse identifies '{ud_subject.text}' as the semantic "
            f"agent via {ud_subject.deprel} relation (head: predicate "
            f"at index {ud_subject.head}). The LLM extracted "
            f"'{llm_subject}' which does not match the UD evidence. "
            f"This may indicate subject-object reversal in passive voice "
            f"or the LLM extracting a modifier instead of the head noun."
        ),
    ))
    feedback_parts.append(
        f"Subject mismatch: LLM extracted '{llm_subject}' but UD parse "
        f"identifies '{ud_subject.text}' as the semantic agent "
        f"(deprel={ud_subject.deprel})."
    )
    return False


def step4_validate_object(
    llm_object: str,
    ud_object: Optional[Token],
    corrections: List[FieldCorrection],
    feedback_parts: List[str],
) -> bool:
    """Step 4: Validate the object field.

    Compares LLM action.object against UD-derived semantic patient.

    Uses the same matching logic as subject validation (Step 3):
      - Exact match after normalization
      - Substring/token overlap match
      - Coordination expansion match

    Edge cases:
      - Intransitive verb: UD has no obj. If LLM also has no object,
        this is valid. If LLM extracted an object for an intransitive
        verb, it's likely a hallucination or extracted from a prepositional
        complement.
      - Passive without nsubj:pass: The patient may not be expressed.
        Rare in legal text but possible in impersonal constructions.

    Args:
        llm_object: The LLM-extracted object text.
        ud_object: The UD-derived semantic patient token (or None).
        corrections: List to append FieldCorrection objects to.
        feedback_parts: List to append feedback strings to.

    Returns:
        True if the object is valid (no correction needed).
    """
    # Case A: Neither LLM nor UD has an object.
    if ud_object is None and (not llm_object or not llm_object.strip()):
        return True

    # Case B: UD found an object but LLM didn't.
    if ud_object is not None and (not llm_object or not llm_object.strip()):
        feedback_parts.append(
            f"Object is missing from the LLM extraction but UD parse "
            f"identifies '{ud_object.text}' as the semantic patient "
            f"(deprel={ud_object.deprel})."
        )
        corrections.append(FieldCorrection(
            field="action.object",
            original="",
            corrected=ud_object.text,
            reason=(
                f"UD parse identifies '{ud_object.text}' as the direct "
                f"object via {ud_object.deprel} relation. The LLM omitted "
                f"the object entirely."
            ),
        ))
        return False

    # Case C: UD has no object but LLM extracted one.
    # The verb may be intransitive, or the LLM may have extracted a
    # prepositional complement as an object.
    if ud_object is None and llm_object and llm_object.strip():
        feedback_parts.append(
            f"Object '{llm_object}' was extracted by the LLM but no "
            f"direct object was found in the UD parse. The predicate "
            f"may be intransitive. If the LLM object is a prepositional "
            f"complement or indirect object, this may be semantically "
            f"valid but syntactically unconfirmed."
        )
        # We cannot disprove the LLM's extraction, but we flag it.
        # No automatic correction — the LLM may be right about
        # semantic content even if it's not a syntactic obj.
        return True

    # Case D: Both have objects — compare them.
    if ud_object is not None and match_text(llm_object, ud_object):
        return True

    # Mismatch: LLM and UD disagree on the object.
    if ud_object is not None:
        corrections.append(FieldCorrection(
            field="action.object",
            original=llm_object,
            corrected=ud_object.text,
            reason=(
                f"UD parse identifies '{ud_object.text}' as the semantic "
                f"patient via {ud_object.deprel} relation (head: predicate "
                f"at index {ud_object.head}). The LLM extracted "
                f"'{llm_object}' which does not match. This may indicate "
                f"subject-object reversal in passive voice or the LLM "
                f"extracting an oblique modifier instead of the direct object."
            ),
        ))
        feedback_parts.append(
            f"Object mismatch: LLM extracted '{llm_object}' but UD parse "
            f"identifies '{ud_object.text}' as the semantic patient "
            f"(deprel={ud_object.deprel})."
        )
        return False

    return True
