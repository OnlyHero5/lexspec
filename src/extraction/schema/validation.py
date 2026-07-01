"""
LexSpec 数据模型 —— 校验模型
===========================

UD 约束校验器的输出模型：LinguisticEvidence, FieldCorrection, ValidationResult。
"""

from typing import Optional, List

from pydantic import BaseModel, Field

from .enums import ValidationStatus, ConditionType, LegalRole
from .triplet import LegalTriplet


class LinguisticEvidence(BaseModel):
    """
    Linguistic evidence extracted from a UD dependency parse for a single clause.

    This model captures everything the constraint validator derives from the
    syntactic analysis, providing the ground-truth reference against which the
    LLM prediction is compared.

    Fields are populated incrementally during validation:
      1. Identify root predicate (predicate, predicate_index)
      2. Extract UD subject (nsubj or obl:agent) and object (obj or nsubj:pass)
      3. Detect passive voice and modality
      4. Extract condition clause span
      5. Derive legal role from modality + syntactic cues
    """
    predicate: str = Field(
        default="",
        description="Identified root predicate in lemma form, e.g. 'deliver', 'indemnify'"
    )
    predicate_index: int = Field(
        default=0,
        description="1-based token index of the root predicate; 0 if not found"
    )
    ud_subject: str = Field(
        default="",
        description="Subject identified via UD — either nsubj (active) or obl:agent (passive)"
    )
    ud_object: str = Field(
        default="",
        description="Object identified via UD — either obj (active) or nsubj:pass (passive)"
    )
    condition_span: str = Field(
        default="",
        description="Full text of the condition clause span, if one was detected"
    )
    condition_type: ConditionType = Field(
        default=ConditionType.NONE,
        description="Semantic type of the detected condition clause"
    )
    passive_detected: bool = Field(
        default=False,
        description="True if passive voice was detected (nsubj:pass or aux:pass present)"
    )
    modality_aux: str = Field(
        default="",
        description="Modal auxiliary verb lemma, e.g. 'shall', 'may', 'must'; empty if none"
    )
    polarity: str = Field(
        default="positive",
        description="Clause polarity: 'positive' or 'negative' (based on negation particles)"
    )
    legal_role: LegalRole = Field(
        default=LegalRole.OTHER,
        description="Derived legal role based on modality + voice + argument structure"
    )


class FieldCorrection(BaseModel):
    """
    A single field-level correction applied by the validator to an LLM prediction.

    Each correction records exactly what changed, why it changed, and the
    linguistic evidence supporting the change. Corrections are logged for
    downstream error analysis and Reflexion prompt construction.
    """
    field: str = Field(
        description="Dot-notation field path, e.g. 'subject.text', 'condition.type', 'action.predicate'"
    )
    original: str = Field(
        default="",
        description="The original value produced by the LLM"
    )
    corrected: str = Field(
        default="",
        description="The corrected value derived from UD syntactic analysis"
    )
    reason: str = Field(
        default="",
        description="Natural-language justification citing UD relations and syntactic evidence"
    )


class ValidationResult(BaseModel):
    """
    Complete output of the UD constraint validator for one clause.

    This is the bridge between extraction and correction/evaluation:
      - extraction produces LegalTriplet predictions
      - linguistic produces LinguisticEvidence from UD parse
      - the validator compares them and produces this ValidationResult
      - correction modules apply or escalate based on status
      - evaluation modules aggregate errors from corrections

    Status semantics:
      - VALID:              prediction == evidence; no action needed
      - CORRECTED:          minor field errors fixed; corrected_prediction is set
      - REFLEXION_REQUIRED: structural error; feedback is populated for LLM re-extraction
    """
    status: ValidationStatus = Field(
        description="Validation outcome: VALID, CORRECTED, or REFLEXION_REQUIRED"
    )
    original_prediction: LegalTriplet = Field(
        description="The LLM's original prediction, preserved for error analysis"
    )
    corrected_prediction: Optional[LegalTriplet] = Field(
        default=None,
        description="Corrected triplet; set when status is CORRECTED, None when REFLEXION_REQUIRED"
    )
    linguistic_evidence: LinguisticEvidence = Field(
        description="UD-derived syntactic evidence used for validation"
    )
    corrections: List[FieldCorrection] = Field(
        default_factory=list,
        description="List of field corrections applied; empty if VALID"
    )
    feedback: str = Field(
        default="",
        description=(
            "Human-readable feedback explaining what went wrong and why. "
            "Used in Reflexion prompts and error logging. Empty if status is VALID."
        )
    )
