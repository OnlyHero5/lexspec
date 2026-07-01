"""
LexSpec Schema -- Error Analysis & Annotation Models
=====================================================

ErrorCase and AnnotationDisagreement for evaluation diagnostics
and multi-annotator gold standard construction.
"""

from typing import List, Dict, Any

from pydantic import BaseModel, Field

from .enums import ErrorCategory, FieldErrorType
from .triplet import LegalTriplet


class ErrorCase(BaseModel):
    """
    A detailed error analysis record for a single clause where the system
    prediction diverged from the gold standard.

    Each ErrorCase captures:
      - What the system predicted vs. what the gold standard says (cross-product)
      - The primary linguistic phenomenon causing the error (ErrorCategory)
      - Which field(s) were affected (FieldErrorType)
      - A detailed linguistic explanation citing UD relations and syntactic evidence

    These records feed into the evaluation module for generating error
    distribution reports, confusion matrices, and diagnostic summaries.
    """
    error_id: str = Field(
        description="Unique error identifier, e.g. 'E-001', for traceability in reports"
    )
    text: str = Field(
        description="Original clause text that was analyzed"
    )
    prediction: LegalTriplet = Field(
        description="The system's predicted triplet (corrected or raw)"
    )
    gold: LegalTriplet = Field(
        description="The gold-standard triplet (human-annotated or adjudicated)"
    )
    primary_category: ErrorCategory = Field(
        description="Primary error category -- the linguistic phenomenon responsible for the error"
    )
    secondary_category: FieldErrorType = Field(
        description="Secondary error category -- which field(s) were incorrect"
    )
    linguistic_explanation: str = Field(
        description=(
            "Detailed natural-language explanation of the error, citing specific UD "
            "dependency relations (e.g., 'nsubj:pass at token 3', 'advcl from token 2 to 8') "
            "and explaining why the syntactic structure caused the extraction to fail."
        )
    )


class AnnotationDisagreement(BaseModel):
    """
    Record of a disagreement between two annotation models (Qwen and Gemma)
    during the multi-annotator gold standard construction pipeline.

    Disagreements are resolved through:
      1. Automatic rule-based adjudication (for clear UD-backed cases)
      2. Human review (for ambiguous or complex cases)

    The resolution process is tracked in `disagreement_fields` and the
    final adjudicated result is stored in `final_gold`.
    """
    clause_id: str = Field(
        description="Unique clause identifier, e.g. 'C-0042', linking to the source document"
    )
    text: str = Field(
        description="The full clause text being annotated"
    )
    qwen_annotation: LegalTriplet = Field(
        description="Annotation produced by the Qwen model"
    )
    gemma_annotation: LegalTriplet = Field(
        description="Annotation produced by the Gemma model"
    )
    disagreement_fields: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of disagreement records, each a dict with keys: "
            "field (str), qwen_value (str), gemma_value (str), "
            "resolution (str), resolved_by (str -- 'auto' or 'human')"
        )
    )
    final_gold: LegalTriplet = Field(
        description="The final gold-standard triplet after disagreement resolution"
    )
