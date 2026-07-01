"""
LexSpec Extraction Package
============================

This package handles the extraction of legal triplets from contract clauses
using LLM-based extraction with post-hoc validation via Universal Dependencies
syntactic constraints.

Public API:
  - LLM Client:       ClientConfig, LLMClient — OpenAI-compatible wrapper for
                      llama.cpp remote inference with retries and structured output.
  - Extractor:        LegalTripletExtractor — prompts the LLM and parses structured
                      LegalTriplet responses with multi-strategy JSON parsing.
  - Schema models:    Subject, Action, Condition, LegalTriplet, DependencyTree,
                      ValidationResult, ErrorCase, AnnotationDisagreement
  - Enums:            LegalRole, ConditionType, ValidationStatus, ErrorCategory,
                      FieldErrorType
"""

# --- Schema models (central data contracts) ---
from src.extraction.schema import (
    # Core Triplet Models
    LegalTriplet,
    Subject,
    Action,
    Condition,
    # Enums
    LegalRole,
    ConditionType,
    ValidationStatus,
    ErrorCategory,
    FieldErrorType,
    # Linguistic Models
    Token,
    ClauseSpan,
    ConditionSpan,
    DependencyTree,
    # Validation Models
    LinguisticEvidence,
    FieldCorrection,
    ValidationResult,
    # Annotation / Error Analysis
    ErrorCase,
    AnnotationDisagreement,
)

# --- LLM Client ---
from src.extraction.client import (
    ClientConfig,
    LLMClient,
)

# --- Extractor ---
from src.extraction.extractor import (
    LegalTripletExtractor,
)

__all__ = [
    # Core Triplet
    "LegalTriplet",
    "Subject",
    "Action",
    "Condition",
    # Enums
    "LegalRole",
    "ConditionType",
    "ValidationStatus",
    "ErrorCategory",
    "FieldErrorType",
    # Linguistic
    "Token",
    "ClauseSpan",
    "ConditionSpan",
    "DependencyTree",
    # Validation
    "LinguisticEvidence",
    "FieldCorrection",
    "ValidationResult",
    # Annotation / Error
    "ErrorCase",
    "AnnotationDisagreement",
    # Client
    "ClientConfig",
    "LLMClient",
    # Extractor
    "LegalTripletExtractor",
]
