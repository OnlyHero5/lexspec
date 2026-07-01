"""
LexSpec Annotation Package
===========================
Dual-model annotation pipeline for gold-standard test set construction.

This package implements the multi-annotator workflow used exclusively in
Phase 1 (LexSpec-100 test set construction). Two annotation models
(Qwen3.6 27B, Gemma4 31B) independently annotate each contract clause,
and their outputs are reconciled through field-level voting consensus.

IMPORTANT: These models are COMPLETELY ISOLATED from the experiment model
(Qwen3.5 9B) used in Phases 2-3. Annotation model predictions never leak
into training, prompting, or evaluation.

Package structure:
  - llm_annotator.py:       LLMAnnotator class for single-model annotation
  - consensus.py:           Field-level voting, disagreement resolution,
                            gold-standard construction
  - statistics.py:          Inter-annotator agreement statistics computation
  - normalization.py:       Text normalization for fuzzy comparison
  - field_helpers.py:       Field extraction, parsing, and comparison helpers
  - triplet_coercion.py:    JSON-to-LegalTriplet coercion utilities
  - prompts.py:             Prompt loading from YAML (no fallbacks)
  - response_parser.py:     LLM response parsing for annotations
  - disagreement_logger.py: AnnotationDisagreement recording
  - disagreement_io.py:     Disagreement record persistence (JSONL I/O)
  - reviewer.py:            Cross-model annotation reviewer

Public API:
  - LLMAnnotator:              Annotate clauses with a single LLM
  - CrossModelReviewer:        Have one LLM review another's annotations
  - field_level_consensus:     Compare two annotations field-by-field
  - resolve_disagreement:      Apply human resolution to a disagreement
  - build_gold_from_consensus: Build final gold triplet from consensus data
  - generate_annotation_stats: Compute inter-annotator agreement statistics
  - normalize_text:            Normalize text for fuzzy comparison
  - coerce_to_triplet:         Coerce raw JSON data to LegalTriplet
  - infer_condition_type:      Infer condition type from text (canonical)
  - log_disagreement:          Create AnnotationDisagreement records
  - save_disagreement_log:     Persist disagreement records to JSONL
"""

from src.annotation.llm_annotator import LLMAnnotator
from src.annotation.reviewer import CrossModelReviewer
from src.annotation.consensus import (
    field_level_consensus,
    resolve_disagreement,
    build_gold_from_consensus,
)
from src.annotation.statistics import generate_annotation_stats
from src.annotation.normalization import normalize_text
from src.annotation.triplet_coercion import coerce_to_triplet, infer_condition_type
from src.annotation.disagreement_logger import log_disagreement
from src.annotation.disagreement_io import save_disagreement_log, append_disagreement_log

__all__ = [
    "CrossModelReviewer",
    "LLMAnnotator",
    "field_level_consensus",
    "resolve_disagreement",
    "build_gold_from_consensus",
    "generate_annotation_stats",
    "normalize_text",
    "coerce_to_triplet",
    "infer_condition_type",
    "log_disagreement",
    "save_disagreement_log",
    "append_disagreement_log",
]
