"""
LexSpec Corpus Package
======================
Corpus construction from CUAD v1 contract data for LexSpec evaluation.

This package handles:
  - Language phenomenon detection (passive, conditional, etc.) on UD parse trees
  - Loading CUAD v1 data in multiple formats (full contracts, expert spans, QA spans)
  - Clause extraction via Stanza sentence segmentation
  - Test-set selection strategies (stratified sampling, full selection)

Usage::

    from src.corpus import (
        detect_phenomena, is_boilerplate_clause,
        load_cuad_data, load_cuad_spans, load_cuad_qa_spans,
        split_into_clauses, build_clause_records,
        select_all_clauses, select_balanced_testset,
    )
"""

from src.corpus.phenomena_detector import (
    detect_phenomena,
    is_boilerplate_clause,
)

from src.corpus.cuad_loader import (
    load_cuad_data,
    load_cuad_spans,
    load_cuad_qa_spans,
)

from src.corpus.clause_processor import (
    split_into_clauses,
    build_clause_records,
)

from src.corpus.selection import (
    select_all_clauses,
    select_balanced_testset,
)

__all__ = [
    "detect_phenomena",
    "is_boilerplate_clause",
    "load_cuad_data",
    "load_cuad_spans",
    "load_cuad_qa_spans",
    "split_into_clauses",
    "build_clause_records",
    "select_all_clauses",
    "select_balanced_testset",
]
