"""
LexSpec — Computational Linguistics for Legal Contract Analysis
================================================================

LexSpec is a computational linguistics project that analyzes legal contract
clauses to extract (subject, action, condition) triplets. It employs LLM-based
extraction with post-hoc validation via Universal Dependencies syntactic
constraints.

Pipeline Overview:
  - extraction:    LLM-based clause triple extraction
  - linguistic:    UD-based syntactic analysis and validation
  - correction:    Post-hoc constraint-based corrections
  - annotation:    Multi-annotator pipeline and gold standard construction
  - evaluation:    Error analysis and performance metrics

Author:  LexSpec Team
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "LexSpec Team"
__description__ = (
    "Computational linguistics pipeline for extracting (subject, action, condition) "
    "triplets from legal contract clauses using LLMs with UD-based syntactic validation."
)
__project_name__ = "lexspec"
