"""
Condition Clause Boundary Extraction
=====================================
Extracts condition/exception/temporal clauses from UD dependency trees
by identifying advcl+mark patterns and extracting full subtree spans.

Submodules:
  _marker_config:   Fallback taxonomy, YAML config loading, marker list
  _extractor:       Primary extraction interface (extract, extract_all, classify)
  _overlap:         Condition span overlap calculation
"""

from __future__ import annotations

from src.linguistic.condition._marker_config import (
    _load_markers as _load_markers_fn,
    _parse_markers_section as _parse_markers_section_fn,
)
from src.linguistic.condition._extractor import (
    extract as _extract_fn,
    extract_all as _extract_all_fn,
    _classify_condition as _classify_condition_fn,
)
from src.linguistic.condition._overlap import (
    compute_condition_overlap as _compute_condition_overlap_fn,
    is_condition_in_main_clause as _is_condition_in_main_clause_fn,
)


class ConditionExtractor:
    """Extract and classify condition clauses from UD dependency trees.

    Loads condition marker taxonomy from configs/constraints.yaml and
    uses advcl+mark UD patterns to identify condition clause boundaries.
    Classifies conditions into TRIGGER, TEMPORAL, or EXCEPTION based on
    the mark word and the legal-domain taxonomy.

    This module is used by the validator (Step 5) to:
      1. Check if the LLM correctly identified the presence/absence of a condition.
      2. Validate the condition clause boundary (span accuracy).
      3. Classify error types: condition omission, over-extraction, boundary error.

    Usage:
        extractor = ConditionExtractor("configs/constraints.yaml")
        spans = extractor.extract_all(tree, predicate_idx)
        if spans:
            print(f"Found {spans[0].condition_type} condition: {spans[0].text}")
    """

    # Imported from submodules — bound as instance/static methods:
    _load_markers = _load_markers_fn
    _parse_markers_section = staticmethod(_parse_markers_section_fn)
    extract = _extract_fn
    extract_all = _extract_all_fn
    _classify_condition = _classify_condition_fn
    compute_condition_overlap = staticmethod(_compute_condition_overlap_fn)
    is_condition_in_main_clause = staticmethod(_is_condition_in_main_clause_fn)

    def __init__(self, constraints_path: str = "configs/constraints.yaml"):
        """Initialize with constraints configuration.

        Args:
            constraints_path: Path to constraints YAML configuration file.
                             Must exist and be well-formed — no fallback.

        Raises:
            FileNotFoundError: If constraints_path does not exist.
            KeyError: If the condition_markers section is missing.
        """
        self._markers: dict = {}
        self._marker_list: list = []
        self._load_markers(constraints_path)

    @property
    def marker_list(self) -> list:
        """Return the flat list of all known condition marker strings.

        This list is passed to find_advcl_with_mark() in ud_features.py
        to identify which advcl children are condition clauses.
        """
        return list(self._marker_list)
