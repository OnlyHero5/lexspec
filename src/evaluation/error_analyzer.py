"""
Linguistic error analysis with two-level classification.

Per PDF requirement: "必须从语言学角度分类归因，每个错误案例需附上语言学解释"
(Must classify from linguistic perspective, each error case with linguistic
explanation citing specific UD relations).

Classification system:

  Primary (linguistic phenomenon) — which syntactic construction caused the error:
    - Passive Voice Error:         nsubj:pass instead of nsubj; patient confused with agent
    - Conditional Boundary Error:  advcl/mark scope misidentified
    - Relative Clause Confusion:   acl:relcl embedding confused the extractor
    - Long-distance Dependency:    Dependency path > 3 edges between predicate and argument
    - Negation/Exception Error:    Negation particle or "except/unless" altered the role
    - Other Error:                 Catch-all

  Secondary (field error type) — which field(s) were affected:
    - Subject Error:               subject.text or subject.role incorrect
    - Role Error:                  subject.role incorrect (text may be correct)
    - Predicate Error:             action.predicate incorrect
    - Object Error:                action.object incorrect
    - Condition Omission:          condition missing when one should exist
    - Condition Over-extension:    condition text extends beyond correct boundary

Error cases are serializable to JSONL for reporting and can be cross-tabulated
to produce error distribution statistics.
"""

from __future__ import annotations

import os
from typing import Optional, List, Dict, Any

from src.extraction.schema import (
    LegalTriplet, DependencyTree, Token, ValidationResult,
    ErrorCase, ErrorCategory, FieldErrorType,
)
from src.evaluation.error_detection import detect_field_errors, determine_secondary_category
from src.evaluation.error_classification import determine_primary_category
from src.evaluation.error_summary import generate_error_id
from src.evaluation.error_explanations import generate_explanation
from src.utils.io import append_jsonl, ensure_dir, write_json
from src.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Single Error Case Generation
# =============================================================================


def generate_error_report(
    prediction: LegalTriplet,
    gold: LegalTriplet,
    tree: Optional[DependencyTree] = None,
    validation_result: Optional[ValidationResult] = None,
    error_id: Optional[str] = None,
) -> Optional[ErrorCase]:
    """Generate a detailed error analysis case with linguistic explanation.

    This is the core analysis function. For a single (prediction, gold) pair,
    it:
      1. Compares each field to determine the secondary (field-level) category.
      2. Analyzes the UD tree (if available) to determine the primary
         (linguistic phenomenon) category.
      3. Generates a bilingual (Chinese + English) linguistic explanation
         citing specific UD relations.

    Only generates a report if there are actual errors (any field mismatch).
    If the prediction perfectly matches the gold, returns None.

    Classification logic for primary category:

    Rule 1 — Passive Voice Error:
        - Tree has nsubj:pass + aux:pass (passive detected)
        - AND predicted subject.text has high overlap with nsubj:pass text
          (the patient) while gold subject matches obl:agent (the true agent)
        -> The LLM confused syntactic subject (patient) with legal subject (agent).

    Rule 2 — Conditional Boundary Error:
        - advcl relation exists in tree
        - AND predicted condition tokens have low IoU (< 0.5) with UD condition span
        -> The LLM misidentified the scope of the condition clause.

    Rule 3 — Relative Clause Confusion:
        - acl:relcl relation exists in tree
        - AND either the predicate or subject/object text matches tokens
          inside the relative clause subtree
        -> The LLM extracted from inside a relative clause instead of the
           main clause.

    Rule 4 — Long-distance Dependency Error:
        - Get the dependency distance between the root predicate and its
          subject/object in the tree
        - When distance > 3 AND the prediction has errors on the distant argument
        -> Syntactic distance made extraction difficult.

    Rule 5 — Negation/Exception Error:
        - neg relation exists in tree
        - OR condition type is EXCEPTION
        - AND subject.role is incorrect (should be prohibited_party, got obligor)
        -> Negation or exception clause confused the legal role assignment.

    Rule 6 — Other Error:
        - No specific linguistic pattern detected; fallback classification.

    Args:
        prediction: System prediction (may be corrected or raw).
        gold: Gold-standard triplet.
        tree: UD dependency tree (optional, enables richer linguistic analysis).
        validation_result: Validation result (optional, provides correction
                           evidence for richer explanation).
        error_id: Optional error identifier. If None, auto-generated as "E-{index}".

    Returns:
        ErrorCase with full two-level classification and bilingual explanation,
        or None if no error (prediction matches gold on all fields).

    Example:
        >>> err = generate_error_report(pred, gold, tree)
        >>> if err:
        ...     print(err.linguistic_explanation)
        # 被动语态错误 (Passive Voice Error): The system incorrectly ...
    """
    # -------------------------------------------------------------------
    # Step 1: Detect whether any error exists.
    # -------------------------------------------------------------------
    field_errors = detect_field_errors(prediction, gold)
    if not field_errors:
        # Perfect match — no error report needed.
        return None

    # -------------------------------------------------------------------
    # Step 2: Determine secondary category (field-level).
    # -------------------------------------------------------------------
    secondary = determine_secondary_category(field_errors)

    # -------------------------------------------------------------------
    # Step 3: Determine primary category (linguistic phenomenon).
    # -------------------------------------------------------------------
    primary = ErrorCategory.OTHER_ERROR  # Default fallback
    ud_evidence: Dict[str, Any] = {}     # Collect UD evidence for explanation.

    if tree is not None and tree.token_count > 0:
        primary, ud_evidence = determine_primary_category(
            prediction, gold, tree, field_errors,
        )

    # -------------------------------------------------------------------
    # Step 4: Generate bilingual linguistic explanation.
    # -------------------------------------------------------------------
    explanation = generate_explanation(
        prediction, gold, tree, primary, secondary, field_errors, ud_evidence,
        validation_result,
    )

    # -------------------------------------------------------------------
    # Step 5: Assemble the ErrorCase.
    # -------------------------------------------------------------------
    error_case = ErrorCase(
        error_id=error_id or generate_error_id(),
        text=gold.subject.text if gold.subject.text else "[no text]",
        prediction=prediction,
        gold=gold,
        primary_category=primary,
        secondary_category=secondary,
        linguistic_explanation=explanation,
    )

    return error_case


# =============================================================================
# Batch Error Classification
# =============================================================================


def classify_errors(
    predictions: List[LegalTriplet],
    gold: List[LegalTriplet],
    trees: Optional[List[DependencyTree]] = None,
    validation_results: Optional[List[ValidationResult]] = None,
) -> List[ErrorCase]:
    """Classify all errors across the full prediction set.

    Processes each (prediction, gold) pair, generates an ErrorCase
    for pairs with errors, and returns the full list. This is the
    primary batch entry point for error analysis.

    Args:
        predictions: List of predicted LegalTriplets.
        gold: List of gold-standard LegalTriplets (same length).
        trees: Optional list of DependencyTree objects (same length).
               When provided, enables richer primary category classification.
        validation_results: Optional list of ValidationResult objects (same length).
                            When provided, enriches explanations with correction
                            evidence.

    Returns:
        List of ErrorCase objects for all samples with errors. Empty list
        if all predictions match gold perfectly.

    Raises:
        ValueError: If input lists have inconsistent lengths.
    """
    n = len(predictions)
    if len(gold) != n:
        raise ValueError(
            f"predictions and gold must have the same length. "
            f"Got {len(predictions)} and {len(gold)}."
        )
    if trees is not None and len(trees) != n:
        raise ValueError(
            f"trees must have the same length as predictions. "
            f"Got {len(trees)} vs {n}."
        )
    if validation_results is not None and len(validation_results) != n:
        raise ValueError(
            f"validation_results must have the same length as predictions. "
            f"Got {len(validation_results)} vs {n}."
        )

    error_cases: List[ErrorCase] = []

    for i in range(n):
        pred = predictions[i]
        g = gold[i]
        t = trees[i] if trees is not None else None
        vr = validation_results[i] if validation_results is not None else None

        error_case = generate_error_report(
            prediction=pred,
            gold=g,
            tree=t,
            validation_result=vr,
            error_id=f"E-{i + 1:04d}",
        )

        if error_case is not None:
            error_cases.append(error_case)

    logger.info(
        "Error classification complete: %d errors found out of %d samples (%.1f%%)",
        len(error_cases), n, (len(error_cases) / n * 100) if n > 0 else 0,
    )

    return error_cases


# =============================================================================
# Error Case Persistence
# =============================================================================


def save_error_cases(
    error_cases: List[ErrorCase],
    output_dir: str = "outputs/error_cases",
) -> None:
    """Save error cases to categorized JSONL files.

    Creates separate JSONL files per primary error category, plus an
    aggregate file with all errors. This structure facilitates targeted
    analysis of specific error types.

    Output files:
      - passive_voice_errors.jsonl
      - conditional_boundary_errors.jsonl
      - relative_clause_errors.jsonl
      - long_distance_dependency_errors.jsonl
      - negation_exception_errors.jsonl
      - other_errors.jsonl
      - all_errors.jsonl (complete set)

    Args:
        error_cases: List of ErrorCase objects to save.
        output_dir: Directory path for output files. Created if it does not
                    exist. Defaults to "outputs/error_cases".
    """
    if not error_cases:
        logger.info("No error cases to save.")
        return

    ensure_dir(output_dir)

    # Map primary category values to output filenames.
    category_files: Dict[str, str] = {
        "passive_voice": "passive_voice_errors.jsonl",
        "conditional_boundary": "conditional_boundary_errors.jsonl",
        "relative_clause": "relative_clause_errors.jsonl",
        "long_distance_dependency": "long_distance_dependency_errors.jsonl",
        "negation_exception": "negation_exception_errors.jsonl",
        "other": "other_errors.jsonl",
    }

    # Track counts per category for logging.
    counts: Dict[str, int] = {cat: 0 for cat in category_files}

    for case in error_cases:
        # Serialize to dict using Pydantic's model_dump for JSON compatibility.
        record = case.model_dump(mode="json")

        # Append to category-specific file.
        category = case.primary_category.value
        filename = category_files.get(category, "other_errors.jsonl")
        filepath = os.path.join(output_dir, filename)
        append_jsonl(filepath, record)
        counts[category] = counts.get(category, 0) + 1

        # Append to aggregate file.
        all_filepath = os.path.join(output_dir, "all_errors.jsonl")
        append_jsonl(all_filepath, record)

    # Log summary of files written.
    for cat, count in counts.items():
        if count > 0:
            logger.info("Saved %d errors to %s/%s", count, output_dir, category_files[cat])

    # Write a summary index file with counts.
    summary_path = os.path.join(output_dir, "error_summary.json")
    write_json(summary_path, {
        "total_errors": len(error_cases),
        "categories": {cat: count for cat, count in counts.items() if count > 0},
    })

    logger.info(
        "Error cases saved to %s: %d total across %d categories",
        output_dir, len(error_cases), sum(1 for c in counts.values() if c > 0),
    )
