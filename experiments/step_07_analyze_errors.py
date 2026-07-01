#!/usr/bin/env python3
"""
LexSpec Experiment 07: Error Analysis -- Linguistic Error Classification
==========================================================================

Step 7 of the experiment pipeline per spec section 8.2.

This script performs comprehensive linguistic error analysis across all
three experiment variants.  For each experiment, it:

  1. Compares predicted triplets against gold-standard triplets.
  2. Classifies errors using a two-level taxonomy:
     - **Primary category** (linguistic phenomenon):
       passive_voice, conditional_boundary, relative_clause,
       long_distance_dependency, negation_exception, other
     - **Secondary category** (field error type):
       subject, role, predicate, object, condition_omission,
       condition_overextension
  3. Generates bilingual (Chinese + English) linguistic explanations
     citing specific UD dependency relations.
  4. Produces error distribution reports with cross-tabulations.
  5. Saves categorized error cases to ``outputs/error_cases/``.

Usage
-----
    python experiments/step_07_analyze_errors.py \\
        --config configs/model.yaml \\
        --output-dir outputs/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Any

import yaml
from tqdm import tqdm

# --- Project root on sys.path ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.extraction.schema import (
    LegalTriplet, DependencyTree, ValidationResult, ErrorCase,
    ErrorCategory, FieldErrorType,
)
from src.evaluation.data_loading import (
    load_predictions, load_gold_triplets, parse_trees_for_testset,
)
from src.evaluation.error_reporting import (
    analyze_experiment_errors, print_error_comparison_table,
)
from src.utils.logging import setup_logging, get_logger
from src.utils.io import read_jsonl, write_jsonl, write_json, load_pydantic_list

logger = get_logger(__name__)


def main() -> None:
    """Main entry point: run error analysis across all experiments."""
    arg_parser = argparse.ArgumentParser(
        description="LexSpec Experiment 07: Linguistic error analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arg_parser.add_argument(
        "--config",
        type=str,
        default="configs/model.yaml",
        help="Path to model configuration YAML",
    )
    arg_parser.add_argument(
        "--predictions-dir",
        type=str,
        default="outputs/predictions",
        help="Directory containing prediction JSONL files",
    )
    arg_parser.add_argument(
        "--gold",
        type=str,
        default="data/processed/gold_triplets.jsonl",
        help="Path to gold triplets JSONL",
    )
    arg_parser.add_argument(
        "--testset",
        type=str,
        default="data/processed/lexspec_100.jsonl",
        help="Path to test set JSONL (for UD parsing)",
    )
    arg_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for output files",
    )
    args = arg_parser.parse_args()

    # --- Setup logging ---
    import logging as _logging
    setup_logging(log_dir=str(Path(args.output_dir) / "logs"), level=_logging.INFO)

    # --- Load gold triplets ---
    gold_triplets = load_gold_triplets(args.gold)
    if not gold_triplets:
        logger.error(
            "Cannot run error analysis without gold triplets. "
            "Generate gold labels first via the annotation pipeline."
        )
        print(
            "\nWARNING: Gold triplets not found at '%s'.\n"
            "Error analysis requires gold-standard labels to compare against.\n"
            "Run the annotation pipeline (src/annotation/) to generate gold triplets,\n"
            "or set --gold to point at an existing gold file.\n"
            % args.gold
        )
        sys.exit(0)  # Not a hard failure -- graceful exit.

    # --- Parse UD trees ---
    trees = parse_trees_for_testset(
        testset_path=args.testset,
        config_path=args.config,
    )

    # --- Load predictions for each experiment ---
    pred_dir = Path(args.predictions_dir)
    experiments = {
        "baseline": load_predictions(str(pred_dir / "baseline.jsonl")),
        "ours_dep": load_predictions(str(pred_dir / "ours_dep.jsonl")),
        "ours_reflexion": load_predictions(str(pred_dir / "ours_reflexion.jsonl")),
    }

    # --- Run error analysis for each experiment ---
    error_dir = Path(args.output_dir) / "error_cases"
    error_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict[str, Any]] = {}

    for exp_name, preds in experiments.items():
        if not preds:
            logger.warning("No predictions for %s -- skipping error analysis", exp_name)
            all_results[exp_name] = {
                "experiment": exp_name,
                "total_samples": 0,
                "error_count": 0,
                "error_rate": 0.0,
                "primary_distribution": {},
                "secondary_distribution": {},
            }
            continue

        result = analyze_experiment_errors(
            experiment_name=exp_name,
            predictions=preds,
            gold=gold_triplets,
            trees=trees,
            output_dir=str(error_dir),
        )
        all_results[exp_name] = result

    # --- Print cross-experiment comparison table ---
    print_error_comparison_table(all_results)

    # --- Save aggregate summary ---
    aggregate_summary = {
        "experiments": all_results,
        "total_errors_by_experiment": {
            k: v.get("error_count", 0) for k, v in all_results.items()
        },
    }
    summary_path = error_dir / "error_summary.json"
    write_json(str(summary_path), aggregate_summary)
    logger.info("Aggregate error summary saved to: %s", summary_path)

    print("\nError analysis files saved to: %s" % error_dir)
    for exp_name in ["baseline", "ours_dep", "ours_reflexion"]:
        exp_dir = error_dir / exp_name
        if exp_dir.exists():
            jsonl_files = sorted(exp_dir.glob("*.jsonl"))
            if jsonl_files:
                print(f"  {exp_name}/:")
                for f in jsonl_files:
                    print(f"    {f.name}")


if __name__ == "__main__":
    main()
