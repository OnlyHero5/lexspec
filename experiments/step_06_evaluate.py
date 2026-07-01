#!/usr/bin/env python3
"""
LexSpec Experiment 06: Dual-Track Evaluation & Significance Testing
=====================================================================

Step 6 of the experiment pipeline per spec section 8.2.

This script performs comprehensive evaluation across all three
experiment variants (Baseline, Ours-Dep, Ours-Reflexion).  It computes:

  **Track 1 -- Task Metrics (Weighted Triplet F1)**
    - Overall weighted F1 (5 component weights: subject_text 0.35,
      subject_role 0.10, predicate 0.20, object 0.20, condition 0.15)
    - Per-field precision, recall, F1
    - Per-sample F1 scores for significance testing

  **Track 2 -- Linguistic Metrics**
    - Dependency Path Legality Rate
    - Passive Voice Recovery Accuracy
    - Condition Boundary IoU
    - Linguistic Correction Rate (for Ours-Dep and Ours-Reflexion)

  **Track 3 -- Significance Testing**
    - Paired bootstrap (10,000 resamples) for pairwise comparisons
    - Wilcoxon signed-rank test (supplementary)
    - Stratified significance by linguistic phenomenon

Usage
-----
    python experiments/step_06_evaluate.py \\
        --config configs/model.yaml \\
        --output-dir outputs/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# --- Project root on sys.path ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.eval_orchestrator import run_evaluation
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main() -> None:
    """Main entry point: run dual-track evaluation."""
    arg_parser = argparse.ArgumentParser(
        description="LexSpec Experiment 06: Dual-track evaluation & significance testing",
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
        help="Path to the test set JSONL (for UD tree parsing)",
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

    logger.info("Starting dual-track evaluation...")
    t_start = time.perf_counter()

    run_evaluation(
        predictions_dir=args.predictions_dir,
        gold_path=args.gold,
        testset_path=args.testset,
        output_dir=args.output_dir,
        config_path=args.config,
    )

    t_elapsed = time.perf_counter() - t_start
    logger.info("Evaluation complete in %.2f seconds", t_elapsed)


if __name__ == "__main__":
    main()
