"""
Evaluation orchestrator: dual-track evaluation across all experiments.

Orchestrates three evaluation tracks:
  - Track 1: Task metrics (weighted triplet F1)
  - Track 2: Linguistic metrics (dependency legality, passive recovery, etc.)
  - Track 3: Significance testing (paired bootstrap, Wilcoxon, stratified)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from src.extraction.schema import (
    LegalTriplet, DependencyTree, ValidationResult,
)
from src.linguistic.stanza_parser import StanzaParser
from src.evaluation.data_loading import (
    load_predictions, load_gold_triplets, load_predictions_as_triplets,
    _load_stanza_config, parse_trees_for_testset,
)
from src.evaluation.reporting import (
    _primary_phenomenon, _write_summary_csv, _print_comparison_table,
)
from src.evaluation.normalization import load_party_aliases
from src.evaluation.triplet_f1 import (
    compute_triplet_f1,
)
from src.evaluation.field_f1 import (
    compute_per_sample_f1,
)
from src.evaluation.linguistic_metrics import (
    compute_all_linguistic_metrics,
)
from src.evaluation.significance import (
    run_all_comparisons, stratified_significance,
)
from src.utils.logging import get_logger
from src.utils.io import read_jsonl, write_json

logger = get_logger(__name__)


def run_evaluation(
    predictions_dir: str,
    gold_path: str,
    testset_path: str,
    output_dir: str,
    config_path: str = "configs/model.yaml",
) -> None:
    """Orchestrate the dual-track evaluation across all experiments.

    This function performs all three evaluation tracks and saves results
    to the ``output_dir/metrics/`` directory.

    Args:
        predictions_dir:  Directory containing ``baseline.jsonl``, etc.
        gold_path:        Path to gold triplets JSONL.
        testset_path:     Path to the test set (for UD parsing).
        output_dir:       Base output directory.
        config_path:      Model config for Stanza initialization.
    """
    pred_dir = Path(predictions_dir)

    # --- Load prediction files ---
    logger.info("Loading predictions...")
    baseline_preds = load_predictions(str(pred_dir / "baseline.jsonl"))
    ours_dep_preds = load_predictions(str(pred_dir / "ours_dep.jsonl"))
    ours_reflex_preds = load_predictions(str(pred_dir / "ours_reflexion.jsonl"))

    # --- Load gold triplets ---
    gold_triplets = load_gold_triplets(gold_path)

    # --- Load test set (for parsing trees) ---
    testset = read_jsonl(testset_path) if Path(testset_path).exists() else []

    # --- Parse UD trees for the test set (needed for linguistic metrics) ---
    trees: List[DependencyTree] = []
    if testset:
        logger.info(
            "Parsing UD trees for %d test clauses (this may take a moment)...",
            len(testset),
        )
        # Initialize Stanza.
        stanza_cfg = _load_stanza_config(config_path)
        nlp_parser = StanzaParser(
            lang=stanza_cfg.get("lang", "en"),
            processors=stanza_cfg.get("processors",
                                       "tokenize,mwt,pos,lemma,depparse"),
            download_method=stanza_cfg.get("download_method", "REUSE_RESOURCES"),
        )
        for clause in tqdm(testset, desc="Parsing test clauses", unit="clause"):
            text = clause.get("text", "")
            if text.strip():
                try:
                    tree = nlp_parser.parse(text)
                    trees.append(tree)
                except Exception as exc:
                    logger.debug("Parse failed for clause: %s", exc)
                    trees.append(DependencyTree(text=text, tokens=[]))  # Empty tree
            else:
                trees.append(DependencyTree(text="", tokens=[]))

    # --- Map experiment names to their prediction data ---
    experiments: Dict[str, List[Dict]] = {
        "baseline": baseline_preds,
        "ours_dep": ours_dep_preds,
        "ours_reflexion": ours_reflex_preds,
    }

    # --- Convert prediction triplets ---
    exp_triplets: Dict[str, List[LegalTriplet]] = {}
    for name, preds in experiments.items():
        exp_triplets[name] = load_predictions_as_triplets(preds)

    # --- Track 1: Task Metrics (Triplet F1) ---
    task_metrics: Dict[str, Dict] = {}
    per_sample_scores: Dict[str, List[float]] = {}

    has_gold = len(gold_triplets) > 0
    if has_gold:
        logger.info("Computing task metrics (weighted triplet F1)...")
        party_aliases = load_party_aliases("configs/constraints.yaml")
        for name, triplets in exp_triplets.items():
            if not triplets:
                logger.warning("No predictions for %s -- skipping task metrics", name)
                task_metrics[name] = {}
                per_sample_scores[name] = []
                continue

            # Ensure length matches gold.
            n = min(len(triplets), len(gold_triplets))
            if len(triplets) != len(gold_triplets):
                logger.warning(
                    "%s predictions (%d) != gold (%d) -- truncating to %d",
                    name, len(triplets), len(gold_triplets), n,
                )

            f1_metrics = compute_triplet_f1(
                predictions=triplets[:n],
                gold=gold_triplets[:n],
                party_aliases=party_aliases,
            )
            task_metrics[name] = f1_metrics

            # Per-sample F1 for significance testing.
            ps_scores = compute_per_sample_f1(
                predictions=triplets[:n],
                gold=gold_triplets[:n],
                party_aliases=party_aliases,
            )
            per_sample_scores[name] = ps_scores
    else:
        logger.warning(
            "No gold triplets available -- task metrics will be empty."
        )
        for name in experiments:
            task_metrics[name] = {}
            per_sample_scores[name] = []

    # --- Track 2: Linguistic Metrics ---
    linguistic_metrics: Dict[str, Dict] = {}
    if trees and has_gold and len(trees) == len(gold_triplets):
        logger.info("Computing linguistic metrics...")

        for name in ["baseline", "ours_dep", "ours_reflexion"]:
            triplets = exp_triplets.get(name, [])
            if not triplets:
                linguistic_metrics[name] = {}
                continue

            n = min(len(triplets), len(gold_triplets), len(trees))

            # Load validation results for correction rate (ours_dep/ours_reflex).
            val_results: Optional[List[ValidationResult]] = None
            if name in ("ours_dep", "ours_reflexion"):
                val_path = pred_dir / f"{name}_validations.jsonl"
                if val_path.exists():
                    try:
                        from src.utils.io import load_pydantic_list
                        val_results = load_pydantic_list(str(val_path), ValidationResult)
                        val_results = val_results[:n]
                    except Exception as exc:
                        logger.debug("Could not load validation results: %s", exc)

            try:
                ling = compute_all_linguistic_metrics(
                    predictions=triplets[:n],
                    gold=gold_triplets[:n],
                    trees=trees[:n],
                    validation_results=val_results,
                )
                linguistic_metrics[name] = ling
            except Exception as exc:
                logger.error(
                    "Linguistic metrics computation failed for %s: %s", name, exc,
                )
                linguistic_metrics[name] = {"error": str(exc)}
    else:
        logger.warning(
            "UD trees not available or mismatched lengths -- "
            "linguistic metrics will be empty."
        )
        for name in experiments:
            linguistic_metrics[name] = {}

    # --- Track 3: Significance Testing ---
    significance_results: Dict[str, Any] = {}
    if len(per_sample_scores) >= 2 and all(
        len(scores) > 0 for scores in per_sample_scores.values()
    ):
        logger.info("Running significance tests...")

        # Pairwise comparisons.
        try:
            sig = run_all_comparisons(
                experiment_results={
                    k: v for k, v in per_sample_scores.items() if v
                },
                n_resamples=10000,
                random_seed=42,
            )
            significance_results = sig
        except Exception as exc:
            logger.error("Significance testing failed: %s", exc)
            significance_results = {"error": str(exc)}

        # Stratified significance by phenomenon (if testset has phenomenon labels).
        if testset and all("phenomena" in c for c in testset):
            stratified_results: Dict[str, Any] = {}
            phen_names = ["passive", "conditional", "relative_clause",
                          "long_distance", "negation"]
            # Only compare baseline vs ours_reflexion for brevity.
            if ("baseline" in per_sample_scores
                    and "ours_reflexion" in per_sample_scores
                    and len(per_sample_scores["baseline"]) == len(testset)):
                labels = [
                    _primary_phenomenon(c.get("phenomena", {}))
                    for c in testset[:len(per_sample_scores["baseline"])]
                ]
                for phen in phen_names:
                    try:
                        strat = stratified_significance(
                            scores_a=per_sample_scores["baseline"],
                            scores_b=per_sample_scores["ours_reflexion"],
                            phenomenon_labels=labels,
                            phenomenon_name=phen,
                            n_resamples=10000,
                            random_seed=42,
                        )
                        stratified_results[phen] = strat
                    except Exception as exc:
                        logger.debug(
                            "Stratified significance failed for '%s': %s", phen, exc,
                        )
                        stratified_results[phen] = {"error": str(exc)}
            significance_results["stratified"] = stratified_results
    else:
        logger.warning(
            "Insufficient per-sample scores for significance testing."
        )

    # --- Save all results ---
    metrics_dir = Path(output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Save task metrics.
    write_json(str(metrics_dir / "task_metrics.json"), task_metrics)

    # Save linguistic metrics.
    write_json(str(metrics_dir / "linguistic_metrics.json"), linguistic_metrics)

    # Save significance results.
    write_json(str(metrics_dir / "significance.json"), significance_results)

    # Write summary CSV.
    _write_summary_csv(task_metrics, linguistic_metrics, significance_results,
                       str(metrics_dir / "summary.csv"))

    # --- Print comprehensive comparison table ---
    _print_comparison_table(task_metrics, linguistic_metrics, significance_results)
