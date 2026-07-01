#!/usr/bin/env python3
"""
LexSpec Experiment 01: Build evaluation corpus from CUAD data
==============================================================

Build the LexSpec evaluation corpus from CUAD v1 contract data. Supports two modes:

  - Stratified sampling mode (default 100 clauses): balanced sampling by
    phenomenon quotas to ensure passive voice, conditional clauses, relative
    clauses, long-distance dependencies, and negation each have sufficient
    samples for statistical comparison.
  - Full mode (--all): use all valid clauses from all 510 contracts.

Usage::

    python experiments/step_01_build_corpus.py
    python experiments/step_01_build_corpus.py --all
    python experiments/step_01_build_corpus.py --source spans --all \\
        --output data/processed/lexspec_spans.jsonl
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.linguistic.stanza_parser import StanzaParser
from src.corpus.phenomena_detector import is_boilerplate_clause
from src.corpus.cuad_loader import load_cuad_data, load_cuad_spans, load_cuad_qa_spans
from src.corpus.clause_processor import split_into_clauses, build_clause_records
from src.corpus.selection import select_all_clauses, select_balanced_testset
from src.utils.logging import setup_logging, get_logger
from src.utils.io import write_jsonl

logger = get_logger(__name__)


# ======================================================================
# Main entry point
# ======================================================================


def main() -> None:
    """Main entry point: Build the LexSpec evaluation corpus from CUAD data."""
    parser = argparse.ArgumentParser(
        description="LexSpec Experiment 01: Build evaluation corpus from CUAD data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Full mode: use all valid clauses from all 510 contracts, no "
            "stratified sampling. Auto-sets --selection-mode all --max-contexts 0 "
            "--target-count 0, default output to data/processed/lexspec_corpus.jsonl"
        ),
    )
    parser.add_argument(
        "--source",
        choices=["sentences", "spans", "qa_spans"],
        default="sentences",
        help=(
            "Clause source: sentences=Stanza sentence segmentation from full "
            "contracts (largest); spans=master_clauses.csv expert fragments (~5k); "
            "qa_spans=CUAD JSON answer spans (~13k)"
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=["stratified", "all"],
        default="stratified",
        help="stratified=balanced stratified sampling; all=all valid parsed clauses",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model.yaml",
        help="Model config file path (default: configs/model.yaml)",
    )
    parser.add_argument(
        "--cuad-path",
        type=str,
        default="data/raw/CUAD_v1/CUAD_v1.json",
        help="CUAD v1 JSON file path",
    )
    parser.add_argument(
        "--master-clauses-path",
        type=str,
        default="data/raw/CUAD_v1/master_clauses.csv",
        help="CUAD master_clauses.csv path (used with --source spans)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/lexspec_100.jsonl",
        help="Output path for the selected test set",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=100,
        help="Target clause count for stratified mode (0 = no cap, use all parsed)",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=0,
        help="Max contract paragraphs to process (0 = all 510; default all)",
    )
    parser.add_argument(
        "--min-passive",
        type=int,
        default=20,
        help="Minimum passive voice clauses (default: 20)",
    )
    parser.add_argument(
        "--min-conditional",
        type=int,
        default=20,
        help="Minimum conditional clause clauses (default: 20)",
    )
    parser.add_argument(
        "--min-relative",
        type=int,
        default=15,
        help="Minimum relative clause clauses (default: 15)",
    )
    parser.add_argument(
        "--min-long-distance",
        type=int,
        default=20,
        help="Minimum long-distance dependency clauses (default: 20)",
    )
    parser.add_argument(
        "--min-negation",
        type=int,
        default=15,
        help="Minimum negation clauses (default: 15)",
    )
    args = parser.parse_args()

    # --all shortcut mode.
    if args.all:
        args.selection_mode = "all"
        args.max_contexts = 0
        args.target_count = 0
        if args.output == "data/processed/lexspec_100.jsonl":
            args.output = "data/processed/lexspec_corpus.jsonl"

    # ---- Initialization ----
    import logging as _logging
    setup_logging(log_dir="outputs", level=_logging.INFO)

    logger.info("Loading model config: %s", args.config)
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
    else:
        logger.warning("Config file '%s' not found -- using Stanza defaults", args.config)
        config = {}

    stanza_cfg = config.get("stanza", {}) if config else {}
    stanza_lang = stanza_cfg.get("lang", "en")
    stanza_processors = stanza_cfg.get("processors", "tokenize,mwt,pos,lemma,depparse")
    stanza_download = stanza_cfg.get("download_method", "REUSE_RESOURCES")

    # ---- Initialize Stanza parser ----
    logger.info(
        "Initializing Stanza: lang=%s, processors=%s, download=%s",
        stanza_lang, stanza_processors, stanza_download,
    )
    nlp_parser = StanzaParser(
        lang=stanza_lang,
        processors=stanza_processors,
        download_method=stanza_download,
    )

    # ---- Load CUAD clauses ----
    if args.source == "spans":
        clauses = load_cuad_spans(args.master_clauses_path)
        source_label = "cuad_v1_spans"
    elif args.source == "qa_spans":
        clauses = load_cuad_qa_spans(args.cuad_path)
        source_label = "cuad_v1_qa"
    else:
        contexts = load_cuad_data(args.cuad_path)
        max_ctx = args.max_contexts if args.max_contexts > 0 else None
        clauses = split_into_clauses(nlp_parser, contexts, max_contexts=max_ctx)
        source_label = "cuad_v1_sentences"

    # ---- Clause selection ----
    if args.selection_mode == "all":
        records, _ = build_clause_records(nlp_parser, clauses, source_label)
        selected = select_all_clauses(records)
    else:
        target = args.target_count if args.target_count > 0 else 100
        selected = select_balanced_testset(
            parser=nlp_parser,
            clauses=clauses,
            target_count=target,
            min_passive=args.min_passive,
            min_conditional=args.min_conditional,
            min_relative=args.min_relative,
            min_long_distance=args.min_long_distance,
            min_negation=args.min_negation,
            source_label=source_label,
        )

    # ---- Save output ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(output_path), selected)
    logger.info("Saved %d clauses to: %s", len(selected), output_path)

    # ---- Print selection statistics ----
    print("\n" + "=" * 60)
    print("LexSpec Corpus Construction Complete")
    print("=" * 60)
    print(f"Source:          {args.source} ({source_label})")
    print(f"Selection mode:  {args.selection_mode}")
    print(f"Total clauses:   {len(selected)}")
    print()

    phen_names = [
        "passive", "conditional", "relative_clause",
        "long_distance", "negation",
    ]
    print(f"{'Phenomenon':<25s} {'Count':>6s}  {'Pct':>10s}")
    print("-" * 43)
    for phen in phen_names:
        count = sum(1 for r in selected if r["phenomena"][phen])
        pct = count / len(selected) * 100 if selected else 0
        print(f"  {phen:<23s} {count:>6d}  {pct:>9.1f}%")
    print()

    # Show filtered definition clause count in full mode.
    if args.selection_mode == "all":
        defs_filtered = sum(
            1 for r in records if r["phenomena"].get("is_definition", False)
        )
        if defs_filtered:
            print(f"  (Filtered {defs_filtered} definition clauses)")
    print()

    # Multi-phenomenon overlap statistics (exclude is_definition).
    print("Multi-phenomenon clause distribution:")
    multi = Counter(
        sum(1 for k, v in r["phenomena"].items() if k != "is_definition" and v)
        for r in selected
    )
    for k in sorted(multi):
        print(f"  {k} phenomena: {multi[k]} clauses")
    print("=" * 60)


if __name__ == "__main__":
    main()
