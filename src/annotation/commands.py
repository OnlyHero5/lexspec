"""
Command implementations and CLI builder for the annotation pipeline.

Exposes three subcommands:
  - annotate:  Run independent annotation with a single model.
  - review:    Cross-review one model's annotations using another model.
  - merge:     Merge annotations and review results into gold standard.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.annotation.cli_utils import (
    load_config, build_client, model_role_name, output_path_for_model,
    review_output_path, triplet_to_dict, dict_to_triplet,
    _deduplicate_annotations, _load_completed_ids, MODEL_ALIASES,
    DEFAULT_TESTSET, GEMMA_ANNOT, QWEN_ANNOT, QWEN_REVIEW_GEMMA,
    GEMMA_REVIEW_QWEN, GOLD_OUT, DISAGREE_OUT,
)
from src.annotation.task_pool import _run_task_pool
from src.annotation.merge_engine import cmd_merge
from src.annotation.llm_annotator import LLMAnnotator
from src.annotation.reviewer import CrossModelReviewer
from src.utils.io import read_jsonl, ensure_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# Stage 1 + 2: Annotation
# ======================================================================


def cmd_annotate(args: argparse.Namespace) -> None:
    """Run independent annotation: annotate all clauses in the test set
    with a single model.

    Supports --resume for checkpoint/resume (skips already-annotated clauses).
    Deduplicates after completion to remove duplicates from resume.
    """
    config = load_config(args.config)

    testset_path = Path(args.input)
    if not testset_path.exists():
        logger.error("Test set not found: %s", testset_path)
        logger.error("Please run first: python experiments/step_01_build_corpus.py")
        sys.exit(1)

    output_path = args.output or output_path_for_model(args.model)
    ensure_dir(Path(output_path).parent)

    clauses = read_jsonl(str(testset_path))
    done_ids = _load_completed_ids(output_path) if args.resume else set()
    model_name = model_role_name(config, args.model)
    model_role = MODEL_ALIASES.get(args.model, args.model)
    workers = max(1, int(args.workers))

    pending = [c for c in clauses if c.get("clause_id") not in done_ids]
    logger.info(
        "Annotation %s: total=%d done=%d pending=%d workers=%d -> %s",
        model_role, len(clauses), len(done_ids), len(pending), workers, output_path,
    )

    if not pending:
        print(f"All {len(clauses)} clauses already annotated ({output_path})")
        removed = _deduplicate_annotations(output_path)
        if removed:
            print(f"Dedup: removed {removed} duplicate records from {output_path}")
        return

    def annotate_one(clause: dict, _idx: int, _total: int) -> dict:
        """Single-annotation worker: each thread creates its own client and annotator."""
        client = build_client(config, args.model)
        annotator = LLMAnnotator(client, prompts_path=args.prompts)
        clause_id = clause.get("clause_id", "UNK")
        text = clause.get("text", "")
        t0 = time.time()
        record: Dict[str, Any] = {
            "clause_id": clause_id,
            "text": text,
            "phenomena": clause.get("phenomena", {}),
            "source": clause.get("source", "cuad_v1"),
            "model": model_name,
            "model_role": model_role,
            "annotated_at": datetime.now(timezone.utc).isoformat(),
            "success": False,
        }
        try:
            triplet = annotator.annotate(text)
            record["triplet"] = triplet_to_dict(triplet)
            record["success"] = True
            record["elapsed_s"] = round(time.time() - t0, 2)
            logger.info(
                "%s success (%.1fs) subject=%s pred=%s",
                clause_id, record["elapsed_s"],
                triplet.subject.text, triplet.action.predicate,
            )
        except Exception as exc:
            record["error"] = str(exc)
            record["elapsed_s"] = round(time.time() - t0, 2)
            logger.error("%s failed: %s", clause_id, exc)
        return record

    _run_task_pool(
        pending, annotate_one, output_path, workers,
        f"Annotate-{model_role}",
        show_progress=not args.no_progress,
    )

    removed = _deduplicate_annotations(output_path)

    print(f"\nAnnotation results saved to: {output_path}")
    print(f"This round: {len(pending)} clauses (workers={workers})")
    if removed:
        print(f"Dedup: removed {removed} duplicate records")


# ======================================================================
# Stage 3: Cross-review
# ======================================================================


def cmd_review(args: argparse.Namespace) -> None:
    """Run cross-review: use one model to review another model's annotations.

    The reviewing model judges each field as accept/reject and provides
    corrections for problematic fields.
    """
    config = load_config(args.config)
    reviewer_role = MODEL_ALIASES.get(args.reviewer, args.reviewer)
    reviewer_name = model_role_name(config, args.reviewer)
    workers = max(1, int(args.workers))

    source_path = args.source_file or output_path_for_model(args.source)
    if not Path(source_path).exists():
        logger.error("Source annotation file not found: %s", source_path)
        sys.exit(1)

    output_path = args.output or review_output_path(args.reviewer, args.source)
    ensure_dir(Path(output_path).parent)

    source_records = read_jsonl(source_path)
    done_ids = _load_completed_ids(output_path) if args.resume else set()
    pending = [r for r in source_records if r.get("clause_id") not in done_ids]

    logger.info(
        "Review %s -> checking %s: total=%d done=%d pending=%d workers=%d -> %s",
        reviewer_role, args.source, len(source_records),
        len(done_ids), len(pending), workers, output_path,
    )

    if not pending:
        print(f"All reviews complete ({output_path})")
        return

    def review_one(src: dict, _idx: int, _total: int) -> dict:
        """Single-review worker function."""
        client = build_client(config, args.reviewer)
        reviewer = CrossModelReviewer(
            client, reviewer_role=reviewer_role, prompts_path=args.prompts,
        )
        clause_id = src.get("clause_id", "UNK")
        text = src.get("text", "")
        triplet = dict_to_triplet(src["triplet"])
        source_model = src.get("model_role") or src.get("model") or args.source
        t0 = time.time()
        record: Dict[str, Any] = {
            "clause_id": clause_id,
            "text": text,
            "source_model": source_model,
            "source_model_name": src.get("model", ""),
            "reviewer_model": reviewer_name,
            "reviewer_role": reviewer_role,
            "source_triplet": src["triplet"],
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "success": False,
        }
        try:
            result = reviewer.review(text, triplet, source_model)
            record["verdict"] = result.verdict
            record["field_judgments"] = [
                {"field": fj.field, "judgment": fj.judgment, "reason": fj.reason}
                for fj in result.field_judgments
            ]
            record["overall_reason"] = result.overall_reason
            if result.corrected_triplet:
                record["corrected_triplet"] = triplet_to_dict(result.corrected_triplet)
            record["success"] = True
            record["elapsed_s"] = round(time.time() - t0, 2)
            logger.info(
                "%s review result=%s (%.1fs)", clause_id, result.verdict, record["elapsed_s"]
            )
        except Exception as exc:
            record["error"] = str(exc)
            record["elapsed_s"] = round(time.time() - t0, 2)
            logger.error("%s review failed: %s", clause_id, exc)
        return record

    _run_task_pool(
        pending, review_one, output_path, workers,
        f"Review-{reviewer_role}",
        show_progress=not args.no_progress,
    )

    print(f"\nReview results saved to: {output_path} (workers={workers})")


# ======================================================================
# CLI builder
# ======================================================================


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="LexSpec staged annotation pipeline: annotate / review / merge",
    )
    parser.add_argument("--config", default="configs/model.yaml", help="Model config file path")
    parser.add_argument("--prompts", default="configs/prompts.yaml", help="Prompt config file path")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- annotate subcommand ----
    p_ann = sub.add_parser("annotate", help="Annotate the test set with a single model")
    p_ann.add_argument(
        "--model", required=True,
        choices=["gemma", "qwen", "primary", "secondary"],
        help="Annotation model identifier",
    )
    p_ann.add_argument("--input", default=DEFAULT_TESTSET, help="Test set JSONL path")
    p_ann.add_argument("--output", default=None, help="Output path (auto-determined by model)")
    p_ann.add_argument(
        "--resume", action="store_true",
        help="Checkpoint/resume: skip already-annotated clause_ids",
    )
    p_ann.add_argument(
        "--workers", type=int, default=8,
        help="Concurrent API request count (default 8)",
    )
    p_ann.add_argument(
        "--no-progress", action="store_true",
        help="Disable tqdm progress bar",
    )

    # ---- review subcommand ----
    p_rev = sub.add_parser("review", help="Review one model's annotations with another model")
    p_rev.add_argument(
        "--reviewer", required=True,
        choices=["gemma", "qwen", "primary", "secondary"],
        help="Model performing the review",
    )
    p_rev.add_argument(
        "--source", required=True,
        choices=["gemma", "qwen", "primary", "secondary"],
        help="Annotation source being reviewed",
    )
    p_rev.add_argument("--source-file", default=None, help="Override source annotation file path")
    p_rev.add_argument("--output", default=None, help="Output path")
    p_rev.add_argument("--resume", action="store_true", help="Checkpoint/resume")
    p_rev.add_argument(
        "--workers", type=int, default=8,
        help="Concurrent API request count (default 8)",
    )
    p_rev.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar")

    # ---- merge subcommand ----
    p_merge = sub.add_parser("merge", help="Merge annotations and reviews into gold standard")
    p_merge.add_argument("--qwen", default=QWEN_ANNOT, help="Qwen annotation file path")
    p_merge.add_argument("--gemma", default=GEMMA_ANNOT, help="Gemma annotation file path")
    p_merge.add_argument("--qwen-review", default=QWEN_REVIEW_GEMMA, help="Qwen review of Gemma")
    p_merge.add_argument("--gemma-review", default=GEMMA_REVIEW_QWEN, help="Gemma review of Qwen")
    p_merge.add_argument("--output", default=GOLD_OUT, help="Gold output path")
    p_merge.add_argument("--disagreements", default=DISAGREE_OUT, help="Human-review items output path")

    return parser
