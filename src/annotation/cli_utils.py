"""
CLI utility functions for the annotation pipeline.

Provides configuration loading, client building, model name resolution,
output path selection, and annotation deduplication/resume support.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

from src.extraction.client import LLMClient, ClientConfig
from src.extraction.schema import LegalTriplet
from src.utils.config import 构建标注客户端
from src.utils.io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default file paths
# ---------------------------------------------------------------------------
DEFAULT_TESTSET = "data/processed/lexspec_100.jsonl"
ANNOT_DIR = "data/annotations"
GEMMA_ANNOT = f"{ANNOT_DIR}/gemma_annotations.jsonl"
QWEN_ANNOT = f"{ANNOT_DIR}/qwen_annotations.jsonl"
QWEN_REVIEW_GEMMA = f"{ANNOT_DIR}/qwen_review_gemma.jsonl"
GEMMA_REVIEW_QWEN = f"{ANNOT_DIR}/gemma_review_qwen.jsonl"
GOLD_OUT = "data/processed/gold_triplets.jsonl"
DISAGREE_OUT = f"{ANNOT_DIR}/needs_human_review.jsonl"

# Model alias mapping -- supports both primary/secondary and specific model names.
MODEL_ALIASES = {
    "gemma": "secondary",
    "qwen": "primary",
    "primary": "primary",
    "secondary": "secondary",
}


# ======================================================================
# Utility functions
# ======================================================================


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def load_config(config_path: str) -> dict:
    """Load a YAML config file; raises FileNotFoundError if missing."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_client(config: dict, model_key: str) -> LLMClient:
    """Build an LLM client for the annotation model identified by model_key.

    Delegates to src.utils.config.构建标注客户端 for unified config loading.
    """
    role = MODEL_ALIASES.get(model_key, model_key)
    if role not in ("primary", "secondary"):
        raise ValueError(f"Unknown model identifier: {model_key}")
    return 构建标注客户端(config=config, model_role=role)


def model_role_name(config: dict, model_key: str) -> str:
    """Resolve the display name of a model from config."""
    role = MODEL_ALIASES.get(model_key, model_key)
    ann = config.get("models", {}).get("annotation", {})
    if role == "primary":
        return ann.get("primary", {}).get("name", "qwen")
    return ann.get("secondary", {}).get("name", "gemma")


def output_path_for_model(model_key: str) -> str:
    """Return the default annotation output path for a model identifier."""
    role = MODEL_ALIASES.get(model_key, model_key)
    if role in ("primary", "qwen"):
        return QWEN_ANNOT
    return GEMMA_ANNOT


def review_output_path(reviewer: str, source: str) -> str:
    """Return the default review output path for a reviewer/source pair."""
    rev = MODEL_ALIASES.get(reviewer, reviewer)
    src = MODEL_ALIASES.get(source, source)
    if rev in ("primary", "qwen") and src in ("secondary", "gemma"):
        return QWEN_REVIEW_GEMMA
    if rev in ("secondary", "gemma") and src in ("primary", "qwen"):
        return GEMMA_REVIEW_QWEN
    raise ValueError(f"Unsupported review combination: {reviewer} reviews {source}")


def triplet_to_dict(t: LegalTriplet) -> dict:
    """LegalTriplet -> serializable dict."""
    return t.model_dump(mode="json")


def dict_to_triplet(d: dict) -> LegalTriplet:
    """Dict -> LegalTriplet."""
    return LegalTriplet.model_validate(d)


# ======================================================================
# Annotation deduplication -- handles duplicates from resume
# ======================================================================


def _deduplicate_annotations(output_path: str) -> int:
    """Deduplicate an annotation file, keeping the best record per clause_id.

    Selection strategy: successful records preferred over failures; among
    same status, the last one is kept. File is overwritten in-place.

    Args:
        output_path: Annotation JSONL file path.

    Returns:
        Number of duplicate records removed.
    """
    path = Path(output_path)
    if not path.exists():
        return 0

    records = read_jsonl(output_path)
    if not records:
        return 0

    best: Dict[str, dict] = {}
    for rec in records:
        cid = rec.get("clause_id", "")
        if not cid:
            continue
        if cid not in best:
            best[cid] = rec
        else:
            existing = best[cid]
            # Success preferred; same status keeps later one.
            if rec.get("success") and not existing.get("success"):
                best[cid] = rec
            elif not rec.get("success") and existing.get("success"):
                pass  # Keep existing success record.
            else:
                best[cid] = rec

    removed = len(records) - len(best)
    if removed > 0:
        deduped = [best[cid] for cid in sorted(best)]
        write_jsonl(output_path, deduped)
        logger.info(
            "Annotation dedup complete %s: %d -> %d (removed %d duplicates)",
            output_path, len(records), len(deduped), removed,
        )
    return removed


# ======================================================================
# Completed annotation tracking -- for --resume mode
# ======================================================================


def _load_completed_ids(output_path: str) -> Set[str]:
    """Read clause_ids of successfully completed annotations from a file.

    Only counts records with success=True and a triplet containing
    substantive content.
    """
    path = Path(output_path)
    if not path.exists():
        return set()
    done: Set[str] = set()
    for rec in read_jsonl(path):
        cid = rec.get("clause_id")
        if not cid or rec.get("success") is not True:
            continue
        triplet = rec.get("triplet") or {}
        subj = (triplet.get("subject") or {}).get("text", "")
        pred = (triplet.get("action") or {}).get("predicate", "")
        obj = (triplet.get("action") or {}).get("object", "")
        if subj.strip() or pred.strip() or obj.strip():
            done.add(cid)
    return done
