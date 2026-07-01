"""
CUAD v1 data loading utilities.

Supports three source formats:
  - Full contract contexts from CUAD_v1.json (sentence mode)
  - Expert-annotated clause spans from master_clauses.csv (spans mode)
  - QA answer spans from CUAD_v1.json (qa_spans mode)
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import List

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Columns to skip when loading CUAD spans.
_SKIP_COLUMNS = frozenset({
    "Filename", "Document Name", "Document Name-Answer",
    "Parties", "Parties-Answer",
})


def load_cuad_data(cuad_path: str) -> List[str]:
    """Load CUAD v1 JSON and extract all paragraph contexts.

    CUAD v1 JSON structure::

        {
          "version": "aok_v1.0",
          "data": [
            {"title": "...", "paragraphs": [
              {"context": "<full contract text>", "qas": [...]},
              ...
            ]},
            ...
          ]
        }

    Extracts the ``context`` field of each paragraph. Each context may
    contain an entire contract; sentence segmentation is done downstream
    via Stanza's sentence splitter.

    Args:
        cuad_path: Path to CUAD_v1.json file.

    Returns:
        List of paragraph context strings (one per paragraph).
    """
    logger.info("Loading CUAD data: %s", cuad_path)
    with open(cuad_path, "r", encoding="utf-8") as fh:
        cuad = json.load(fh)

    contexts: List[str] = []
    for doc in cuad.get("data", []):
        for para in doc.get("paragraphs", []):
            ctx = para.get("context", "")
            if ctx and ctx.strip():
                contexts.append(ctx.strip())

    logger.info(
        "Loaded %d paragraph contexts from %d CUAD documents",
        len(contexts), len(cuad.get("data", [])),
    )
    return contexts


def load_cuad_spans(master_clauses_path: str) -> List[str]:
    """Load expert-annotated clause fragments from CUAD master_clauses.csv.

    Each non-answer column may contain clause text for one of 41 categories.
    Returns a deduplicated list of clause texts (typically ~5k unique fragments).
    """
    path = Path(master_clauses_path)
    if not path.exists():
        raise FileNotFoundError(f"master_clauses.csv not found: {master_clauses_path}")

    seen: set = set()
    clauses: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for col, val in row.items():
                if col in _SKIP_COLUMNS or col.endswith("-Answer"):
                    continue
                text = (val or "").strip()
                if len(text) < 20:
                    continue
                norm = re.sub(r"\s+", " ", text)
                if norm in seen:
                    continue
                seen.add(norm)
                clauses.append(text)
    logger.info("Loaded %d unique clause fragments from %s", len(clauses), path)
    return clauses


def load_cuad_qa_spans(cuad_path: str) -> List[str]:
    """Load answer spans from CUAD SQuAD-format JSON (~13k annotated fragments)."""
    with open(cuad_path, "r", encoding="utf-8") as fh:
        cuad = json.load(fh)

    seen: set = set()
    clauses: List[str] = []
    for doc in cuad.get("data", []):
        for para in doc.get("paragraphs", []):
            for qa in para.get("qas", []):
                if qa.get("is_impossible"):
                    continue
                for ans in qa.get("answers", []):
                    text = (ans.get("text") or "").strip()
                    if len(text) < 20:
                        continue
                    norm = re.sub(r"\s+", " ", text)
                    if norm in seen:
                        continue
                    seen.add(norm)
                    clauses.append(text)
    logger.info("Loaded %d unique QA answer spans from %s", len(clauses), cuad_path)
    return clauses
