"""
Clause extraction and phenomenon annotation.

Splits contract contexts into individual sentences (clauses) using Stanza,
then parses each clause and annotates it with detected language phenomena.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from src.linguistic.stanza_parser import StanzaParser
from src.corpus.phenomena_detector import detect_phenomena, is_boilerplate_clause
from src.utils.logging import get_logger

logger = get_logger(__name__)


def split_into_clauses(
    parser: StanzaParser,
    contexts: List[str],
    max_contexts: Optional[int] = None,
) -> List[str]:
    """Split paragraph contexts into individual sentences (clauses).

    Uses Stanza's sentence splitter to segment each context into sentences,
    returning a flat list of sentences. Each sentence is a candidate clause.

    For very large CUAD contexts, contexts can be randomly sampled first
    to control runtime. Sampling uses a fixed seed (42) for reproducibility.

    Filters:
      - Retains sentences with 5-200 tokens
      - Excludes template text (headers, signatures, etc.)

    Args:
        parser:        Configured StanzaParser instance.
        contexts:      List of paragraph context strings.
        max_contexts:  Max number of contexts to process (0 or None = all).

    Returns:
        Flat list of extracted sentence strings.
    """
    if max_contexts and max_contexts > 0 and len(contexts) > max_contexts:
        rng = random.Random(42)
        contexts = rng.sample(contexts, max_contexts)
        logger.info("Sampled %d contexts for sentence splitting", max_contexts)

    clauses: List[str] = []
    import stanza as _stanza  # noqa: F811

    for i, ctx in enumerate(
        tqdm(contexts, desc="Splitting contexts into sentences", unit="ctx")
    ):
        try:
            doc = parser.nlp(ctx)
            for sent in doc.sentences:
                sent_text = sent.text.strip()
                token_count = len(sent.words)
                if 5 <= token_count <= 200:
                    if not is_boilerplate_clause(sent_text):
                        clauses.append(sent_text)
        except Exception as exc:
            logger.debug(
                "Skipping context %d/%d (len=%d): %s",
                i + 1, len(contexts), len(ctx), exc,
            )
            continue

    logger.info(
        "Extracted %d clauses from %d contexts", len(clauses), len(contexts)
    )
    return clauses


def build_clause_records(
    parser: StanzaParser,
    clauses: List[str],
    source_label: str,
) -> Tuple[List[Dict], Dict[str, List[int]]]:
    """Parse a list of clauses and attach phenomenon annotations.

    For each clause, performs dependency parsing and detects language
    phenomena, building the candidate pool. If parsing fails (extremely
    rare), silently skips.

    Args:
        parser:       StanzaParser instance.
        clauses:      List of clause text strings.
        source_label: Data source label (e.g. "cuad_v1_sentences").

    Returns:
        (clause_records, phenomenon_pools) tuple:
          - clause_records: list of dicts with text/phenomena/source
          - phenomenon_pools: phenomenon name -> list of matching record indices
    """
    clause_records: List[Dict] = []
    phenomenon_pools: Dict[str, List[int]] = defaultdict(list)

    logger.info("Parsing %d candidate clauses for phenomenon detection...", len(clauses))

    for idx, clause_text in enumerate(
        tqdm(clauses, desc="Detecting language phenomena", unit="clause")
    ):
        if is_boilerplate_clause(clause_text):
            continue
        try:
            tree = parser.parse(clause_text)
            if tree.token_count < 3:
                continue
            phen = detect_phenomena(tree)
        except Exception as exc:
            logger.debug(
                "Clause %d parse failed (len=%d): %s", idx, len(clause_text), exc,
            )
            continue

        record = {
            "clause_id": "",
            "text": clause_text,
            "phenomena": phen,
            "source": source_label,
        }
        clause_records.append(record)
        record_idx = len(clause_records) - 1

        # Add clause index to phenomenon pools (is_definition is excluded
        # from stratified sampling).
        if phen["passive"]:
            phenomenon_pools["passive"].append(record_idx)
        if phen["conditional"]:
            phenomenon_pools["conditional"].append(record_idx)
        if phen["relative_clause"]:
            phenomenon_pools["relative_clause"].append(record_idx)
        if phen["long_distance"]:
            phenomenon_pools["long_distance"].append(record_idx)
        if phen["negation"]:
            phenomenon_pools["negation"].append(record_idx)

    logger.info(
        "Parsing complete: %d valid clauses. Phenomenon pool sizes: %s",
        len(clause_records),
        {k: len(v) for k, v in phenomenon_pools.items()},
    )
    return clause_records, phenomenon_pools
