"""
Clause selection strategies for building the LexSpec evaluation corpus.

Supports two modes:
  - ``select_all_clauses``:  Return all valid parsed clauses (full mode).
  - ``select_balanced_testset``: Stratified sampling to meet per-phenomenon
    quotas, prioritizing rare phenomena.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List

from src.linguistic.stanza_parser import StanzaParser
from src.corpus.clause_processor import build_clause_records
from src.utils.logging import get_logger

logger = get_logger(__name__)


def select_all_clauses(clause_records: List[Dict]) -> List[Dict]:
    """Full mode: return all valid parsed clauses (excluding definition clauses).

    Definition clauses ("X means Y") lack operative legal actions and cannot
    yield meaningful triplets, so they are excluded from the test set.
    """
    result: List[Dict] = []
    for i, record in enumerate(clause_records):
        if record["phenomena"].get("is_definition", False):
            continue
        out = dict(record)
        out["clause_id"] = f"C-{i + 1:05d}"
        result.append(out)
    return result


def select_balanced_testset(
    parser: StanzaParser,
    clauses: List[str],
    target_count: int = 100,
    min_passive: int = 20,
    min_conditional: int = 20,
    min_relative: int = 15,
    min_long_distance: int = 20,
    min_negation: int = 15,
    source_label: str = "cuad_v1",
) -> List[Dict]:
    """Select a balanced test set from candidate clauses via stratified sampling.

    Algorithm:
      1. Parse each clause, detect its language phenomena.
      2. Group clauses into candidate pools by phenomenon.
      3. Fill quotas in order of phenomenon rarity (rarest first),
         prioritizing clauses that cover multiple underfilled quotas.
      4. If a quota cannot be met exactly (extremely rare), fill the
         remaining slots up to target_count.

    Definition clauses are excluded before sampling.

    Each selected clause is stored as a dict with keys:
      - ``clause_id``:    unique zero-padded ID (e.g. "C-0001")
      - ``text``:         original clause text
      - ``phenomena``:    dict of boolean phenomenon flags
      - ``source``:       data source label

    Args:
        parser:            StanzaParser for clause analysis.
        clauses:           Candidate clause strings.
        target_count:      Desired total number of clauses.
        min_passive:       Minimum passive voice clauses.
        min_conditional:   Minimum conditional clause clauses.
        min_relative:      Minimum relative clause clauses.
        min_long_distance: Minimum long-distance dependency clauses.
        min_negation:      Minimum negation clauses.
        source_label:      Data source label.

    Returns:
        List of selected clause dicts, one per clause.
    """
    rng = random.Random(42)

    # Shuffle to avoid document-order bias.
    rng.shuffle(clauses)

    clause_records, phenomenon_pools = build_clause_records(
        parser, clauses, source_label,
    )

    # --- Exclude definition clauses ---
    def_indices = {
        i for i, r in enumerate(clause_records)
        if r["phenomena"].get("is_definition", False)
    }
    for pool_name in phenomenon_pools:
        phenomenon_pools[pool_name] = [
            i for i in phenomenon_pools[pool_name] if i not in def_indices
        ]
    if def_indices:
        logger.info("Excluded %d definition clauses from candidate pool", len(def_indices))

    # Fill quotas in ascending pool size order (rare phenomena first).
    selected: set = set()
    quotas = {
        "passive": min_passive,
        "conditional": min_conditional,
        "relative_clause": min_relative,
        "long_distance": min_long_distance,
        "negation": min_negation,
    }
    fill_order = sorted(quotas.keys(), key=lambda k: len(phenomenon_pools.get(k, [])))

    for phen_name in fill_order:
        quota = quotas[phen_name]
        pool = phenomenon_pools.get(phen_name, [])
        rng.shuffle(pool)

        for idx in pool:
            # Check if quota for this phenomenon is already met.
            current_count = sum(
                1 for i in selected
                if clause_records[i]["phenomena"].get(phen_name, False)
            )
            if current_count >= quota:
                break
            if idx not in selected:
                selected.add(idx)

    # --- Fill remaining slots to target_count ---
    # Prefer clauses with at least one phenomenon and non-definition.
    remaining_indices = [
        i for i in range(len(clause_records))
        if i not in selected and i not in def_indices
    ]
    remaining_indices.sort(
        key=lambda i: -sum(
            1 for k, v in clause_records[i]["phenomena"].items()
            if k != "is_definition" and v
        )
    )

    while len(selected) < target_count and remaining_indices:
        idx = remaining_indices.pop(0)
        selected.add(idx)

    # --- Build final output ---
    result: List[Dict] = []
    for i, idx in enumerate(sorted(selected)):
        record = dict(clause_records[idx])
        record["clause_id"] = f"C-{i + 1:04d}"
        result.append(record)

    # Statistics log.
    phen_counts = {
        phen: sum(1 for r in result if r["phenomena"][phen])
        for phen in quotas
    }
    logger.info("Selected %d clauses: %s", len(result), phen_counts)

    return result
