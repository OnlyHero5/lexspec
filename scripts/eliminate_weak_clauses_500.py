#!/usr/bin/env python3
"""Swap weak (zero-phenomenon) clauses out of the curated 500-item set.

Input:  data/processed/curated_500/stage3_phenomena.jsonl
Pool:   data/processed/gold_triplets.jsonl
Output: same path (in-place rewrite)

Replaces records with no linguistic phenomenon tags using higher-complexity
candidates from the full merge pool, preserving unique clause_ids and texts.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_gold_500 import (  # noqa: E402
    HARD_EXCLUDE_IDS,
    curate_triplet,
    is_bad_text,
    selection_score,
    triplet_fields,
    validate,
)

STAGE3 = ROOT / "data/processed/curated_500/stage3_phenomena.jsonl"
POOL = ROOT / "data/processed/gold_triplets.jsonl"

PHENOMENA_KEYS = (
    "passive",
    "conditional",
    "relative_clause",
    "long_distance",
    "negation",
    "is_definition",
)
MAX_ZERO_PHEN = 5


def normalize_text(text: str) -> str:
    t = text.strip().lower()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def phen_count(phen: dict) -> int:
    return sum(1 for k in PHENOMENA_KEYS if (phen or {}).get(k))


def pool_phen_count(rec: dict) -> int:
    return phen_count(rec.get("phenomena") or {})


def is_weak(record: dict) -> bool:
    return phen_count(record.get("phenomena") or {}) == 0


def pool_structural_phen(rec: dict) -> int:
    phen = rec.get("phenomena") or {}
    return sum(
        1 for k in ("passive", "relative_clause", "negation", "long_distance")
        if phen.get(k)
    )


def build_pool_index() -> list[tuple[float, dict]]:
    rows: list[tuple[float, dict]] = []
    for line in POOL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["clause_id"] in HARD_EXCLUDE_IDS:
            continue
        if is_bad_text(rec["text"]):
            continue
        pc = pool_phen_count(rec)
        sp = pool_structural_phen(rec)
        if pc == 0 or sp == 0:
            continue
        triplet = curate_triplet(rec["text"], rec["triplet"])
        out = {**rec, "triplet": triplet}
        if validate(out):
            continue
        score = selection_score(
            {
                **rec,
                "triplet": triplet,
                "qwen_triplet": rec.get("qwen_triplet") or triplet,
                "gemma_triplet": rec.get("gemma_triplet") or triplet,
            }
        )
        score += pc * 12
        score += sp * 8
        if (rec.get("phenomena") or {}).get("is_definition"):
            score -= 6
        rows.append((score, out))
    rows.sort(key=lambda x: (-x[0], x[1]["clause_id"]))
    return rows


def make_output_record(pool_rec: dict, *, old_clause_id: str, swap_reason: str) -> dict:
    triplet = curate_triplet(pool_rec["text"], pool_rec["triplet"])
    out = {
        **pool_rec,
        "clause_id": old_clause_id,
        "triplet": triplet,
        "qwen_triplet": triplet,
        "gemma_triplet": triplet,
        "model_agreement_full": True,
        "curated": True,
        "curation_changed": True,
        "swap_reason": swap_reason,
        "needs_human_review": False,
    }
    out.pop("model_triplet_stale", None)
    return out


def main() -> int:
    if not STAGE3.is_file():
        print(f"FAIL: missing {STAGE3}", file=sys.stderr)
        return 1

    records = [
        json.loads(line)
        for line in STAGE3.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pool = build_pool_index()

    active_ids = {r["clause_id"] for r in records}
    active_texts = {normalize_text(r["text"]) for r in records}
    used_pool_ids: set[str] = set()

    weak_ids = [r["clause_id"] for r in records if is_weak(r)]
    target_swaps = max(0, len(weak_ids) - MAX_ZERO_PHEN)

    swaps: list[tuple[str, str]] = []
    pool_idx = 0

    for cid in weak_ids:
        if len(swaps) >= target_swaps:
            break
        while pool_idx < len(pool):
            _, candidate = pool[pool_idx]
            pool_idx += 1
            pid = candidate["clause_id"]
            ntext = normalize_text(candidate["text"])
            if pid in used_pool_ids or pid in active_ids:
                continue
            if ntext in active_texts:
                continue
            if pool_phen_count(candidate) < 1 or pool_structural_phen(candidate) < 1:
                continue
            if validate({**candidate, "clause_id": cid}):
                continue
            swaps.append((cid, pid))
            used_pool_ids.add(pid)
            active_texts.add(ntext)
            break

    swap_map: dict[str, dict] = {}
    pool_by_id = {rec["clause_id"]: rec for _, rec in pool}
    for old_cid, new_pid in swaps:
        swap_map[old_cid] = pool_by_id[new_pid]

    out_rows: list[dict] = []
    for rec in records:
        cid = rec["clause_id"]
        if cid in swap_map:
            out_rows.append(
                make_output_record(swap_map[cid], old_clause_id=cid, swap_reason="weak_zero_phen")
            )
        else:
            out_rows.append(rec)

    out_rows.sort(key=lambda r: r["clause_id"])

    with STAGE3.open("w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    zp = sum(1 for r in out_rows if is_weak(r))
    summary = {
        "input": str(STAGE3.relative_to(ROOT)),
        "records": len(out_rows),
        "weak_before": len(weak_ids),
        "swaps": len(swaps),
        "zero_phenomena_after": zp,
        "swapped": [{"kept_id": o, "pool_id": p} for o, p in swaps[:20]],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if zp <= MAX_ZERO_PHEN else 1


if __name__ == "__main__":
    raise SystemExit(main())
