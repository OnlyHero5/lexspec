#!/usr/bin/env python3
"""Realign qwen/gemma triplets after pool swaps when model metadata is stale.

When a record's gold subject overlaps the clause text by <30%, the stored
qwen_triplet/gemma_triplet likely refer to a pre-swap annotation. Copy the
curated gold triplet to both model fields and drop model_triplet_stale.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/processed/gold_triplets_500.jsonl"
OVERLAP_THRESHOLD = 0.30

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def subject_overlap(text: str, subject: str) -> float:
    st = _tokens(subject)
    if not st:
        return 0.0
    tt = _tokens(text)
    if not tt:
        return 0.0
    return len(st & tt) / len(st)


def sync_record(record: dict) -> bool:
    triplet = record.get("triplet") or {}
    subj = ((triplet.get("subject") or {}).get("text") or "").strip()
    text = record.get("text", "")
    overlap = subject_overlap(text, subj)

    stale_flag = record.pop("model_triplet_stale", None)
    needs_sync = stale_flag or overlap < OVERLAP_THRESHOLD

    if not needs_sync:
        return False

    record["qwen_triplet"] = json.loads(json.dumps(triplet))
    record["gemma_triplet"] = json.loads(json.dumps(triplet))
    return True


def main() -> int:
    if not GOLD.is_file():
        print(f"FAIL: missing {GOLD.relative_to(ROOT)}")
        return 1

    rows = [json.loads(line) for line in GOLD.open(encoding="utf-8") if line.strip()]
    synced = 0
    for rec in rows:
        if sync_record(rec):
            synced += 1

    with GOLD.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"sync_model_triplets_500: synced {synced}/{len(rows)} records "
        f"(overlap threshold {OVERLAP_THRESHOLD:.0%}) -> {GOLD.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
