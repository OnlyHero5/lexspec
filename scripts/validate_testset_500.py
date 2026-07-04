#!/usr/bin/env python3
"""Validate the LexSpec 500-item gold test set after remediation."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/processed/gold_triplets_500.jsonl"
TEST = ROOT / "data/processed/gold_testset_500.jsonl"

# Reuse QA heuristics from build script
sys.path.insert(0, str(ROOT / "scripts"))
from build_gold_500 import is_bad_text  # noqa: E402

PHENOMENA_KEYS = (
    "passive",
    "conditional",
    "relative_clause",
    "long_distance",
    "negation",
    "is_definition",
)

MIN_LONG_DISTANCE = 75
MAX_ZERO_PHEN = 10
EXPECTED_LEN = 500


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def count_zero_phenomena(rows: list[dict]) -> int:
    n = 0
    for r in rows:
        phen = r.get("phenomena") or {}
        if not any(phen.get(k) for k in PHENOMENA_KEYS):
            n += 1
    return n


def phenomenon_counts(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {k: 0 for k in PHENOMENA_KEYS}
    for r in rows:
        phen = r.get("phenomena") or {}
        for k in PHENOMENA_KEYS:
            if phen.get(k):
                out[k] += 1
    return out


def validate(rows: list[dict]) -> tuple[bool, list[str], dict]:
    issues: list[str] = []
    metrics: dict = {}

    n = len(rows)
    metrics["count"] = n
    if n != EXPECTED_LEN:
        issues.append(f"count={n}, expected {EXPECTED_LEN}")

    ids = [r.get("clause_id") for r in rows]
    texts = [r.get("text", "") for r in rows]
    metrics["unique_clause_ids"] = len(set(ids))
    metrics["unique_texts"] = len(set(texts))

    if len(set(ids)) != n:
        dup_ids = [k for k, v in Counter(ids).items() if v > 1]
        issues.append(f"duplicate clause_ids: {dup_ids[:10]}")
    if len(set(texts)) != n:
        dup_t = [t[:60] for t, c in Counter(texts).items() if c > 1]
        issues.append(f"duplicate texts: {len(dup_t)} groups")

    ld = sum(1 for r in rows if (r.get("phenomena") or {}).get("long_distance"))
    zp = count_zero_phenomena(rows)
    metrics["long_distance"] = ld
    metrics["zero_phenomena"] = zp
    metrics["phenomena"] = phenomenon_counts(rows)

    if ld < MIN_LONG_DISTANCE:
        issues.append(f"long_distance={ld}, need >={MIN_LONG_DISTANCE}")
    if zp > MAX_ZERO_PHEN:
        issues.append(f"zero_phenomena={zp}, max {MAX_ZERO_PHEN}")

    bad: list[tuple[str, str]] = []
    for r in rows:
        reason = is_bad_text(r.get("text", ""))
        if reason:
            bad.append((r.get("clause_id", "?"), reason))
    metrics["bad_text_count"] = len(bad)
    if bad:
        issues.append(f"bad_text: {bad[:8]}")

    passed = not issues
    metrics["passed"] = passed
    return passed, issues, metrics


def main() -> int:
    rows = load_jsonl(GOLD)
    source = GOLD
    if not rows:
        rows = load_jsonl(TEST)
        source = TEST

    if not rows:
        print("FAIL: no gold_triplets_500.jsonl or gold_testset_500.jsonl")
        return 1

    passed, issues, metrics = validate(rows)

    print(f"Source: {source.relative_to(ROOT)}")
    print(f"count={metrics['count']} unique_ids={metrics['unique_clause_ids']} "
          f"unique_texts={metrics['unique_texts']}")
    print(f"long_distance={metrics['long_distance']} (min {MIN_LONG_DISTANCE})")
    print(f"zero_phenomena={metrics['zero_phenomena']} (max {MAX_ZERO_PHEN})")
    print(f"bad_text={metrics['bad_text_count']}")
    phen = metrics["phenomena"]
    print(
        "phenomena: "
        + ", ".join(f"{k}={phen[k]}" for k in PHENOMENA_KEYS)
    )

    if issues:
        print("FAIL")
        for item in issues:
            print(f"  - {item}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
