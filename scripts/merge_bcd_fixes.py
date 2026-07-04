#!/usr/bin/env python3
"""Merge fixed_batch_bcd_*.jsonl corrections into gold_triplets_500.jsonl."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD_IN = ROOT / "data/processed/gold_triplets_500.jsonl"
GOLD_OUT = ROOT / "data/processed/gold_triplets_500.jsonl"
TEST_OUT = ROOT / "data/processed/gold_testset_500.jsonl"
FIX_DIR = ROOT / "data/processed/curated_500"
MANIFEST = FIX_DIR / "manifest.json"

MULTIWORD_PRED = re.compile(r"\s")


def validate_record(record: dict) -> list[str]:
    issues = []
    t = record["triplet"]
    s, a, c = t["subject"], t["action"], t["condition"]
    pred = a.get("predicate", "")
    if MULTIWORD_PRED.search(pred):
        issues.append("multiword_pred")
    ctext, ctype = (c.get("text") or "").strip(), c.get("type", "none")
    if ctext and ctype == "none":
        issues.append("cond_mismatch")
    if not ctext and ctype != "none":
        issues.append("cond_empty")
    if len(a.get("object", "")) > 185:
        issues.append("long_obj")
    return issues


def load_fixes() -> dict[str, dict]:
    fixes: dict[str, dict] = {}
    for path in sorted(FIX_DIR.glob("fixed_batch_bcd_*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            fixes[row["clause_id"]] = row
    return fixes


def _merge_phenomena(rec: dict, fix: dict) -> dict:
    phen = dict(fix.get("phenomena") or rec.get("phenomena") or {})
    triplet = fix["triplet"]
    if (triplet.get("condition") or {}).get("text", "").strip():
        phen["conditional"] = True
    return phen


def main() -> None:
    fixes = load_fixes()
    if not fixes:
        raise SystemExit("No fixed_batch_bcd_*.jsonl files found.")

    records = [json.loads(line) for line in GOLD_IN.read_text().splitlines() if line.strip()]
    applied = 0
    validation_failures: list[tuple[str, list[str]]] = []

    for rec in records:
        cid = rec["clause_id"]
        if cid not in fixes:
            continue
        fix = fixes[cid]
        old_triplet = json.dumps(rec["triplet"], sort_keys=True)
        rec["triplet"] = fix["triplet"]
        rec["phenomena"] = _merge_phenomena(rec, fix)
        rec["curation_changed"] = (
            json.dumps(rec["triplet"], sort_keys=True) != old_triplet
        )
        rec["needs_human_review"] = False
        rec["curated"] = True
        rec["bcd_fix_notes"] = fix.get("fix_notes", "")
        applied += 1
        issues = validate_record(rec)
        if issues:
            validation_failures.append((cid, issues))

    records.sort(key=lambda x: x["clause_id"])

    with GOLD_OUT.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with TEST_OUT.open("w") as f:
        for rec in records:
            f.write(json.dumps({
                "clause_id": rec["clause_id"],
                "text": rec["text"],
                "phenomena": rec.get("phenomena", {}),
            }, ensure_ascii=False) + "\n")

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest["bcd_fixes_applied"] = applied
    manifest["bcd_fix_validation_issues"] = len(validation_failures)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({
        "fixes_loaded": len(fixes),
        "fixes_applied": applied,
        "validation_failures": len(validation_failures),
        "sample_failures": validation_failures[:10],
    }, indent=2))


if __name__ == "__main__":
    main()
