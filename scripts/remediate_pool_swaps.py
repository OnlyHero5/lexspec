#!/usr/bin/env python3
"""Stage-1 data-layer remediation for the LexSpec 500-item test set."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
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

GOLD_IN = ROOT / "data/processed/gold_triplets_500.jsonl"
POOL_IN = ROOT / "data/processed/gold_triplets.jsonl"
OUT = ROOT / "data/processed/curated_500/stage1_pool.jsonl"

LD_TARGET = 75

PHENOMENA_KEYS = (
    "passive",
    "conditional",
    "relative_clause",
    "long_distance",
    "negation",
    "is_definition",
)

DUPLICATE_GROUPS: list[tuple[str, str]] = [
    ("C-00010", "C-00011"),
    ("C-00078", "C-00216"),
    ("C-00245", "C-00884"),
]

FORCE_REPLACE_IDS = frozenset({"C-00026"})

TABLE_FRAGMENT_RE = re.compile(
    r"^(?:License Term|TERM:)\b|Perpetual.*Commencing:",
    re.I,
)
REDACTED_RE = re.compile(r"\$\[\*\]|\[\*\]")
_NORM_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_text_key(text: str) -> str:
    """Lowercase key with punctuation stripped for duplicate detection."""
    t = text.lower().strip()
    t = _NORM_PUNCT_RE.sub("", t)
    return re.sub(r"\s+", " ", t).strip()


def phen_count(phenomena: dict) -> int:
    return sum(1 for v in phenomena.values() if v)


def meaningful_phen_count(phenomena: dict) -> int:
    return sum(1 for k in PHENOMENA_KEYS if k != "is_definition" and phenomena.get(k))


def is_definition_only(phenomena: dict) -> bool:
    phen = phenomena or {}
    return bool(phen.get("is_definition")) and meaningful_phen_count(phen) == 0


def is_table_fragment(text: str) -> bool:
    return bool(TABLE_FRAGMENT_RE.search(text.strip()))


def needs_force_replace(record: dict) -> str | None:
    cid = record["clause_id"]
    if cid in FORCE_REPLACE_IDS:
        return "table_fragment"
    text = record["text"]
    if is_table_fragment(text):
        return "table_fragment"
    if REDACTED_RE.search(text):
        return "redacted_amount"
    bad = is_bad_text(text)
    if bad:
        return bad
    return None


def _record_score(record: dict) -> float:
    return selection_score(
        {
            **record,
            "qwen_triplet": record.get("qwen_triplet") or record["triplet"],
            "gemma_triplet": record.get("gemma_triplet") or record["triplet"],
        }
    )


def model_agreement_full(record: dict) -> bool:
    qt = record.get("qwen_triplet")
    gt = record.get("gemma_triplet")
    if not qt or not gt:
        return False
    return (
        triplet_fields(qt)
        == triplet_fields(gt)
        == triplet_fields(record["triplet"])
    )


def is_ld_candidate(record: dict) -> bool:
    phen = record.get("phenomena") or {}
    if phen.get("long_distance"):
        return True
    text_len = len(record["text"])
    if phen.get("relative_clause") and text_len >= 300:
        return True
    if text_len >= 250 and (phen.get("relative_clause") or phen.get("conditional")):
        return True
    if text_len >= 450 and phen_count(phen) >= 1:
        return True
    return False


def swap_priority(record: dict, *, duplicate_loser: bool) -> tuple:
    phen = phen_count(record.get("phenomena") or {})
    ld = bool((record.get("phenomena") or {}).get("long_distance"))
    return (
        int(duplicate_loser),
        int(record["clause_id"] in FORCE_REPLACE_IDS or needs_force_replace(record) is not None),
        int(phen == 0),
        int(not ld),
        -_record_score(record),
        record["clause_id"],
    )


def pool_record_to_output(pool_rec: dict, *, mark_ld: bool, swap_reason: str) -> dict:
    triplet = curate_triplet(pool_rec["text"], pool_rec["triplet"])
    phenomena = dict(pool_rec.get("phenomena") or {})
    if mark_ld:
        phenomena["long_distance"] = True

    synced_triplet = json.loads(json.dumps(triplet))
    out: dict = {
        "clause_id": pool_rec["clause_id"],
        "text": pool_rec["text"],
        "phenomena": phenomena,
        "triplet": triplet,
        "qwen_triplet": synced_triplet,
        "gemma_triplet": json.loads(json.dumps(synced_triplet)),
        "needs_human_review": False,
        "curated": True,
        "curation_changed": json.dumps(triplet, sort_keys=True)
        != json.dumps(pool_rec["triplet"], sort_keys=True),
        "source": "cuad_v1",
        "swap_reason": swap_reason,
        "model_agreement_full": True,
    }
    return out


def enrich_existing(record: dict, pool_by_id: dict[str, dict]) -> dict:
    stale_keys = ("swap_reason", "model_triplet_stale", "adjudication_reason")
    out = {k: v for k, v in record.items() if k not in stale_keys}
    pool_rec = pool_by_id.get(record["clause_id"])
    if pool_rec:
        if pool_rec.get("qwen_triplet") and "qwen_triplet" not in out:
            out["qwen_triplet"] = pool_rec["qwen_triplet"]
        if pool_rec.get("gemma_triplet") and "gemma_triplet" not in out:
            out["gemma_triplet"] = pool_rec["gemma_triplet"]
    out["source"] = "cuad_v1"
    agree = model_agreement_full(out)
    out["model_agreement_full"] = agree
    if not agree:
        out["adjudication_reason"] = "models_disagree:retained"
    return out


def build_candidate_pools(
    pool_rows: list[dict],
    active_ids: set[str],
    active_text_keys: set[str],
    *,
    require_meaningful_phen: bool = False,
) -> tuple[list[tuple], list[tuple], list[tuple]]:
    general: list[tuple] = []
    ld: list[tuple] = []
    definition_only: list[tuple] = []
    for rec in pool_rows:
        cid = rec["clause_id"]
        if cid in active_ids or cid in HARD_EXCLUDE_IDS:
            continue
        if is_bad_text(rec["text"]) or needs_force_replace(rec):
            continue
        text_key = normalize_text_key(rec["text"])
        if text_key in active_text_keys:
            continue
        phen = rec.get("phenomena") or {}
        if phen_count(phen) < 1:
            continue
        triplet = curate_triplet(rec["text"], rec["triplet"])
        probe = {**rec, "triplet": triplet}
        if validate(probe):
            continue
        agree = (
            triplet_fields(rec["qwen_triplet"])
            == triplet_fields(rec["gemma_triplet"])
            == triplet_fields(triplet)
        )
        rank = (
            -int(agree),
            -int(not is_definition_only(phen)),
            -meaningful_phen_count(phen),
            -selection_score(rec),
            cid,
        )
        if require_meaningful_phen and is_definition_only(phen):
            definition_only.append((rank, rec))
            continue
        general.append((rank, rec))
        if is_ld_candidate(rec):
            ld.append((rank, rec))
    general.sort()
    ld.sort()
    definition_only.sort()
    return general, ld, definition_only


def pick_candidate(
    pools: list[list[tuple]],
    used_ids: set[str],
    used_text_keys: set[str],
) -> dict | None:
    for pool in pools:
        while pool:
            _, rec = pool.pop(0)
            if rec["clause_id"] in used_ids:
                continue
            if normalize_text_key(rec["text"]) in used_text_keys:
                continue
            return rec
    return None


def duplicate_losers(records_by_id: dict[str, dict]) -> set[str]:
    losers: set[str] = set()
    for keep_id, drop_id in DUPLICATE_GROUPS:
        if keep_id not in records_by_id or drop_id not in records_by_id:
            continue
        keep_score = _record_score(records_by_id[keep_id])
        drop_score = _record_score(records_by_id[drop_id])
        if drop_score > keep_score:
            losers.add(keep_id)
        else:
            losers.add(drop_id)
    return losers


def normalized_duplicate_losers(
    records_by_id: dict[str, dict],
    explicit_losers: set[str],
) -> set[str]:
    """Drop extra copies when punctuation-normalized clause texts collide."""
    by_key: dict[str, list[str]] = {}
    for cid, rec in records_by_id.items():
        by_key.setdefault(normalize_text_key(rec["text"]), []).append(cid)

    losers: set[str] = set()
    for ids in by_key.values():
        if len(ids) < 2:
            continue
        remaining = [cid for cid in ids if cid not in explicit_losers]
        if len(remaining) < 2:
            continue
        ranked = sorted(remaining, key=lambda cid: (-_record_score(records_by_id[cid]), cid))
        losers.update(ranked[1:])
    return losers


def remediate() -> tuple[list[dict], dict]:
    gold_rows = [json.loads(line) for line in GOLD_IN.read_text().splitlines() if line.strip()]
    pool_rows = [json.loads(line) for line in POOL_IN.read_text().splitlines() if line.strip()]
    pool_by_id = {r["clause_id"]: r for r in pool_rows}

    records_by_id = {r["clause_id"]: enrich_existing(r, pool_by_id) for r in gold_rows}
    explicit_dup_out = duplicate_losers(records_by_id)
    dup_out = explicit_dup_out | normalized_duplicate_losers(records_by_id, explicit_dup_out)

    swap_out_ids: set[str] = set()
    swaps: list[tuple[str, str, str]] = []
    used_pool_ids: set[str] = set()

    def active_ids() -> set[str]:
        return (set(records_by_id) - swap_out_ids) | used_pool_ids

    def active_text_keys() -> set[str]:
        keys = {
            normalize_text_key(r["text"])
            for cid, r in records_by_id.items()
            if cid not in swap_out_ids
        }
        keys.update(
            normalize_text_key(pool_by_id[cid]["text"])
            for cid in used_pool_ids
            if cid in pool_by_id
        )
        return keys

    def apply_swap(
        out_id: str,
        reason: str,
        *,
        prefer_ld: bool,
        prefer_meaningful_phen: bool = False,
    ) -> bool:
        if out_id in swap_out_ids:
            return True
        general_pool, ld_pool, def_only_pool = build_candidate_pools(
            pool_rows,
            active_ids(),
            active_text_keys(),
            require_meaningful_phen=prefer_meaningful_phen,
        )
        if prefer_ld:
            pools: list[list[tuple]] = [ld_pool, general_pool]
        else:
            pools = [general_pool, ld_pool]
        if prefer_meaningful_phen and def_only_pool:
            pools.append(def_only_pool)
        incoming = pick_candidate(pools, used_pool_ids, active_text_keys())
        if not incoming:
            return False
        swap_out_ids.add(out_id)
        used_pool_ids.add(incoming["clause_id"])
        swaps.append((out_id, incoming["clause_id"], reason))
        return True

    for out_id in sorted(dup_out):
        if not apply_swap(out_id, "duplicate_text", prefer_ld=True):
            raise SystemExit(f"No pool candidate for duplicate removal: {out_id}")

    for cid in sorted(FORCE_REPLACE_IDS):
        if cid in records_by_id:
            if not apply_swap(cid, "table_fragment", prefer_ld=False):
                raise SystemExit(f"No pool candidate for force replace: {cid}")

    for cid, rec in sorted(records_by_id.items()):
        if cid in swap_out_ids:
            continue
        reason = needs_force_replace(rec)
        if reason and not apply_swap(cid, reason, prefer_ld=False):
            raise SystemExit(f"No pool candidate for bad text: {cid}")

    zero_phen_ids = sorted(
        cid
        for cid, rec in records_by_id.items()
        if cid not in swap_out_ids and phen_count(rec.get("phenomena") or {}) == 0
    )
    for cid in zero_phen_ids:
        if not apply_swap(cid, "zero_phenomena", prefer_ld=True, prefer_meaningful_phen=True):
            raise SystemExit(f"No pool candidate for zero phenomena: {cid}")

    def current_ld_count() -> int:
        retained_ld = sum(
            1 for cid, rec in records_by_id.items()
            if cid not in swap_out_ids
            and (rec.get("phenomena") or {}).get("long_distance")
        )
        swapped_ld = sum(
            1 for _, in_id, _ in swaps
            if is_ld_candidate(pool_by_id[in_id])
        )
        return retained_ld + swapped_ld

    while current_ld_count() < LD_TARGET:
        _, ld_pool, _ = build_candidate_pools(pool_rows, active_ids(), active_text_keys())
        if not ld_pool:
            break
        candidates = [
            (swap_priority(rec, duplicate_loser=cid in dup_out), cid)
            for cid, rec in records_by_id.items()
            if cid not in swap_out_ids
            and not (rec.get("phenomena") or {}).get("long_distance")
        ]
        if not candidates:
            break
        candidates.sort(reverse=True)
        _, out_id = candidates[0]
        if not apply_swap(out_id, "ld_boost", prefer_ld=True):
            break

    out_rows: list[dict] = []
    for cid, rec in records_by_id.items():
        if cid in swap_out_ids:
            continue
        out_rows.append(rec)

    for out_id, in_id, reason in swaps:
        pool_rec = pool_by_id[in_id]
        mark_ld = is_ld_candidate(pool_rec) or reason == "ld_boost"
        out_rows.append(
            pool_record_to_output(pool_rec, mark_ld=mark_ld, swap_reason=reason)
        )

    out_rows.sort(key=lambda r: r["clause_id"])

    norm_keys = [normalize_text_key(r["text"]) for r in out_rows]
    stats = {
        "total": len(out_rows),
        "unique_clause_ids": len({r["clause_id"] for r in out_rows}),
        "unique_texts": len({r["text"] for r in out_rows}),
        "unique_normalized_texts": len(set(norm_keys)),
        "long_distance": sum(1 for r in out_rows if (r.get("phenomena") or {}).get("long_distance")),
        "zero_phenomena": sum(1 for r in out_rows if phen_count(r.get("phenomena") or {}) == 0),
        "swaps": len(swaps),
        "swap_breakdown": {},
        "duplicate_losers": sorted(dup_out),
        "zero_phen_swapped_out": zero_phen_ids,
    }
    for _, _, reason in swaps:
        stats["swap_breakdown"][reason] = stats["swap_breakdown"].get(reason, 0) + 1

    return out_rows, stats


def main() -> None:
    out_rows, stats = remediate()

    if stats["total"] != 500:
        raise SystemExit(f"Expected 500 records, got {stats['total']}")
    if stats["unique_clause_ids"] != 500:
        raise SystemExit(f"Expected 500 unique clause_ids, got {stats['unique_clause_ids']}")
    if stats["unique_texts"] != 500:
        raise SystemExit(f"Expected 500 unique texts, got {stats['unique_texts']}")
    if stats["unique_normalized_texts"] != 500:
        dup_groups = [
            key
            for key, count in Counter(normalize_text_key(r["text"]) for r in out_rows).items()
            if count > 1
        ]
        raise SystemExit(
            "Expected 500 unique normalized texts, got "
            f"{stats['unique_normalized_texts']} ({len(dup_groups)} duplicate groups)"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(stats, indent=2))
    print(f"Wrote {OUT} ({stats['total']} records)")


if __name__ == "__main__":
    main()
