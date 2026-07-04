#!/usr/bin/env python3
"""Final pass: replace fragments/duplicates, fix remaining B/C/D, trim long objects."""

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
    validate,
)

GOLD_PATH = ROOT / "data/processed/gold_triplets_500.jsonl"
TEST_PATH = ROOT / "data/processed/gold_testset_500.jsonl"
SRC_PATH = ROOT / "data/processed/gold_triplets.jsonl"
MANIFEST = ROOT / "data/processed/curated_500/manifest.json"

REPLACE_IDS = frozenset({
    "C-00010",   # SUBPARAGRAH typo in source text
    "C-00078",   # [**] redaction in royalty amounts
    "C-00884",   # [**] redaction + compound clause
})

MANUAL: dict[str, dict] = {
    "C-00015": {
        "action": {
            "object": (
                "all laws, regulations, license conditions and decisions of the CRTC and "
                "applicable municipal, provincial and federal authorities (\"Applicable Law\")"
            )
        },
        "phenomena": {"conditional": False},
    },
    "C-00280": {
        "subject": {"text": "Limitation of liability as described in this article", "role": "other"},
        "action": {"predicate": "apply", "object": "the limitations described in this article"},
        "condition": {
            "text": (
                "in case the damage or loss is caused by a Party's willful misconduct "
                "(including fraud) or gross negligence, or in case of a breach of a Party's "
                "obligation under article 11 (confidentiality) and article 15 "
                "(indemnification for breach of intellectual property rights)"
            ),
            "type": "trigger",
        },
        "phenomena": {"conditional": True},
    },
    "C-00632": {
        "action": {"object": "this Agreement or any rights or obligations hereunder"},
        "condition": {"text": "", "type": "none"},
        "phenomena": {"conditional": False, "negation": True},
    },
    "C-00845": {
        "action": {"object": "termination of this Agreement"},
        "condition": {
            "text": "unless this Agreement was terminated for a material breach",
            "type": "exception",
        },
        "phenomena": {"conditional": True},
    },
    "C-00037": {
        "action": {
            "object": (
                "an exclusive, non-transferable and non-sublicensable license to reproduce, "
                "perform, display, transmit and distribute the Licensed Content"
            )
        },
    },
    "C-00209": {
        "action": {
            "object": (
                "the exclusive, non-transferable right and license to promote, distribute "
                "and sell the Products identified in Exhibit A"
            )
        },
    },
    "C-00263": {
        "action": {
            "object": (
                "5% of capital expenditure sign-on fees and 5% commission of revenue "
                "from third party franchisees"
            )
        },
    },
    "C-00363": {
        "action": {
            "object": (
                "Village Media Company or its Affiliates any live (or near live) rights "
                "to Exploit events or content owned or controlled by PFHOF"
            )
        },
    },
    "C-00367": {
        "action": {
            "object": (
                "the rights of PFHOF in and to any PFHOF Work or the validity, legality "
                "or enforceability of this Agreement"
            )
        },
    },
    "C-00394": {
        "action": {
            "object": (
                "a non-exclusive, non-transferable license to make, have made, import, "
                "offer for sale and sell the LICENSED PRODUCT(S) and LICENSED SERVICE(S)"
            )
        },
    },
    "C-00522": {
        "action": {
            "object": (
                "a revocable, royalty-free, non-exclusive license to use the marks "
                "set forth on Exhibit E (XSPA's Marks)"
            )
        },
    },
    "C-00528": {
        "phenomena": {"conditional": True, "negation": True},
    },
    "C-00689": {
        "action": {
            "object": (
                "the Issuer, the Depositor and the Indenture Trustee access to the "
                "records and documents"
            )
        },
    },
    "C-00883": {
        "phenomena": {"conditional": True, "negation": True},
    },
}


def _deep_merge(base: dict, patch: dict) -> dict:
    out = json.loads(json.dumps(base))
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def trim_object(triplet: dict, max_len: int = 175) -> dict:
    obj = (triplet.get("action") or {}).get("object", "")
    if len(obj) <= max_len:
        return triplet
    cut = obj[:max_len].rsplit(" ", 1)[0]
    return {**triplet, "action": {**triplet["action"], "object": cut}}


def build_replacement_pool(current_ids: set[str]) -> list[dict]:
    src = [json.loads(line) for line in SRC_PATH.read_text().splitlines() if line.strip()]
    pool: list[tuple[float, dict]] = []
    for r in src:
        cid = r["clause_id"]
        if cid in current_ids or cid in HARD_EXCLUDE_IDS or cid in REPLACE_IDS:
            continue
        if is_bad_text(r["text"]):
            continue
        triplet = curate_triplet(r["text"], r["triplet"])
        rec = {
            "clause_id": cid,
            "text": r["text"],
            "phenomena": r.get("phenomena", {}),
            "triplet": triplet,
            "needs_human_review": False,
            "curated": True,
            "curation_changed": True,
            "model_agreement_full": False,
            "bcd_fix_notes": "replacement from excluded clean pool",
            "replaced_from": None,
        }
        if validate(rec):
            continue
        pool.append((selection_score(r), rec))
    pool.sort(key=lambda x: (-x[0], x[1]["clause_id"]))
    return [rec for _, rec in pool]


def main() -> None:
    records = [json.loads(line) for line in GOLD_PATH.read_text().splitlines() if line.strip()]
    by_id = {r["clause_id"]: r for r in records}
    current_ids = set(by_id)

    pool = build_replacement_pool(current_ids)
    if len(pool) < len(REPLACE_IDS):
        raise SystemExit(f"Need {len(REPLACE_IDS)} replacements, only {len(pool)} available.")

    replacements = {}
    pool_iter = iter(pool)
    for old_id in sorted(REPLACE_IDS):
        rep = next(pool_iter)
        rep = json.loads(json.dumps(rep))
        rep["replaced_from"] = old_id
        rep["clause_id"] = old_id  # keep slot id stable for eval continuity
        replacements[old_id] = rep

    out: list[dict] = []
    for rec in records:
        cid = rec["clause_id"]
        if cid in replacements:
            out.append(replacements[cid])
            continue
        if cid in MANUAL:
            patch = MANUAL[cid]
            if "triplet" not in patch:
                t = rec["triplet"]
                if "subject" in patch:
                    t["subject"] = _deep_merge(t["subject"], patch["subject"])
                if "action" in patch:
                    t["action"] = _deep_merge(t["action"], patch["action"])
                if "condition" in patch:
                    t["condition"] = patch["condition"]
                rec["triplet"] = trim_object(t)
            if "phenomena" in patch:
                rec["phenomena"] = {**rec.get("phenomena", {}), **patch["phenomena"]}
            rec["bcd_fix_notes"] = (rec.get("bcd_fix_notes", "") + "; final_pass").strip("; ")
        rec["triplet"] = trim_object(rec["triplet"])
        out.append(rec)

    out.sort(key=lambda x: x["clause_id"])
    assert len(out) == 500

    val_fail = [(r["clause_id"], validate(r)) for r in out if validate(r)]

    with GOLD_PATH.open("w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with TEST_PATH.open("w") as f:
        for r in out:
            f.write(json.dumps({
                "clause_id": r["clause_id"],
                "text": r["text"],
                "phenomena": r.get("phenomena", {}),
            }, ensure_ascii=False) + "\n")

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest["final_pass_replacements"] = {
        old: rep.get("text", "")[:80] for old, rep in replacements.items()
    }
    manifest["final_pass_validation_failures"] = len(val_fail)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({
        "replacements": len(replacements),
        "validation_failures": len(val_fail),
        "sample_replacements": {k: v["text"][:60] for k, v in list(replacements.items())[:3]},
    }, indent=2))


if __name__ == "__main__":
    main()
