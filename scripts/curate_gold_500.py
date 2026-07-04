#!/usr/bin/env python3
"""Second-pass LexSpec gold curation for the 500-item test set."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD_IN = ROOT / "data/processed/gold_triplets_500.jsonl"
GOLD_OUT = ROOT / "data/processed/gold_triplets_500.jsonl"
TEST_OUT = ROOT / "data/processed/gold_testset_500.jsonl"

GOVERN_RE = re.compile(r"\b(shall be governed|is governed|governed by)\b", re.I)
WITHOUT_CONSENT_RE = re.compile(
    r"\bwithout\s+(?:the\s+)?(?:prior\s+)?(?:written\s+)?consent\b", re.I
)
TERM_PREDICATES = frozenset({
    "commence", "begin", "start", "end", "expire", "renew", "continue", "terminate",
    "become", "remain", "extend", "automatically renew",
})
MANNER_IN_CONDITION = re.compile(
    r"\b(upon giving|by giving|in writing|via overnight|with prior written notice)\b",
    re.I,
)


def _strip_without_consent_condition(text: str, triplet: dict) -> dict:
    """Inline 'without consent' on prohibitions is not a subordinate condition."""
    cond = triplet.get("condition") or {}
    ctext = (cond.get("text") or "").strip()
    if not ctext:
        return triplet
    if not WITHOUT_CONSENT_RE.search(text):
        return triplet
    if re.search(r"\bshall not\b|\bmay not\b|\bneither\b.*\bshall\b", text, re.I):
        if WITHOUT_CONSENT_RE.search(ctext):
            triplet = {**triplet, "condition": {"text": "", "type": "none"}}
    return triplet


def _fix_govern(text: str, triplet: dict) -> dict:
    if not GOVERN_RE.search(text):
        return triplet
    m = re.search(
        r"governed by (?:and construed in accordance with )?"
        r"((?:the )?(?:substantive )?(?:laws of (?:the )?[^.;,\n]+|[^.;,\n]+ law))",
        text,
        re.I,
    )
    if not m:
        m = re.search(r"governed by ([^.;,\n]+)", text, re.I)
    laws = m.group(1).strip() if m else "the applicable laws"
    if not laws.lower().startswith("the "):
        laws = f"the laws of {laws}" if "law" not in laws.lower() else laws
    obj = "This Agreement" if "this agreement" in text.lower() else "the Agreement"
    triplet = {
        **triplet,
        "subject": {"text": laws if laws.lower().startswith("the") else f"the {laws}", "role": "other"},
        "action": {"predicate": "govern", "object": obj},
    }
    exc = re.search(r"(without regard to[^.;]+|except as[^.;]+)", text, re.I)
    if exc:
        triplet["condition"] = {"text": exc.group(1).strip(), "type": "exception"}
    else:
        triplet["condition"] = {"text": "", "type": "none"}
    return triplet


def _fix_liable(triplet: dict) -> dict:
    pred = (triplet.get("action") or {}).get("predicate", "")
    if pred in ("be liable", "be Liable"):
        triplet = {**triplet, "action": {**triplet["action"], "predicate": "liable"}}
    if pred == "be" and re.search(r"\bliable\b", (triplet.get("action") or {}).get("object", ""), re.I):
        triplet = {**triplet, "action": {**triplet["action"], "predicate": "liable"}}
    return triplet


def _fix_term_role(text: str, triplet: dict) -> dict:
    pred = (triplet.get("action") or {}).get("predicate", "")
    subj = (triplet.get("subject") or {}).get("text", "")
    if pred in TERM_PREDICATES and re.search(
        r"\b(term|agreement|license|this agreement)\b", subj, re.I
    ):
        triplet = {
            **triplet,
            "subject": {**triplet["subject"], "role": "other"},
        }
    return triplet


def _fix_effective_predicate(text: str, triplet: dict) -> dict:
    pred = (triplet.get("action") or {}).get("predicate", "")
    if pred == "effective" and re.search(r"\b(shall become|is) effective\b", text, re.I):
        triplet = {
            **triplet,
            "subject": triplet.get("subject") or {"text": "", "role": "other"},
            "action": {"predicate": "be", "object": "effective"},
        }
    return triplet


def _fix_section_subject(text: str, triplet: dict) -> dict:
    subj = (triplet.get("subject") or {}).get("text", "")
    if re.match(r"^Section\s+\d", subj, re.I):
        triplet = {
            **triplet,
            "subject": {**triplet["subject"], "role": "other"},
        }
    return triplet


def _trim_object(triplet: dict, max_len: int = 140) -> dict:
    obj = (triplet.get("action") or {}).get("object", "")
    if len(obj) <= max_len:
        return triplet
    cut = obj[:max_len].rsplit(" ", 1)[0]
    triplet = {**triplet, "action": {**triplet["action"], "object": cut}}
    return triplet


def _fix_manner_condition(triplet: dict) -> dict:
    cond = triplet.get("condition") or {}
    ctext = (cond.get("text") or "").strip()
    if ctext and MANNER_IN_CONDITION.search(ctext):
        if not re.search(r"\b(if|when|unless|except|provided that|in the event)\b", ctext, re.I):
            triplet = {**triplet, "condition": {"text": "", "type": "none"}}
    return triplet


def _fix_condition_consistency(triplet: dict) -> dict:
    cond = triplet.get("condition") or {}
    ctext = (cond.get("text") or "").strip()
    ctype = cond.get("type", "none")
    if not ctext:
        triplet = {**triplet, "condition": {"text": "", "type": "none"}}
    elif ctype == "none":
        triplet = {**triplet, "condition": {**cond, "type": "temporal"}}
    return triplet


def curate_record(record: dict) -> tuple[dict, bool]:
    text = record["text"]
    orig = json.dumps(record["triplet"], sort_keys=True)
    t = dict(record["triplet"])
    t = _fix_govern(text, t)
    t = _fix_liable(t)
    t = _fix_term_role(text, t)
    t = _fix_effective_predicate(text, t)
    t = _fix_section_subject(text, t)
    t = _strip_without_consent_condition(text, t)
    t = _fix_manner_condition(t)
    t = _fix_condition_consistency(t)
    t = _trim_object(t)
    changed = json.dumps(t, sort_keys=True) != orig
    record = {**record, "triplet": t}
    if changed:
        record["curation_changed"] = True
        note = record.get("curation_note") or ""
        record["curation_note"] = (note + "; pass2 rule fix").strip("; ")
    return record, changed


def validate(record: dict) -> list[str]:
    issues = []
    text = record["text"]
    if len(text.strip()) < 25:
        issues.append("short_text")
    if "<omitted>" in text.lower() or "***" in text:
        issues.append("bad_text")
    t = record["triplet"]
    s, a, c = t.get("subject", {}), t.get("action", {}), t.get("condition", {})
    pred = (a.get("predicate") or "").strip()
    subj = (s.get("text") or "").strip()
    obj = (a.get("object") or "").strip()
    if " " in pred:
        issues.append("multiword_pred")
    if subj and obj and subj.lower() == obj.lower() and pred not in ("mean", "be"):
        issues.append("subj_eq_obj")
    if GOVERN_RE.search(text) and pred == "govern":
        if re.match(r"^(This Agreement|this Agreement)", subj):
            issues.append("gov_inversion")
    ctext = (c.get("text") or "").strip()
    ctype = c.get("type", "none")
    if ctext and ctype == "none":
        issues.append("cond_mismatch")
    if not ctext and ctype != "none":
        issues.append("cond_empty")
    if len(obj) > 160:
        issues.append("long_obj")
    return issues


def main() -> None:
    rows = [json.loads(line) for line in GOLD_IN.open()]
    changed = 0
    out_rows = []
    bad = []
    for r in rows:
        r, ch = curate_record(r)
        if ch:
            changed += 1
        iss = validate(r)
        if iss:
            bad.append((r["clause_id"], iss))
        out_rows.append(r)

    out_rows.sort(key=lambda x: x["clause_id"])
    with GOLD_OUT.open("w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with TEST_OUT.open("w") as f:
        for r in out_rows:
            f.write(json.dumps({
                "clause_id": r["clause_id"],
                "text": r["text"],
                "phenomena": r.get("phenomena", {}),
            }, ensure_ascii=False) + "\n")

    clean = sum(1 for r in out_rows if not validate(r))
    print(f"Pass2 changed: {changed}/{len(rows)}")
    print(f"Validation clean: {clean}/{len(rows)} ({100*clean/len(rows):.1f}%)")
    if bad:
        print(f"Remaining issues: {len(bad)}")
        for cid, iss in bad[:15]:
            print(f"  {cid}: {iss}")


if __name__ == "__main__":
    main()
