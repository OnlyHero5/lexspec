#!/usr/bin/env python3
"""Build a curated 500-item LexSpec gold test set from the 869-item merge output."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/processed/gold_triplets.jsonl"
GOLD_OUT = ROOT / "data/processed/gold_triplets_500.jsonl"
TEST_OUT = ROOT / "data/processed/gold_testset_500.jsonl"
MANIFEST = ROOT / "data/processed/curated_500/manifest.json"

# Manually confirmed bad gold after QA (do not include in final 500).
HARD_EXCLUDE_IDS = frozenset({
    "C-00515",  # inverted condition / missing temporal scope
    "C-00858",  # compound govern + jurisdiction clause
})

GOVERN_RE = re.compile(r"\b(shall be governed|is governed|governed by)\b", re.I)
LAWS_RE = re.compile(
    r"\b(?:the )?(?:substantive )?laws of (?:the )?[^.;,\n]+",
    re.I,
)
WITHOUT_CONSENT_RE = re.compile(
    r"\bwithout\s+(?:the\s+)?(?:prior\s+)?(?:written\s+)?(?:express\s+)?consent\b",
    re.I,
)
SUBJECT_TO_RE = re.compile(
    r"^Subject to (?:the terms(?: and conditions)? of this Agreement|Section\s+\d)",
    re.I,
)
TERM_PREDICATES = frozenset({
    "commence", "begin", "start", "end", "expire", "renew", "continue",
    "terminate", "become", "remain", "extend",
})
MANNER_IN_CONDITION = re.compile(
    r"\b(upon giving|by giving|in writing|via overnight|with prior written notice)\b",
    re.I,
)


def is_bad_text(text: str) -> str | None:
    t = text.strip()
    if len(t) < 25:
        return "too_short"
    if re.match(r"^\s*\d{1,2}(st|nd|rd|th)?\s+day\s+of\s+\w+\s+\d{4}\s*$", t, re.I):
        return "date_only"
    if "<omitted>" in t.lower():
        return "truncated"
    if re.search(r"\*{3,}", t):
        return "redacted"
    if re.search(r"\d+t\s+h\s+day", t, re.I):
        return "ocr"
    if re.search(r"\[Implied|\[REV", t):
        return "placeholder"
    return None


def triplet_fields(t: dict) -> tuple:
    s, a, c = t.get("subject", {}), t.get("action", {}), t.get("condition", {})
    return (
        s.get("text", ""), s.get("role", ""),
        a.get("predicate", ""), a.get("object", ""),
        c.get("text", ""), c.get("type", "none"),
    )


def selection_score(record: dict) -> float:
    if is_bad_text(record["text"]):
        return -999
    qt, gt, ft = record["qwen_triplet"], record["gemma_triplet"], record["triplet"]
    qf, gf, ff = triplet_fields(qt), triplet_fields(gt), triplet_fields(ft)
    score = 0.0
    if qf == gf == ff:
        score += 60
    else:
        for i in range(6):
            if qf[i] == gf[i] == ff[i]:
                score += 8
    text = record["text"]
    phen = record.get("phenomena") or {}
    if GOVERN_RE.search(text) or re.search(r"\bconstrued in accordance\b", text, re.I):
        score -= 18
    if phen.get("passive"):
        score -= 4
    if len(text) > 700:
        score -= 8
    elif len(text) > 450:
        score -= 3
    pred = ff[2]
    if " " in pred or pred in ("be liable",):
        score -= 6
    if ff[0].startswith("["):
        score -= 20
    if text.count(";") >= 2:
        score -= 8
    elif text.count(";") == 1:
        score -= 2
    if re.search(r"\bshall not be liable\b|\bBE LIABLE\b", text, re.I):
        if pred == "be":
            score -= 4
    return score


def fix_govern(text: str, triplet: dict) -> dict:
    if not GOVERN_RE.search(text):
        return triplet
    m = LAWS_RE.search(text)
    laws = m.group(0).strip() if m else "the applicable laws"
    obj = "This Agreement" if "this agreement" in text.lower() else "the Agreement"
    exc = re.search(
        r"(without regard to (?:its )?conflicts of law[^.;,\n]*|except as[^.;,\n]+)",
        text,
        re.I,
    )
    triplet = {
        **triplet,
        "subject": {"text": laws, "role": "other"},
        "action": {"predicate": "govern", "object": obj},
        "condition": (
            {"text": exc.group(1).strip(), "type": "exception"}
            if exc else {"text": "", "type": "none"}
        ),
    }
    return triplet


def fix_multiword_predicate(triplet: dict) -> dict:
    pred = (triplet.get("action") or {}).get("predicate", "")
    mapping = {
        "be liable": "liable",
        "be available": "available",
        "be valid": "valid",
        "establish and maintain": "maintain",
        "keep and maintain": "maintain",
    }
    if pred in mapping:
        triplet = {**triplet, "action": {**triplet["action"], "predicate": mapping[pred]}}
    return triplet


def fix_interpret_construe(text: str, triplet: dict) -> dict:
    if not re.search(r"\b(construed|interpreted|enforced) (?:in accordance|exclusively)\b", text, re.I):
        return triplet
    m = LAWS_RE.search(text)
    if not m:
        return triplet
    laws = m.group(0).strip()
    obj = "This Agreement" if "this agreement" in text.lower() else "the Agreement"
    pred = "construe" if "constru" in text.lower() else "interpret"
    triplet = {
        **triplet,
        "subject": {"text": laws, "role": "other"},
        "action": {"predicate": pred, "object": obj},
    }
    exc = re.search(r"(without regard to[^.;,\n]+|notwithstanding[^.;,\n]+)", text, re.I)
    triplet["condition"] = (
        {"text": exc.group(1).strip(), "type": "exception"}
        if exc else {"text": "", "type": "none"}
    )
    return triplet


def fix_liable(triplet: dict) -> dict:
    pred = (triplet.get("action") or {}).get("predicate", "")
    obj = (triplet.get("action") or {}).get("object", "")
    if pred in ("be liable", "be Liable"):
        triplet = {**triplet, "action": {**triplet["action"], "predicate": "liable"}}
    elif pred == "be" and re.search(r"\bliable\b", obj, re.I):
        triplet = {**triplet, "action": {**triplet["action"], "predicate": "liable"}}
    return triplet


def fix_term_role(text: str, triplet: dict) -> dict:
    pred = (triplet.get("action") or {}).get("predicate", "")
    subj = (triplet.get("subject") or {}).get("text", "")
    if pred in TERM_PREDICATES and re.search(
        r"\b(term|agreement|license|this agreement)\b", subj, re.I
    ):
        triplet = {**triplet, "subject": {**triplet["subject"], "role": "other"}}
    return triplet


def fix_effective(text: str, triplet: dict) -> dict:
    pred = (triplet.get("action") or {}).get("predicate", "")
    if pred == "effective" and re.search(r"\b(shall become|is) effective\b", text, re.I):
        subj = (triplet.get("subject") or {}).get("text") or "This Agreement"
        triplet = {
            **triplet,
            "subject": {"text": subj, "role": "other"},
            "action": {"predicate": "be", "object": "effective"},
        }
    return triplet


def fix_section_role(triplet: dict) -> dict:
    subj = (triplet.get("subject") or {}).get("text", "")
    if re.match(r"^Section\s+\d", subj, re.I):
        triplet = {**triplet, "subject": {**triplet["subject"], "role": "other"}}
    return triplet


def strip_without_consent(text: str, triplet: dict) -> dict:
    cond = triplet.get("condition") or {}
    ctext = (cond.get("text") or "").strip()
    if not ctext or not WITHOUT_CONSENT_RE.search(text):
        return triplet
    if re.search(r"\bshall not\b|\bmay not\b|\bneither\b.*\bshall\b", text, re.I):
        if WITHOUT_CONSENT_RE.search(ctext):
            triplet = {**triplet, "condition": {"text": "", "type": "none"}}
    return triplet


def strip_procedural_conditions(triplet: dict) -> dict:
    cond = triplet.get("condition") or {}
    ctext = (cond.get("text") or "").strip()
    if ctext and SUBJECT_TO_RE.match(ctext):
        triplet = {**triplet, "condition": {"text": "", "type": "none"}}
    return triplet


def strip_manner_conditions(triplet: dict) -> dict:
    cond = triplet.get("condition") or {}
    ctext = (cond.get("text") or "").strip()
    if ctext and MANNER_IN_CONDITION.search(ctext):
        if not re.search(r"\b(if|unless|except|provided that|in the event)\b", ctext, re.I):
            triplet = {**triplet, "condition": {"text": "", "type": "none"}}
    return triplet


def normalize_condition(triplet: dict) -> dict:
    cond = triplet.get("condition") or {}
    ctext = (cond.get("text") or "").strip()
    ctype = cond.get("type", "none")
    if not ctext:
        triplet = {**triplet, "condition": {"text": "", "type": "none"}}
    elif ctype == "none":
        triplet = {**triplet, "condition": {"text": ctext, "type": "temporal"}}
    return triplet


def trim_object(triplet: dict, max_len: int = 175) -> dict:
    obj = (triplet.get("action") or {}).get("object", "")
    if len(obj) <= max_len:
        return triplet
    cut = obj[:max_len].rsplit(" ", 1)[0]
    return {**triplet, "action": {**triplet["action"], "object": cut}}


def curate_triplet(text: str, triplet: dict) -> dict:
    t = dict(triplet)
    t = fix_govern(text, t)
    t = fix_interpret_construe(text, t)
    t = fix_liable(t)
    t = fix_multiword_predicate(t)
    t = fix_term_role(text, t)
    t = fix_effective(text, t)
    t = fix_section_role(t)
    t = strip_without_consent(text, t)
    t = strip_procedural_conditions(t)
    t = strip_manner_conditions(t)
    t = normalize_condition(t)
    t = trim_object(t)
    return t


def validate(record: dict) -> list[str]:
    issues = []
    text = record["text"]
    if is_bad_text(text):
        issues.append("bad_text")
    t = record["triplet"]
    s, a, c = t["subject"], t["action"], t["condition"]
    pred, subj, obj = a.get("predicate", ""), s.get("text", ""), a.get("object", "")
    if " " in pred:
        issues.append("multiword_pred")
    if subj and obj and subj.lower() == obj.lower() and pred not in ("mean", "be"):
        role = s.get("role", "")
        if not (role == "other" and pred in ("renew", "continue", "begin", "expire", "extend")):
            issues.append("subj_eq_obj")
    if GOVERN_RE.search(text) and pred == "govern":
        if re.match(r"^(This Agreement|this Agreement)", subj):
            issues.append("gov_inversion")
        if " and construed" in subj.lower():
            issues.append("gov_corrupt")
    ctext, ctype = c.get("text", ""), c.get("type", "none")
    if ctext and ctype == "none":
        issues.append("cond_mismatch")
    if not ctext and ctype != "none":
        issues.append("cond_empty")
    if len(obj) > 185:
        issues.append("long_obj")
    return issues


def main() -> None:
    records = [json.loads(line) for line in SRC.open()]
    clean_pool: list[tuple[float, dict]] = []
    for r in records:
        if selection_score(r) <= 0:
            continue
        triplet = curate_triplet(r["text"], r["triplet"])
        rec = {
            "clause_id": r["clause_id"],
            "text": r["text"],
            "phenomena": r.get("phenomena", {}),
            "triplet": triplet,
            "needs_human_review": False,
            "curated": True,
            "curation_changed": json.dumps(triplet, sort_keys=True)
            != json.dumps(r["triplet"], sort_keys=True),
            "model_agreement_full": triplet_fields(r["qwen_triplet"])
            == triplet_fields(r["gemma_triplet"])
            == triplet_fields(triplet),
        }
        if validate(rec):
            continue
        if rec["clause_id"] in HARD_EXCLUDE_IDS:
            continue
        clean_pool.append((selection_score(r), rec))

    clean_pool.sort(
        key=lambda x: (
            -int(x[1].get("model_agreement_full", False)),
            -x[0],
            x[1]["clause_id"],
        )
    )
    selected = [rec for _, rec in clean_pool[:500]]
    if len(selected) < 500:
        raise SystemExit(
            f"Only {len(selected)} validation-clean records available; need 500."
        )

    issues = [(r["clause_id"], validate(r)) for r in selected if validate(r)]
    out_rows = selected
    out_rows.sort(key=lambda x: x["clause_id"])
    changed = sum(1 for r in out_rows if r.get("curation_changed"))
    perfect = sum(1 for r in out_rows if r.get("model_agreement_full"))
    GOLD_OUT.parent.mkdir(parents=True, exist_ok=True)
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

    manifest = {
        "total": len(out_rows),
        "source_total": len(records),
        "clean_pool_size": len(clean_pool),
        "full_model_agreement": perfect,
        "curation_changed": changed,
        "validation_issues": len(issues),
        "validation_clean_rate": round((len(out_rows) - len(issues)) / len(out_rows), 4),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    if issues:
        print("Sample issues:", issues[:10])


if __name__ == "__main__":
    main()
