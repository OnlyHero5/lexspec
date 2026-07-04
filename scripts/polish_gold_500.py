#!/usr/bin/env python3
"""Conservative polish: repair polluted subjects and broken frames; roles via fix_triplet."""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/processed/gold_triplets_500.jsonl"
TEST = ROOT / "data/processed/gold_testset_500.jsonl"

sys.path.insert(0, str(ROOT / "scripts"))
from fix_annotations_500 import (  # noqa: E402
    AUTO_RENEW_RE,
    GOVERN_RE,
    TERM_PREDICATES,
    _agreement_np,
    _extract_laws,
    fix_triplet,
)

MAX_SUBJECT = 80
MAX_SUBJECT_HARD = 100

PREFIX_START_RE = re.compile(
    r"^(?:Subject to|During|If |Notwithstanding|In order|Except |For so long|All |"
    r"The provisions|The Parties acknowledge|In the event|Prior to|Upon |Unless |"
    r"In addition|With respect to|For purposes|Failure to|Not more than|"
    r"NEITHER PARTY|EXCEPT FOR|Nothwithstanding)[^,;]*?,?\s*",
    re.I,
)
MODAL_IN_SUBJ_RE = re.compile(r"\b(?:shall|may|must|will|grants|warrants)\b", re.I)
NEITHER_PARTY_RE = re.compile(r"\b(Neither [Pp]arty|NEITHER PARTY)\b")
PASSIVE_BY_RE = re.compile(
    r"\bby\s+([A-Z][^.;,\n]+?)(?:\.|,|;|$|\s+without|\s+except|\s+which|\s+in\s+connection|\s+prior)",
    re.I,
)


def is_polluted_subject(subj: str) -> bool:
    subj = (subj or "").strip()
    if not subj:
        return False
    if len(subj) > MAX_SUBJECT:
        return True
    if PREFIX_START_RE.match(subj):
        return True
    if MODAL_IN_SUBJ_RE.search(subj):
        return True
    return False


def _strip_prefixes(text: str) -> str:
    out = text.strip()
    for _ in range(6):
        nxt = PREFIX_START_RE.sub("", out, count=1).strip()
        if nxt == out:
            break
        out = nxt
    return out


def model_consensus_subject(record: dict) -> str | None:
    q = record.get("qwen_triplet") or {}
    g = record.get("gemma_triplet") or {}
    qs = (q.get("subject") or {}).get("text", "").strip()
    gs = (g.get("subject") or {}).get("text", "").strip()
    if qs and qs == gs and not is_polluted_subject(qs):
        return qs
    return None


def extract_subject_near_predicate(text: str, predicate: str) -> str | None:
    pred = (predicate or "").strip().lower()
    if not pred:
        return None
    if pred == "govern":
        laws = _extract_laws(text)
        if laws != "the applicable laws":
            return laws[:MAX_SUBJECT_HARD]
    if pred in TERM_PREDICATES:
        m = re.search(r"\b((?:This |The )?[Aa]greement)\b", text)
        if m:
            return m.group(1)
    if NEITHER_PARTY_RE.search(text) and re.search(
        r"\b(?:Neither [Pp]arty|NEITHER PARTY)\s+may\s+assign\b", text, re.I
    ):
        return NEITHER_PARTY_RE.search(text).group(1)  # type: ignore[union-attr]
    if re.search(r"\b(?:may|shall)\s+be\s+\w+", text, re.I):
        by_m = PASSIVE_BY_RE.search(text)
        if by_m:
            return by_m.group(1).strip()[:MAX_SUBJECT_HARD]
    pat = rf"([A-Z][^.;]{{0,70}}?)\s+(?:shall|may|must|will)\s+(?:not\s+)?{re.escape(pred)}\b"
    for m in re.finditer(pat, text, re.I):
        subj = _strip_prefixes(m.group(1).strip())
        if subj and not is_polluted_subject(subj):
            return subj
    party_pat = re.compile(
        r"((?:Each of )?[A-Z][A-Za-z0-9&'().,\-]+(?:\s+(?:and|or)\s+[A-Z][A-Za-z0-9&'().,\-]+)*|"
        r"(?:Neither|Either|each) [Pp]arty|the [Pp]arties|The Parties|"
        r"Company|Customer|Manufacturer|Licensor|Licensee|Supplier|Investor|"
        r"Erchonia|ESSI|Exact|Pfizer|Skype|MusclePharm|Todos|AT&T|"
        r"ExxonMobil and FCE|Dexcel and Kitov|the University and ArTara)"
        r"(?:\'s)?\s+(?:shall|may|must|will|hereby grants|grants|warrants|agrees)",
        re.I,
    )
    idx = text.lower().find(pred) if pred in text.lower() else -1
    if idx >= 0:
        hits = list(party_pat.finditer(text[: idx + len(pred)]))
        if hits:
            subj = _strip_prefixes(hits[-1].group(1).strip())
            if subj and not is_polluted_subject(subj):
                return subj
    return None


def fix_auto_renew_final(text: str, triplet: dict) -> dict:
    if not AUTO_RENEW_RE.search(text):
        return triplet
    agr_m = re.match(r"^((?:This |The )?[Aa]greement)", text.strip())
    subj = agr_m.group(1) if agr_m else _agreement_np(text)
    unless = re.search(r"\bunless\b.+", text, re.I | re.DOTALL)
    condition = (
        {"text": unless.group(0).strip().rstrip("."), "type": "exception"}
        if unless
        else {"text": "", "type": "none"}
    )
    return {
        **triplet,
        "subject": {"text": subj, "role": "other"},
        "action": {"predicate": "renew", "object": ""},
        "condition": condition,
    }


def fix_govern_subject_trim(text: str, triplet: dict) -> dict:
    pred = ((triplet.get("action") or {}).get("predicate") or "").strip().lower()
    if pred != "govern" and not GOVERN_RE.search(text):
        return triplet
    subj = ((triplet.get("subject") or {}).get("text") or "").strip()
    m = re.search(
        r"\b((?:the )?(?:substantive )?laws of [^.;,\n]+?)(?:\s+and\s+all|\s+applicable|\.)",
        subj,
        re.I,
    )
    if not m:
        m = re.search(r"\b((?:the )?(?:substantive )?laws of [^.;,\n]+)", text, re.I)
    if m:
        return {**triplet, "subject": {**triplet["subject"], "text": m.group(1).strip(), "role": "other"}}
    return triplet


def fix_wrong_be_frame(text: str, triplet: dict) -> dict:
    pred = ((triplet.get("action") or {}).get("predicate") or "").strip().lower()
    if pred not in (
        "be", "have", "any", "consider", "first", "notify", "remain",
        "available", "exclude", "constitute",
    ):
        return triplet

    m = re.search(
        r"\b(Neither [Pp]arty|NEITHER PARTY|Company|[A-Z][A-Za-z0-9&]+)\s+(?:may not|shall not)\s+"
        r"(?:assign|transfer|delegate)\b",
        text,
        re.I,
    )
    if m:
        subj = m.group(1).strip()
        obj_m = re.search(
            r"\b(?:assign|transfer|delegate)(?:\s+or\s+\w+)*\s+"
            r"((?:any of )?its (?:rights|duties|obligations)[^.;]*)",
            text,
            re.I,
        )
        obj = obj_m.group(1).strip() if obj_m else "its rights or obligations under this Agreement"
        if len(obj) > 200:
            obj = obj[:200].rsplit(" ", 1)[0]
        return {
            **triplet,
            "subject": {"text": subj, "role": triplet["subject"].get("role", "prohibited_party")},
            "action": {"predicate": "assign", "object": obj},
            "condition": {"text": "", "type": "none"},
        }

    if re.search(r"\b(?:Agreement|this Agreement)\s+may be assigned\b", text, re.I):
        by_m = PASSIVE_BY_RE.search(text)
        if by_m:
            obj = "this Agreement" if "this Agreement" in text else "the Agreement"
            return {
                **triplet,
                "subject": {"text": by_m.group(1).strip(), "role": triplet["subject"].get("role", "right_holder")},
                "action": {"predicate": "assign", "object": obj},
                "condition": {"text": "", "type": "none"},
            }

    if re.search(r"\b(?:hereby )?grants?\s+to\b", text, re.I):
        gm = re.search(r"\b([A-Z][A-Za-z0-9&]+)\s+(?:hereby )?grants?\s+to\b", text, re.I)
        if gm:
            subj = gm.group(1).strip()
            obj_m = re.search(
                r"\bgrants?\s+(?:to\s+[A-Z][^,]+,\s*)?(a\s+[^.;]+?)(?:\.|;|$|\s+During|\s+Subject)",
                text,
                re.I | re.DOTALL,
            )
            obj = obj_m.group(1).strip() if obj_m else ((triplet.get("action") or {}).get("object") or "")
            if len(obj) > 200:
                obj = obj[:200].rsplit(" ", 1)[0]
            return {
                **triplet,
                "subject": {"text": subj, "role": triplet["subject"].get("role", "obligor")},
                "action": {"predicate": "grant", "object": obj},
            }

    if re.search(
        r"\b(?:IN NO EVENT SHALL|in no event shall)\s+(?:ANY PARTY|any party|NEITHER PARTY)\b",
        text,
        re.I,
    ):
        subj = "any party"
        obj_m = re.search(
            r"\b(?:HAVE ANY LIABILITY|be liable)\s+(?:to\s+[^.;]+?\s+)?(?:for\s+)?([^.;]+?)"
            r"(?:\.|;|$|\s+HOWEVER|\s+EVEN IF|\s+AND\s+\()",
            text,
            re.I | re.DOTALL,
        )
        obj = obj_m.group(1).strip() if obj_m else "any lost profits or consequential damages"
        if len(obj) > 200:
            obj = obj[:200].rsplit(" ", 1)[0]
        exc_m = re.search(r"\b(EXCEPT FOR[^.;]+?)(?:,\s*\([a-z]\)|\.|;|$)", text, re.I | re.DOTALL)
        condition = (
            {"text": exc_m.group(1).strip(), "type": "exception"}
            if exc_m
            else {"text": "", "type": "none"}
        )
        return {
            **triplet,
            "subject": {"text": subj, "role": "prohibited_party"},
            "action": {"predicate": "liable", "object": obj},
            "condition": condition,
        }

    if re.search(r"\b(?:shall not be liable|SHALL NOT BE LIABLE|shall not have any liability)\b", text, re.I):
        subj_m = re.search(
            r"\b((?:NEITHER PARTY|Neither party|either party|any party))\b",
            text,
            re.I,
        )
        subj = subj_m.group(1) if subj_m else "neither party"
        obj_m = re.search(
            r"\b(?:liable|LIABILITY)\s+(?:to\s+[^.;]+?\s+)?for\s+([^.;]+?)(?:\.|;|$|\s+REGARDLESS|\s+ARISING)",
            text,
            re.I | re.DOTALL,
        )
        if not obj_m:
            obj_m = re.search(
                r"\bHAVE ANY LIABILITY\s+(?:to\s+)?([^.;]+?)(?:\.|;|$|\s+REGARDLESS|\s+ARISING)",
                text,
                re.I | re.DOTALL,
            )
        obj = obj_m.group(1).strip() if obj_m else "any liability to the other party"
        if len(obj) > 200:
            obj = obj[:200].rsplit(" ", 1)[0]
        exc_m = re.search(r"\b(EXCEPT[^.;]+?)(?:\.|;|$)", text, re.I)
        condition = (
            {"text": exc_m.group(1).strip(), "type": "exception"}
            if exc_m
            else {"text": "", "type": "none"}
        )
        return {
            **triplet,
            "subject": {"text": subj, "role": triplet["subject"].get("role", "prohibited_party")},
            "action": {"predicate": "liable", "object": obj},
            "condition": condition,
        }
    return triplet


def fix_if_then_subject(text: str, triplet: dict) -> dict:
    m = re.search(
        r"\bthen\s+\(?[a-z]\)?\s*([A-Z][A-Za-z0-9&]+)\s+shall\s+(?:have the right to\s+,)?(\w+)\b",
        text,
        re.I,
    )
    if not m:
        return triplet
    subj = m.group(1).strip()
    pred = m.group(2).strip().lower()
    obj = ((triplet.get("action") or {}).get("object") or "").strip()
    if pred == "have":
        obj_m = re.search(r"\bhave the right to[,\s]+(.+?)(?:\.|;|$)", text, re.I)
        if obj_m:
            obj = obj_m.group(1).strip()
    trigger = re.search(r"\bIf\b.+?\bthen\b", text, re.I | re.DOTALL)
    condition = (
        {"text": trigger.group(0).strip().rstrip(","), "type": "trigger"}
        if trigger
        else (triplet.get("condition") or {"text": "", "type": "none"})
    )
    if len(obj) > 200:
        obj = obj[:200].rsplit(" ", 1)[0]
    return {
        **triplet,
        "subject": {**triplet["subject"], "text": subj},
        "action": {"predicate": pred, "object": obj},
        "condition": condition,
    }


def fix_subject_field(text: str, triplet: dict, record: dict) -> dict:
    subj = ((triplet.get("subject") or {}).get("text") or "").strip()
    if not is_polluted_subject(subj):
        return triplet
    candidate = model_consensus_subject(record)
    if not candidate:
        candidate = extract_subject_near_predicate(text, (triplet.get("action") or {}).get("predicate", ""))
    if candidate and not is_polluted_subject(candidate):
        return {**triplet, "subject": {**triplet["subject"], "text": candidate}}
    return triplet


def fix_permission_without_consent(text: str, triplet: dict) -> dict:
    """Rogers may, without consent, assign — permission on lead clause only."""
    lead = re.split(r"\.\s+(?:A change of control|Any )", text, maxsplit=1)[0]
    m = re.search(
        r"\b([A-Z][A-Za-z0-9&]+)\s+may,\s*without consent,\s*(assign|transfer)\b",
        lead,
        re.I,
    )
    if not m:
        return triplet
    subj, pred = m.group(1).strip(), m.group(2).lower()
    obj_m = re.search(rf"\b{pred}\s+(its [^.;]+?)(?:\.|;|$|\s+to:)", lead, re.I)
    obj = obj_m.group(1).strip() if obj_m else "its rights and obligations under this Agreement"
    return {
        **triplet,
        "subject": {"text": subj, "role": "right_holder"},
        "action": {"predicate": pred, "object": obj},
        "condition": {"text": "", "type": "none"},
    }


def finalize_frames(text: str, triplet: dict, record: dict) -> dict:
    t = fix_wrong_be_frame(text, triplet)
    t = fix_govern_subject_trim(text, t)
    t = fix_subject_field(text, t, record)
    if re.search(r"\b(?:Neither [Pp]arty|NEITHER PARTY)\s+may\s+(?:assign|transfer|delegate)\b", text, re.I):
        obj_m = re.search(
            r"\b(?:assign|transfer|delegate)(?:\s+or\s+\w+)*\s+"
            r"((?:any of )?its (?:rights|duties|obligations)[^.;]*)",
            text,
            re.I,
        )
        obj = obj_m.group(1).strip() if obj_m else "its duties or obligations under this Agreement"
        if len(obj) > 200:
            obj = obj[:200].rsplit(" ", 1)[0]
        t = {
            **t,
            "subject": {"text": "Neither Party", "role": t["subject"].get("role", "prohibited_party")},
            "action": {"predicate": "assign", "object": obj},
            "condition": {"text": "", "type": "none"},
        }
    return t


def polish_triplet(text: str, triplet: dict, record: dict) -> tuple[dict, bool]:
    orig = json.dumps(triplet, sort_keys=True)
    t = copy.deepcopy(triplet)

    t = fix_wrong_be_frame(text, t)
    t = fix_if_then_subject(text, t)
    t = fix_govern_subject_trim(text, t)
    t = fix_subject_field(text, t, record)
    t, _ = fix_triplet(text, t)
    t = fix_auto_renew_final(text, t)
    t = finalize_frames(text, t, record)
    t, _ = fix_triplet(text, t)
    t = fix_permission_without_consent(text, t)
    t = finalize_frames(text, t, record)

    subj = ((t.get("subject") or {}).get("text") or "").strip()
    if is_polluted_subject(subj):
        candidate = model_consensus_subject(record)
        if candidate:
            t = {**t, "subject": {**t["subject"], "text": candidate}}

    return t, json.dumps(t, sort_keys=True) != orig


def write_testset(rows: list[dict]) -> None:
    with TEST.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(
                json.dumps(
                    {"clause_id": r["clause_id"], "text": r["text"], "phenomena": r.get("phenomena") or {}},
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> int:
    if not GOLD.is_file():
        print(f"FAIL: missing {GOLD}")
        return 1

    rows = [json.loads(line) for line in GOLD.open(encoding="utf-8") if line.strip()]
    changed: list[str] = []
    for rec in rows:
        triplet, did_change = polish_triplet(rec["text"], rec.get("triplet") or {}, rec)
        if did_change:
            changed.append(rec["clause_id"])
            rec["triplet"] = triplet
            rec["polish_changed"] = True

    with GOLD.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    write_testset(rows)
    print(json.dumps({"records": len(rows), "polished": len(changed)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
