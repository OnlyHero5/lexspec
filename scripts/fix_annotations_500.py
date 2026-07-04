#!/usr/bin/env python3
"""Stage-2 annotation fixes for the curated 500-item gold set.

Input (first existing file wins):
  data/processed/curated_500/stage1_pool.jsonl
  data/processed/gold_triplets_500.jsonl

Output:
  data/processed/curated_500/stage2_annotations.jsonl

Applies rule-based triplet repairs reused from build_gold_500 / curate_gold_500,
with semantic govern/construe frames (laws subject, Agreement object).
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CURATED_DIR = ROOT / "data/processed/curated_500"
INPUT_CANDIDATES = (
    CURATED_DIR / "stage1_pool.jsonl",
    ROOT / "data/processed/gold_triplets_500.jsonl",
)
OUTPUT = CURATED_DIR / "stage2_annotations.jsonl"

GOVERN_RE = re.compile(r"\b(shall be governed|is governed|governed by)\b", re.I)
CONSTRUE_RE = re.compile(
    r"\b(construed|interpreted|enforced)\s+(?:in accordance|exclusively)\b"
    r"|\bshall be construed\b|\bshall be interpreted\b",
    re.I,
)
LAWS_RE = re.compile(
    r"\b(?:the )?(?:substantive )?laws of (?:the )?[^.;,\n(]+",
    re.I,
)
WITHOUT_CONSENT_RE = re.compile(
    r"\bwithout\s+(?:the\s+)?(?:prior\s+)?(?:written\s+)?(?:express\s+)?(?:approval|consent)\b",
    re.I,
)
PROHIBITION_RE = re.compile(
    r"\b(shall not|may not|must not|neither\b|will not|agrees that it will not)\b",
    re.I,
)
TERM_PREDICATES = frozenset({
    "commence", "begin", "start", "end", "expire", "renew", "continue", "terminate",
    "become", "remain", "extend",
})
AGREEMENT_NP_RE = re.compile(r"\b(this agreement|the agreement)\b", re.I)
AUTO_RENEW_RE = re.compile(
    r"\b(?:shall|will)\s+(?:be\s+)?(?:automatically\s+)?renew",
    re.I,
)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
DEFINITION_RE = re.compile(r'^"?([^"]+)"?\s+shall mean\b', re.I)
PROVISO_RE = re.compile(
    r";\s*(?:provided(?:\s*,\s*however\s*,\s*that|\s+that)|however\s*,\s*that)\s+(.+)",
    re.I | re.DOTALL,
)
TERM_DESC_RE = re.compile(
    r"\b(?:term of|shall continue|shall be effective|shall remain in (?:full )?force|"
    r"shall be for a (?:term|period)|effective as of)\b",
    re.I,
)
PASSIVE_PROHIB_RE = re.compile(r"\b(may not|shall not|must not)\s+be\s+\w+", re.I)
NEGATION_MODALITY_RE = re.compile(
    r"\b(?:shall not|may not|must not|will not|neither\b|nor\b|no party\b|"
    r"not be entitled|does not have|do not have|has no right|have no right|"
    r"agrees that it will not|undertakes not to)\b",
    re.I,
)
RIGHT_HOLDER_RE = re.compile(
    r"\b(?:(?:is|are|shall be|will be)\s+entitled|"
    r"has the right|have the right|shall have the right|"
    r"may elect to|may assign|may terminate|may continue|may be terminated by)\b",
    re.I,
)
AFFIRMATIVE_MAY_RE = re.compile(r"(?<!\bnot\s)(?<!\bno\s)\bmay\b(?! not\b)", re.I)
OBLIGOR_RE = re.compile(r"\b(?:shall|must|agrees to|undertakes to)\b", re.I)
THEME_LIKE_SUBJ_RE = re.compile(r"^(?:Such |This |The |Any )", re.I)


def resolve_input_path() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    names = ", ".join(str(p.relative_to(ROOT)) for p in INPUT_CANDIDATES)
    raise FileNotFoundError(f"No input file found. Tried: {names}")


def _agreement_np(text: str) -> str:
    return "This Agreement" if "this agreement" in text.lower() else "the Agreement"


def _extract_laws(text: str) -> str:
    m = LAWS_RE.search(text)
    if m:
        return m.group(0).strip()
    m = re.search(
        r"governed by (?:and construed in accordance with )?"
        r"((?:the )?(?:substantive )?(?:laws of (?:the )?[^.;,\n]+|law of [^.;,\n]+|[^.;,\n]+ law))",
        text,
        re.I,
    )
    if m:
        laws = m.group(1).strip()
        if laws.lower().startswith("law of "):
            laws = f"the laws of {laws[7:]}"
        elif not laws.lower().startswith("the "):
            laws = f"the laws of {laws}" if "law" not in laws.lower() else f"the {laws}"
        return laws
    m = re.search(r"\b(?:the )?law of ([^.;,\n]+)", text, re.I)
    if m:
        return f"the laws of {m.group(1).strip()}"
    return "the applicable laws"


def _govern_exception(text: str) -> dict:
    exc = re.search(
        r"(without regard to (?:its )?conflicts of law[^.;,\n]*|"
        r"without regard to[^.;,\n]+|except as[^.;,\n]+|notwithstanding[^.;,\n]+)",
        text,
        re.I,
    )
    if exc:
        return {"text": exc.group(1).strip(), "type": "exception"}
    return {"text": "", "type": "none"}


def fix_govern_semantic(text: str, triplet: dict) -> dict:
    """Laws govern Agreement (matches build_gold_500.fix_govern)."""
    if not GOVERN_RE.search(text):
        return triplet
    # Non-jurisdiction senses: "governed by the terms", "agreement governing the VOD..."
    if re.search(r"\bgoverned by the terms\b", text, re.I):
        return _fix_terms_govern_clause(text, triplet)
    if re.search(r"\bagreement governing\b", text, re.I) and not LAWS_RE.search(text):
        return triplet
    if not (
        LAWS_RE.search(text)
        or re.search(r"\bgoverned by (?:the )?(?:substantive )?(?:laws|law of)\b", text, re.I)
    ):
        return triplet
    laws = _extract_laws(text)
    obj = _agreement_np(text)
    exc = _govern_exception(text)
    return {
        **triplet,
        "subject": {"text": laws, "role": "other"},
        "action": {"predicate": "govern", "object": obj},
        "condition": exc if exc.get("text") else {"text": "", "type": "none"},
    }


def _fix_terms_govern_clause(text: str, triplet: dict) -> dict:
    """'governed by the terms of this Agreement' → termination-right proviso frame."""
    term_m = re.search(
        r"\b(?:each of )?Licensor and Rogers\b|\b(?:each of )?([^,]+ and [^,]+)\b",
        text,
        re.I,
    )
    subj = term_m.group(0).strip() if term_m else "the parties"
    notice_m = re.search(
        r"on\s+sixty\s*\(\s*60\s*\)\s*days['']?\s+prior written notice",
        text,
        re.I,
    )
    trigger_m = re.search(
        r"\bif,\s*at the expiry of this Agreement[^,]+?(?=,\s*such continued|\.)",
        text,
        re.I | re.DOTALL,
    )
    cond_parts = []
    if trigger_m:
        cond_parts.append(trigger_m.group(0).strip())
    if notice_m:
        cond_parts.append(notice_m.group(0).strip())
    condition = (
        {"text": ", ".join(cond_parts), "type": "trigger"}
        if cond_parts
        else {"text": "", "type": "none"}
    )
    return {
        **triplet,
        "subject": {"text": subj, "role": "right_holder"},
        "action": {"predicate": "terminate", "object": "this Agreement"},
        "condition": condition,
    }


def fix_interpret_construe(text: str, triplet: dict) -> dict:
    """Laws construe/interpret Agreement (matches build_gold_500.fix_interpret_construe)."""
    if GOVERN_RE.search(text):
        return triplet
    if not CONSTRUE_RE.search(text):
        return triplet
    laws = _extract_laws(text)
    if laws == "the applicable laws" and not LAWS_RE.search(text):
        m = re.search(r"\b(?:in accordance with|exclusively in accordance with)\s+(.+?)(?:\.|;|$)", text, re.I)
        if m:
            laws = m.group(1).strip().rstrip(".")
    obj = _agreement_np(text)
    pred = "construe" if re.search(r"\bconstru", text, re.I) else "interpret"
    exc = _govern_exception(text)
    return {
        **triplet,
        "subject": {"text": laws, "role": "other"},
        "action": {"predicate": pred, "object": obj},
        "condition": exc if exc.get("text") else {"text": "", "type": "none"},
    }


def fix_definition_triplet(text: str, triplet: dict) -> dict:
    """\"X\" shall mean → subject=term, predicate=mean, role=other."""
    m = DEFINITION_RE.search(text.strip())
    if not m:
        m = re.search(r'\b"([^"]+)"\s+shall mean\b', text, re.I)
    if not m:
        return triplet
    term = m.group(1).strip()
    if not term.startswith('"'):
        term = f'"{term}"'
    obj_m = re.search(r"\bshall mean\s+(?:that\s+)?(.+)", text, re.I | re.DOTALL)
    obj = obj_m.group(1).strip().rstrip(".") if obj_m else ""
    return {
        **triplet,
        "subject": {"text": term, "role": "other"},
        "action": {"predicate": "mean", "object": obj},
        "condition": {"text": "", "type": "none"},
    }


def _frame_from_proviso(prov: str) -> dict | None:
    """Extract deontic frame from a provided/however proviso clause."""
    elect = re.search(
        r"^(.+?)\s+may\s+(?:elect to\s+)?(\w+)\s+(.+)$",
        prov.strip().rstrip("."),
        re.I | re.DOTALL,
    )
    if elect:
        subj = elect.group(1).strip()
        pred = elect.group(2).strip().lower()
        rest = elect.group(3).strip()
        obj_m = re.match(r"(this Agreement|the Agreement|[^,;]+)", rest, re.I)
        obj = obj_m.group(1).strip() if obj_m else rest.split(",")[0].strip()
        cond_text = rest[len(obj):].strip().lstrip(",").strip() if obj_m else ""
        role = "right_holder"
        return {
            "subject": {"text": subj, "role": role},
            "action": {"predicate": pred, "object": obj},
            "condition": (
                {"text": cond_text, "type": "trigger"}
                if cond_text
                else {"text": "", "type": "none"}
            ),
        }

    shall_m = re.search(
        r"^(.+?)\s+shall\s+(not\s+)?(\w+)\s*(.*)$",
        prov.strip().rstrip("."),
        re.I | re.DOTALL,
    )
    if shall_m:
        subj = shall_m.group(1).strip()
        neg = bool(shall_m.group(2))
        pred = shall_m.group(3).strip().lower()
        rest = shall_m.group(4).strip()
        obj = rest.split(",")[0].strip() if rest else ""
        role = "prohibited_party" if neg else "obligor"
        if re.search(r"\bmay\b", prov, re.I) and not neg:
            role = "right_holder"
        return {
            "subject": {"text": subj, "role": role},
            "action": {"predicate": pred, "object": obj},
            "condition": {"text": "", "type": "none"},
        }
    return None


def _first_deontic_shall_frame(lead: str) -> dict | None:
    """First deontic shall in the lead sentence (C-00089-type duty continue)."""
    duty_continue = re.search(
        r"((?:[\w']+(?:'s)?\s+)?(?:duty|obligation)(?:\s+not to [^.;]+?)?)\s+shall continue\b",
        lead,
        re.I,
    )
    if duty_continue:
        subj = duty_continue.group(1).strip()
        cond_m = re.search(
            r"\b(for a period of[^.;]+|following [^.;]+|until [^.;]+|"
            r"for one year[^.;]*|for \d+[^.;]+)",
            lead,
            re.I,
        )
        condition = (
            {"text": cond_m.group(0).strip(), "type": "temporal"}
            if cond_m
            else {"text": "", "type": "none"}
        )
        return {
            "subject": {"text": subj, "role": "other"},
            "action": {"predicate": "continue", "object": ""},
            "condition": condition,
        }

    shall_m = re.search(
        r"^(.+?)\s+shall\s+(not\s+)?(\w+)\s*(.*)$",
        lead.strip().rstrip("."),
        re.I | re.DOTALL,
    )
    if shall_m and shall_m.group(1).strip().lower() not in ("it", "they"):
        subj = shall_m.group(1).strip()
        neg = bool(shall_m.group(2))
        pred = shall_m.group(3).strip().lower()
        rest = shall_m.group(4).strip()
        obj = rest.split(",")[0].strip() if rest and pred not in TERM_PREDICATES else ""
        if pred in TERM_PREDICATES:
            obj = ""
        role = "prohibited_party" if neg else ("other" if pred in TERM_PREDICATES else "obligor")
        return {
            "subject": {"text": subj, "role": role},
            "action": {"predicate": pred, "object": obj},
            "condition": {"text": "", "type": "none"},
        }
    return None


def fix_multi_shall_primary(text: str, triplet: dict) -> dict:
    """When multiple shall-clauses, pick the primary deontic frame."""
    if DEFINITION_RE.search(text.strip()) or GOVERN_RE.search(text):
        return triplet

    sents = [s.strip() for s in SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    lead = sents[0] if sents else text.strip()
    shall_count = len(re.findall(r"\bshall\b", text, re.I))

    # (c) auto-renew + notice: prefer renew from first sentence when auto-renew present
    if AUTO_RENEW_RE.search(lead):
        subj_m = re.match(r"^(.+?)\s+(?:shall|will)\s+(?:be\s+)?(?:automatically\s+)?renew", lead, re.I)
        subj = subj_m.group(1).strip() if subj_m else _agreement_np(text)
        cond_m = re.search(
            r"\b(?:for a period of[^.;]+|upon [^.;]+|and upon [^.;]+)",
            lead,
            re.I,
        )
        condition = (
            {"text": cond_m.group(0).strip(), "type": "trigger" if "upon" in cond_m.group(0).lower() else "temporal"}
            if cond_m
            else (triplet.get("condition") or {"text": "", "type": "none"})
        )
        return {
            **triplet,
            "subject": {"text": subj, "role": "other"},
            "action": {"predicate": "renew", "object": ""},
            "condition": condition,
        }

    # (a) provided/however proviso over term description
    prov_m = PROVISO_RE.search(text)
    if prov_m and TERM_DESC_RE.search(text[: prov_m.start()]):
        frame = _frame_from_proviso(prov_m.group(1))
        if frame:
            return {**triplet, **frame}

    # (b) first deontic shall in lead sentence when multiple shall-clauses
    if shall_count >= 2 or len(sents) >= 2:
        frame = _first_deontic_shall_frame(lead)
        if frame:
            pred = ((triplet.get("action") or {}).get("predicate") or "").strip().lower()
            if pred != frame["action"]["predicate"]:
                return {**triplet, **frame}

    return triplet


def fix_renew_subj_obj(text: str, triplet: dict) -> dict:
    """Auto-renew intransitive: empty object when subject repeats agreement (C-00206 pattern)."""
    pred = (triplet.get("action") or {}).get("predicate", "")
    if pred != "renew":
        return triplet
    subj = ((triplet.get("subject") or {}).get("text") or "").strip()
    obj = ((triplet.get("action") or {}).get("object") or "").strip()
    if not subj or not obj:
        return triplet
    if subj.lower() != obj.lower():
        return triplet
    if not (AUTO_RENEW_RE.search(text) or AGREEMENT_NP_RE.search(subj)):
        return triplet
    triplet = copy.deepcopy(triplet)
    triplet["action"] = {**triplet["action"], "object": ""}
    return triplet


def fix_role_modality(text: str, triplet: dict) -> dict:
    """Align subject role with deontic modality (matches validate_quality_95 heuristics)."""
    subj = triplet.get("subject") or {}
    role = (subj.get("role") or "").strip()
    if role == "other":
        return triplet

    subj_text = (subj.get("text") or "").strip()
    if not subj_text:
        return triplet

    pred = ((triplet.get("action") or {}).get("predicate") or "").strip().lower()
    if pred in TERM_PREDICATES and re.search(
        r"\b(term|agreement|license|this agreement)\b", subj_text, re.I
    ):
        return triplet
    if pred in ("govern", "construe", "interpret", "mean"):
        return triplet

    prohibition = re.compile(
        r"\b(?:shall\s+not|may\s+not|must\s+not|will\s+not|"
        r"neither\b[^.;]{0,160}?\b(?:shall|may)\b|"
        r"neither\s+party\s+may\b|"
        r"no\s+party\s+shall|"
        r"in\s+no\s+event|"
        r"nothing\s+(?:in\s+)?(?:herein|this\s+Agreement))\b",
        re.I | re.DOTALL,
    )
    permission = re.compile(
        r"\b(?:may\b(?!\s+not\b)|is\s+entitled\s+to|has\s+the\s+right\s+to|shall\s+have\s+the\s+right)\b",
        re.I,
    )
    obligation = re.compile(
        r"\b(?:shall\b(?!\s+not\b)|must\b(?!\s+not\b)|agrees?\s+to|undertakes?\s+to)\b",
        re.I,
    )

    if prohibition.search(text):
        correct = "prohibited_party"
    elif permission.search(text):
        correct = "right_holder"
    elif obligation.search(text):
        correct = "obligor"
    else:
        return triplet

    allowed = {
        "obligor": {"obligor", "indemnifying_party", "other"},
        "right_holder": {"right_holder", "other"},
        "prohibited_party": {"prohibited_party", "other"},
    }.get(correct, {correct, "other"})

    if role not in allowed:
        return {**triplet, "subject": {**subj, "role": correct}}
    return triplet


def _extract_passive_agent(text: str) -> str | None:
    m = re.search(
        r"\bby\s+([^.;,\n]+?)(?:\.|,|;|$|\s+without|\s+except|\s+which\b)",
        text,
        re.I,
    )
    if m:
        return m.group(1).strip()
    return None


def fix_inferred_subject(text: str, triplet: dict) -> tuple[dict, str | None]:
    """Infer prohibited-party agent for passive prohibitions without explicit subject."""
    subj = triplet.get("subject") or {}
    subj_text = (subj.get("text") or "").strip()
    action = triplet.get("action") or {}
    obj = (action.get("object") or "").strip()
    note: str | None = None

    if not PASSIVE_PROHIB_RE.search(text):
        return triplet, note

    theme_m = re.match(
        r"^([^,.]+?)\s+(?:may not|shall not|must not)\s+be\s+(\w+)",
        text.strip(),
        re.I,
    )
    if not theme_m:
        return triplet, note

    theme = theme_m.group(1).strip()
    verb = theme_m.group(2).strip().lower()

    wrong_subj = (
        subj_text.lower() == theme.lower()
        or (THEME_LIKE_SUBJ_RE.match(subj_text) and subj_text.lower().startswith("such "))
        or (subj_text and subj_text.lower() not in ("any party", "either party", "neither party")
            and not _extract_passive_agent(text)
            and THEME_LIKE_SUBJ_RE.match(subj_text))
    )

    if not wrong_subj and subj_text:
        return triplet, note

    agent = _extract_passive_agent(text)
    new_subj = agent if agent else "any party"
    if not agent:
        note = "inferred subject any party (passive prohibition, no explicit agent)"

    out = {
        **triplet,
        "subject": {"text": new_subj, "role": "prohibited_party"},
        "action": {
            **action,
            "predicate": verb,
            "object": theme if not obj or obj.lower() == new_subj.lower() else obj,
        },
        "condition": triplet.get("condition") or {"text": "", "type": "none"},
    }
    return out, note


def fix_liable(triplet: dict) -> dict:
    pred = (triplet.get("action") or {}).get("predicate", "")
    obj = (triplet.get("action") or {}).get("object", "")
    if pred in ("be liable", "be Liable"):
        return {**triplet, "action": {**triplet["action"], "predicate": "liable"}}
    if pred == "be" and re.search(r"\bliable\b", obj, re.I):
        return {**triplet, "action": {**triplet["action"], "predicate": "liable"}}
    return triplet


def fix_term_role(text: str, triplet: dict) -> dict:
    pred = (triplet.get("action") or {}).get("predicate", "")
    subj = (triplet.get("subject") or {}).get("text", "")
    if pred in TERM_PREDICATES and re.search(
        r"\b(term|agreement|license|this agreement)\b", subj, re.I
    ):
        return {**triplet, "subject": {**triplet["subject"], "role": "other"}}
    return triplet


def fix_effective(text: str, triplet: dict) -> dict:
    pred = (triplet.get("action") or {}).get("predicate", "")
    if pred == "effective" and re.search(r"\b(shall become|is) effective\b", text, re.I):
        subj = (triplet.get("subject") or {}).get("text") or _agreement_np(text)
        return {
            **triplet,
            "subject": {"text": subj, "role": "other"},
            "action": {"predicate": "be", "object": "effective"},
        }
    return triplet


def fix_section_role(triplet: dict) -> dict:
    subj = (triplet.get("subject") or {}).get("text", "")
    if re.match(r"^Section\s+\d", subj, re.I):
        return {**triplet, "subject": {**triplet["subject"], "role": "other"}}
    return triplet


def strip_without_consent(text: str, triplet: dict) -> dict:
    """Inline without-consent on prohibitions is not a subordinate condition."""
    cond = triplet.get("condition") or {}
    ctext = (cond.get("text") or "").strip()
    ctype = (cond.get("type") or "none").strip().lower()
    if not ctext or not WITHOUT_CONSENT_RE.search(text):
        return triplet
    if not PROHIBITION_RE.search(text) or not WITHOUT_CONSENT_RE.search(ctext):
        return triplet
    except_m = re.search(r"\b(except that|except as)\b.+", ctext, re.I | re.DOTALL)
    if except_m:
        return {
            **triplet,
            "condition": {"text": except_m.group(0).strip().rstrip(","), "type": "exception"},
        }
    if ctype == "trigger" or ctype == "exception":
        return {**triplet, "condition": {"text": "", "type": "none"}}
    return triplet


def normalize_condition(triplet: dict) -> dict:
    cond = triplet.get("condition") or {}
    ctext = (cond.get("text") or "").strip()
    ctype = cond.get("type", "none")
    if not ctext:
        return {**triplet, "condition": {"text": "", "type": "none"}}
    if ctype == "none":
        return {**triplet, "condition": {"text": ctext, "type": "temporal"}}
    return triplet


def trim_object(triplet: dict, max_len: int = 200) -> dict:
    obj = (triplet.get("action") or {}).get("object", "")
    if len(obj) <= max_len:
        return triplet
    cut = obj[:max_len].rsplit(" ", 1)[0]
    return {**triplet, "action": {**triplet["action"], "object": cut}}


def fix_triplet(text: str, triplet: dict) -> tuple[dict, list[str]]:
    notes: list[str] = []
    t = copy.deepcopy(triplet)
    t = fix_definition_triplet(text, t)
    t = fix_govern_semantic(text, t)
    t = fix_interpret_construe(text, t)
    t = fix_multi_shall_primary(text, t)
    t = fix_renew_subj_obj(text, t)
    t, inferred_note = fix_inferred_subject(text, t)
    if inferred_note:
        notes.append(inferred_note)
    t = fix_role_modality(text, t)
    t = fix_liable(t)
    t = fix_term_role(text, t)
    t = fix_effective(text, t)
    t = fix_section_role(t)
    t = strip_without_consent(text, t)
    t = normalize_condition(t)
    t = trim_object(t)
    return t, notes


def fix_record(record: dict) -> tuple[dict, bool]:
    text = record["text"]
    orig = json.dumps(record.get("triplet") or {}, sort_keys=True)
    triplet, extra_notes = fix_triplet(text, record.get("triplet") or {})
    changed = json.dumps(triplet, sort_keys=True) != orig
    out = copy.deepcopy(record)
    out["triplet"] = triplet
    if changed or extra_notes:
        out["annotation_fix_changed"] = True
        note = (out.get("annotation_fix_note") or out.get("curation_note") or "").strip()
        parts = [note] if note else []
        if "stage2 rule fix" not in note:
            parts.append("stage2 rule fix")
        parts.extend(n for n in extra_notes if n and n not in note)
        out["annotation_fix_note"] = "; ".join(p for p in parts if p)
    return out, changed or bool(extra_notes)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    input_path = args.input or resolve_input_path()
    output_path = args.output or OUTPUT
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit(f"Input file is empty: {input_path}")

    changed_ids: list[str] = []
    out_rows: list[dict] = []
    for rec in records:
        fixed, changed = fix_record(rec)
        out_rows.append(fixed)
        if changed:
            changed_ids.append(fixed["clause_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(input_path.relative_to(ROOT)),
        "output": str(output_path.relative_to(ROOT)),
        "records": len(out_rows),
        "modified": len(changed_ids),
        "modified_ids": changed_ids,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
