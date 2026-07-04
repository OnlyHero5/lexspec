#!/usr/bin/env python3
"""Curate fix_batch_bcd_4.jsonl to LexSpec gold quality."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

INPUT = Path("data/processed/curated_500/fix_batch_bcd_4.jsonl")
OUTPUT = Path("data/processed/curated_500/fixed_batch_bcd_4.jsonl")

WITHOUT_CONSENT_RE = re.compile(
    r"\bwithout\s+(?:the\s+)?(?:prior\s+)?(?:written\s+)?(?:express\s+)?consent\b",
    re.I,
)
DEONTIC_NEG_RE = re.compile(
    r"\b(?:shall\s+not|may\s+not|must\s+not|neither\b|in\s+no\s+event)\b",
    re.I,
)

# Per-clause gold corrections: triplet/phenomena patches + fix_notes
MANUAL: dict[str, dict] = {
    "C-00434": {
        "phenomena": {"conditional": True},
        "note": "set conditional=true to align with in-the-event trigger condition",
    },
    "C-00436": {
        "phenomena": {"conditional": True},
        "note": "set conditional=true to align with notwithstanding exception condition",
    },
    "C-00443": {
        "phenomena": {"conditional": False, "is_definition": True},
        "note": "definition clause; no subordinate condition; is_definition=true",
    },
    "C-00450": {
        "phenomena": {"conditional": False, "negation": True},
        "note": "no subordinate condition; negation for not-less-than delivery spec",
    },
    "C-00452": {
        "action": {
            "object": (
                "any and all rights, title and interest in any Intellectual Property Rights "
                "resulting from any development made by Dexcel related to the Product"
            )
        },
        "note": "minimal complete Joint IP ownership object within length limit",
    },
    "C-00454": {
        "phenomena": {"conditional": False, "negation": True},
        "note": "parenthetical manner limits are inline; negation for not-more-often cap",
    },
    "C-00457": {
        "action": {
            "object": (
                "written notification of any shortfalls in shipment quantity, and (a) any "
                "out-of-specification temperature excursions based on the downloaded data logger "
                "information following compliance with the provisions of the Quality Agreement, "
                "and/or (b) any failure of the Product to meet the Specifications which are "
                "apparent upon visual inspection and/or identification testing of the Product "
                "delivered to it by Dexcel"
            )
        },
        "phenomena": {"negation": True},
        "note": "complete truncated notification object; negation for shall-not-remove clause",
    },
    "C-00468": {
        "phenomena": {"conditional": False},
        "note": "coordinated transfer duties have no subordinate trigger/temporal/exception",
    },
    "C-00472": {
        "action": {
            "object": (
                "the amount that is one and one half (1½) times the aggregate amounts paid or "
                "payable pursuant to this Agreement in the preceding twelve (12) month period "
                "preceding the loss date"
            )
        },
        "phenomena": {"negation": True},
        "note": "complete truncated liability-cap object; negation for neither Party",
    },
    "C-00474": {
        "phenomena": {"conditional": False, "negation": True},
        "note": "insurance-structure clause has no subordinate condition; negation in limits",
    },
    "C-00482": {
        "phenomena": {"passive": False, "negation": True},
        "note": "active may-not prohibition; without-consent is inline frame",
    },
    "C-00483": {
        "phenomena": {"conditional": False, "negation": True},
        "note": "shall-not-sell prohibition; no subordinate condition on main action",
    },
    "C-00486": {
        "note": "confirmed auto-renewal gold; empty renew object; unless exception retained",
    },
    "C-00490": {
        "phenomena": {"conditional": False, "negation": True},
        "note": "right-of-first-refusal has no subordinate condition; negation in relative clause",
    },
    "C-00493": {
        "phenomena": {"conditional": False},
        "note": "entitlement to sub-distributor agreements has no subordinate condition",
    },
    "C-00494": {
        "phenomena": {"conditional": True},
        "note": "set conditional=true to align with during-term temporal condition",
    },
    "C-00496": {
        "phenomena": {"conditional": True},
        "note": "set conditional=true to align with except-with-regard-to exception",
    },
    "C-00503": {
        "phenomena": {"conditional": True, "negation": True},
        "note": "temporal termination window; negation for non-renewal notice",
    },
    "C-00507": {
        "condition": {"text": "", "type": "none"},
        "phenomena": {"conditional": False, "negation": True},
        "note": "strip inline without-consent; negation for neither Party shall",
    },
    "C-00512": {
        "note": "confirmed term-commencement gold; unless exception and role other retained",
    },
    "C-00513": {
        "action": {"object": ""},
        "phenomena": {"negation": True},
        "note": "auto-renew with unless notice; empty renew object; negation in unless",
    },
    "C-00519": {
        "phenomena": {"conditional": True},
        "note": "set conditional=true to align with on-a-monthly-basis temporal",
    },
    "C-00524": {
        "phenomena": {"conditional": True},
        "note": "set conditional=true to align with term-and-notice temporal condition",
    },
    "C-00529": {
        "phenomena": {"conditional": True},
        "note": "set conditional=true to align with as-from effective-date temporal",
    },
    "C-00530": {
        "action": {"object": ""},
        "phenomena": {"conditional": True},
        "note": "tacit renewal declarative; empty object; align conditional with every-year",
    },
    "C-00535": {
        "phenomena": {"conditional": True},
        "note": "set conditional=true to align with on-the-Effective-Date temporal",
    },
    "C-00539": {
        "note": "confirmed termination-right gold; at-any-time-on-notice is inline manner",
    },
    "C-00542": {
        "phenomena": {"conditional": True, "negation": True},
        "note": "during-term temporal; negation for no-more-than frequency cap",
    },
    "C-00545": {
        "action": {
            "object": (
                "a paid-up, royalty-free, non-exclusive license to Customer's "
                "Confidential Information and the Customer Technology"
            )
        },
        "phenomena": {"conditional": True},
        "note": "trim license object; align conditional with During the Term",
    },
    "C-00562": {
        "phenomena": {"negation": True},
        "note": "negation for may-not-assign prohibition; without-consent inline",
    },
    "C-00569": {
        "action": {
            "object": (
                "commercial general liability insurance, including products liability insurance, "
                "with minimum \"A-\" AM Best rated insurance carriers, in each case with limits "
                "of not less than five million dollars ($5,000,000) per occurrence and in the aggregate"
            )
        },
        "phenomena": {"negation": True},
        "note": "complete truncated insurance object; negation for not-less-than limits",
    },
    "C-00585": {
        "action": {
            "object": (
                "(1) THE OBLIGATIONS OF EITHER PARTY TO INDEMNIFY THE OTHER PARTY FROM AND "
                "AGAINST THIRD PARTY CLAIMS UNDER SECTION 11.1 OR 11.2, AS APPLICABLE, OR "
                "(2) DAMAGES AVAILABLE FOR A PARTY'S BREACH OF THE CONFIDENTIALITY AND NON-USE "
                "OBLIGATIONS IN ARTICLE 9"
            )
        },
        "phenomena": {"negation": True},
        "note": "complete truncated limit object; negation for SHALL NOT LIMIT",
    },
    "C-00589": {
        "phenomena": {"conditional": True, "negation": True},
        "note": "during-term temporal; negation for shall not impair",
    },
    "C-00590": {
        "phenomena": {"conditional": True},
        "note": "set conditional=true to align with commence-on-date temporal",
    },
    "C-00591": {
        "phenomena": {"conditional": True},
        "note": "set conditional=true to align with prior-to-term-end temporal",
    },
    "C-00595": {
        "action": {"object": ""},
        "phenomena": {"negation": True},
        "note": "auto-renew declarative; empty object; negation for unless non-renewal",
    },
    "C-00616": {
        "phenomena": {"conditional": True},
        "note": "set conditional=true to align with on-the-Effective-Date temporal",
    },
    "C-00617": {
        "action": {"object": ""},
        "phenomena": {"conditional": True},
        "note": "auto-renew declarative; empty object; align conditional with Thereafter",
    },
    "C-00628": {
        "action": {
            "object": (
                "an agreement (a \"Competitive Transaction\") with any other Person related to "
                "the license, sub-license, sale, resale or provide service, solutions, goods or "
                "products, that are substantially similar to or competitive with the Ehave "
                "Companion Solution"
            )
        },
        "phenomena": {"negation": True},
        "note": "complete truncated competitive-transaction object; negation for shall not enter",
    },
    "C-00639": {
        "action": {
            "object": (
                "the aggregate of all amounts paid under this Agreement and amounts that have "
                "accrued but not yet been paid in the twelve (12) months preceding the event "
                "giving rise to the claim"
            )
        },
        "phenomena": {"negation": True},
        "note": "complete truncated liability-cap object; negation for in no event shall exceed",
    },
    "C-00642": {
        "phenomena": {"conditional": True, "negation": True},
        "note": "provided-however trigger; negation for are-not-separated",
    },
    "C-00644": {
        "phenomena": {"conditional": True},
        "note": "set conditional=true to align with on-the-Effective-Date temporal",
    },
    "C-00645": {
        "action": {"object": ""},
        "phenomena": {"negation": True},
        "note": "auto-renew declarative; empty object; negation for unless terminate notice",
    },
    "C-00647": {
        "phenomena": {"conditional": True, "negation": True},
        "note": "absent-consent exception on resell prohibition; negation for in no event may",
    },
    "C-00648": {
        "phenomena": {"negation": True},
        "note": "negation for may-not-assign; without-consent is inline frame",
    },
    "C-00653": {
        "action": {
            "object": (
                "the (i) use of the Mobility Management Services; (ii) unlimited iPass network "
                "access; and (iii) iPass Hosted Authentication Service"
            )
        },
        "note": "complete fees-include object as minimal enumerated NP",
    },
    "C-00665": {
        "phenomena": {"conditional": True},
        "note": "set conditional=true to align with during-the-Term temporal",
    },
}


def trim_object(triplet: dict, max_len: int = 175) -> dict:
    obj = (triplet.get("action") or {}).get("object", "") or ""
    if len(obj) <= max_len:
        return triplet
    cut = obj[:max_len].rsplit(" ", 1)[0]
    return {**triplet, "action": {**triplet["action"], "object": cut}}


def strip_without_consent(text: str, triplet: dict) -> dict:
    cond = triplet.get("condition") or {}
    ctext = (cond.get("text") or "").strip()
    if not ctext or not WITHOUT_CONSENT_RE.search(text):
        return triplet
    if re.search(r"\bshall not\b|\bmay not\b|\bneither\b.*\bshall\b", text, re.I):
        if WITHOUT_CONSENT_RE.search(ctext):
            return {**triplet, "condition": {"text": "", "type": "none"}}
    return triplet


def sync_conditional(phenomena: dict, triplet: dict) -> dict:
    phen = copy.deepcopy(phenomena)
    has_cond = bool((triplet.get("condition") or {}).get("text", "").strip())
    phen["conditional"] = has_cond
    return phen


def apply_patch(triplet: dict, phenomena: dict, patch: dict) -> tuple[dict, dict, str]:
    t = copy.deepcopy(triplet)
    p = copy.deepcopy(phenomena)
    note = patch.get("note", "")

    for section in ("subject", "action", "condition"):
        if section in patch:
            t[section] = {**t.get(section, {}), **patch[section]}

    if "phenomena" in patch:
        p = {**p, **patch["phenomena"]}

    return t, p, note


def validate(triplet: dict) -> list[str]:
    issues = []
    pred = (triplet.get("action") or {}).get("predicate", "")
    if " " in pred:
        issues.append("multiword_pred")
    c = triplet.get("condition") or {}
    ctext, ctype = (c.get("text") or "").strip(), c.get("type", "none")
    if ctext and ctype == "none":
        issues.append("cond_mismatch")
    if not ctext and ctype != "none":
        issues.append("cond_empty")
    obj = (triplet.get("action") or {}).get("object", "") or ""
    if len(obj) > 185:
        issues.append("long_obj")
    return issues


def main() -> None:
    records = [
        json.loads(line)
        for line in INPUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out_rows = []
    issue_counts: dict[str, int] = {}
    changed = 0

    for rec in records:
        cid = rec["clause_id"]
        orig_t = copy.deepcopy(rec["triplet"])
        orig_p = copy.deepcopy(rec["phenomena"])

        triplet = copy.deepcopy(rec["triplet"])
        phenomena = copy.deepcopy(rec["phenomena"])
        note = ""

        triplet = strip_without_consent(rec["text"], triplet)

        if cid in MANUAL:
            triplet, phenomena, note = apply_patch(triplet, phenomena, MANUAL[cid])

        triplet = trim_object(triplet)

        # Final conditional sync from triplet condition field
        phenomena = sync_conditional(phenomena, triplet)

        # Ensure deontic negation flagged when text has prohibition modals
        if DEONTIC_NEG_RE.search(rec["text"]):
            phenomena["negation"] = True

        if json.dumps(triplet, sort_keys=True) != json.dumps(orig_t, sort_keys=True) or (
            phenomena != orig_p
        ):
            changed += 1

        for code in rec.get("issues", []):
            issue_counts[code] = issue_counts.get(code, 0) + 1

        val_issues = validate(triplet)
        out_rows.append(
            {
                "clause_id": cid,
                "triplet": triplet,
                "phenomena": phenomena,
                "fix_notes": note,
                "_validation": val_issues,
            }
        )

    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in out_rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    val_fail = [(r["clause_id"], r["_validation"]) for r in out_rows if r["_validation"]]
    print(json.dumps({
        "total": len(out_rows),
        "changed": changed,
        "issues_addressed": issue_counts,
        "validation_failures": len(val_fail),
        "validation_samples": val_fail[:5],
        "output": str(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
