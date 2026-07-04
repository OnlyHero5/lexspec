#!/usr/bin/env python3
"""Generate fixed_batch_bcd_5.jsonl from fix_batch_bcd_5.jsonl."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data/processed/curated_500/fix_batch_bcd_5.jsonl"
OUTPUT = ROOT / "data/processed/curated_500/fixed_batch_bcd_5.jsonl"

WITHOUT_CONSENT_RE = re.compile(
    r"\bwithout\s+(?:the\s+)?(?:prior\s+)?(?:written\s+)?(?:express\s+)?consent\b",
    re.I,
)
SUBJECT_TO_PROCEDURAL = re.compile(
    r"^Subject to (?:the terms(?: and conditions)? of this Agreement|Section\s+\d)",
    re.I,
)
DEONTIC_NEG_RE = re.compile(
    r"\b(shall not|may not|must not|neither\b|nor\b.*\bshall\b|"
    r"not to exceed|no longer|have no right|have the right or power to)\b",
    re.I,
)
GOVERN_RE = re.compile(r"\b(shall be governed|governed by)\b", re.I)
LAWS_RE = re.compile(
    r"\b(?:the )?(?:substantive )?laws of (?:the )?[^.;,\n]+",
    re.I,
)


def sync_conditional(phen: dict, triplet: dict) -> None:
    has_cond = bool((triplet.get("condition") or {}).get("text", "").strip())
    phen["conditional"] = has_cond


def strip_without_consent(text: str, triplet: dict) -> dict:
    t = copy.deepcopy(triplet)
    if re.search(r"\b(shall not|may not|neither\b)\b", text, re.I) and WITHOUT_CONSENT_RE.search(text):
        cond = (t.get("condition") or {}).get("text", "")
        if cond and WITHOUT_CONSENT_RE.search(cond):
            t["condition"] = {"text": "", "type": "none"}
    return t


def strip_procedural_subject_to(triplet: dict) -> dict:
    t = copy.deepcopy(triplet)
    ctext = ((t.get("condition") or {}).get("text") or "").strip()
    if ctext and SUBJECT_TO_PROCEDURAL.match(ctext):
        t["condition"] = {"text": "", "type": "none"}
    return t


def fix_govern_clause(text: str, triplet: dict) -> dict:
    if not GOVERN_RE.search(text):
        return triplet
    m = LAWS_RE.search(text)
    laws = m.group(0).strip() if m else "the applicable laws"
    obj = "this Agreement" if "this agreement" in text.lower() else "the Agreement"
    return {
        "subject": {"text": laws, "role": "other"},
        "action": {"predicate": "govern", "object": obj},
        "condition": {"text": "", "type": "none"},
    }


def fix_determine_laws(text: str, triplet: dict) -> dict:
    if "determined in accordance with the laws" not in text.lower():
        return triplet
    m = LAWS_RE.search(text)
    laws = m.group(0).strip() if m else "the applicable laws"
    obj_m = re.match(
        r"(Any controversy[^,]+(?:,[^,]+)*?)\s+shall be determined",
        text,
        re.I,
    )
    obj = obj_m.group(1).strip() if obj_m else "Any controversy between the parties"
    return {
        "subject": {"text": laws, "role": "other"},
        "action": {"predicate": "govern", "object": obj},
        "condition": {"text": "", "type": "none"},
    }


MANUAL: dict[str, dict] = {
    "C-00667": {
        "action": {
            "object": (
                "the HOF Entities the right to audit all relevant Constellation records "
                "related to New Business"
            )
        },
        "note": "trim object; sync conditional flag with temporal condition",
    },
    "C-00670": {
        "phenomena": {"negation": True},
        "note": "role other for term clause; tag negation in relative clauses",
    },
    "C-00674": {
        "action": {
            "object": (
                "annual audits, semi-annual financial statements, quarterly earnings statements, "
                "monthly securities lists, monthly balance sheets, and additional Fund financial information"
            )
        },
        "note": "complete truncated furnish object list",
    },
    "C-00688": {
        "note": "sync conditional flag with temporal condition",
    },
    "C-00695": {
        "triplet": {
            "subject": {"text": "the laws of the State of California, USA", "role": "other"},
            "action": {"predicate": "govern", "object": "this Agreement"},
            "condition": {"text": "", "type": "none"},
        },
        "note": "govern clause: laws subject, govern predicate; without-reference is inline",
    },
    "C-00696": {
        "condition": {
            "text": "where possible and appropriate, where reasonably practicable",
            "type": "trigger",
        },
        "note": "where-clauses are trigger conditions on communicate duty",
    },
    "C-00697": {
        "action": {
            "object": (
                "any defamatory, misleading or disparaging remarks, comments or statements "
                "concerning the other Party, its affiliates, or their software, products or services"
            )
        },
        "note": "complete object; tag shall-not negation",
    },
    "C-00699": {
        "note": "neither-party prohibition; tag negation",
    },
    "C-00705": {
        "note": "sync conditional flag with term commencement temporal",
    },
    "C-00707": {
        "condition": {"text": "", "type": "none"},
        "note": "without-consent is inline prohibition frame; tag may-not negation",
    },
    "C-00720": {
        "condition": {"text": "", "type": "none"},
        "note": "without-consent inline on may-not transfer prohibition",
    },
    "C-00721": {
        "action": {
            "object": "Product Name and Agreed Quantity of Units to be purchased per Annum"
        },
        "note": "passive listing clause needs table header object",
    },
    "C-00729": {
        "note": "passive may-not assign; agent from by-phrase; tag negation",
    },
    "C-00731": {
        "condition": {"text": "", "type": "none"},
        "note": "passive shall-not-be-placed; without-consent inline; tag negation",
    },
    "C-00732": {
        "condition": {
            "text": "without the express prior written approval of the Association",
            "type": "exception",
        },
        "note": "may-not permit prohibition; sync conditional; tag negation",
    },
    "C-00734": {
        "note": "shall-not challenge prohibition; sync temporal conditional flag",
    },
    "C-00735": {
        "action": {"predicate": "valid", "object": "for 5 years"},
        "note": "valid lemma; sync temporal conditional flag",
    },
    "C-00754": {
        "note": "by-giving-notice is manner not condition; clear false conditional flag",
    },
    "C-00755": {
        "note": "non-party termination subject → role other; tag no-longer-than negation",
    },
    "C-00756": {
        "condition": {"text": "", "type": "none"},
        "note": "may-not assign; without-consent inline; tag negation",
    },
    "C-00765": {
        "condition": {"text": "", "type": "none"},
        "note": "passive may-not assign by Licensee; without-consent inline; tag negation",
    },
    "C-00766": {
        "action": {
            "object": (
                "a non-exclusive, worldwide royalty-free license for continued use of the "
                "Licensed Mark for production and sale of inventory containing the Licensed Mark"
            )
        },
        "note": "trim grant object to complete NP",
    },
    "C-00777": {
        "note": "Subject-to-herein is trigger condition; sync conditional flag",
    },
    "C-00781": {
        "note": "declarative expire clause; role other",
    },
    "C-00786": {
        "condition": {"text": "", "type": "none"},
        "note": "Subject-to-agreement is procedural inline modifier not condition",
    },
    "C-00787": {
        "action": {
            "object": (
                "on all public-facing materials that the Licensee is no longer operating under "
                "the Licensed Mark or associated with the Licensor"
            )
        },
        "note": "trim specify object; sync temporal conditional; tag no-longer negation",
    },
    "C-00790": {
        "subject": {"role": "other"},
        "note": "declarative expire clause → role other",
    },
    "C-00793": {
        "condition": {"text": "", "type": "none"},
        "note": "shall-not transfer; without-consent inline; tag negation",
    },
    "C-00795": {
        "action": {
            "object": (
                "on all public-facing materials that Licensee is no longer operating under "
                "the Licensed Mark or associated with Licensor"
            )
        },
        "note": "trim specify object; sync temporal conditional; tag no-longer negation",
    },
    "C-00798": {
        "note": "term commencement metadata; role other",
    },
    "C-00802": {
        "note": "sync For-each-Day temporal conditional; tag not-to-exceed negation",
    },
    "C-00806": {
        "note": "sync indemnity scope trigger condition; tag BUT-NOT-TO carve-out negation",
    },
    "C-00809": {
        "note": "govern-style controversy clause; laws subject; tag not-resolved negation",
    },
    "C-00836": {
        "note": "term continue clause; role other with unless exception",
    },
    "C-00842": {
        "action": {"object": "a per Transaction Inquiry amount"},
        "note": "trim consist object; no subordinate condition (false conditional flag)",
    },
    "C-00844": {
        "condition": {"text": "", "type": "none"},
        "note": "Subject-to-agreement is procedural not extracted condition",
    },
    "C-00845": {
        "subject": {"role": "other"},
        "action": {
            "object": "to the extent necessary to complete pending Customer transactions"
        },
        "note": "survival scope as object; declarative survive clause → role other",
    },
    "C-00848": {
        "note": "term commence metadata; role other",
    },
    "C-00850": {
        "action": {
            "object": (
                "that the foregoing restriction applies only to persistent sponsorship "
                "placement, and not to run-of-site banner advertisements or other rotating "
                "promotional placements"
            )
        },
        "note": "complete acknowledge object; tag not-to scope negation",
    },
    "C-00853": {
        "condition": {"text": "", "type": "none"},
        "action": {
            "object": (
                "a non-exclusive, nontransferable, royalty-free, worldwide license to use, "
                "reproduce, publish, perform and display the Snap Marks, Snap Brand Features, "
                "and Snap Content"
            )
        },
        "note": "Subject-to-Section procedural; expand grant object",
    },
    "C-00871": {
        "note": "sync conditional flag with term begin temporal",
    },
    "C-00873": {
        "action": {
            "object": (
                "advertising relating to the commercial printing entities listed on Exhibit \"A\""
            )
        },
        "note": "sync temporal conditional; tag shall-not negation; trim object",
    },
    "C-00877": {
        "note": "sync conditional flag with upon-termination temporal",
    },
    "C-00878": {
        "note": "in-accordance-with is procedural; clear false conditional flag",
    },
    "C-00879": {
        "note": "simple grant with no subordinate condition; clear false conditional flag",
    },
}


def apply_manual(triplet: dict, patch: dict) -> dict:
    t = copy.deepcopy(triplet)
    if "triplet" in patch:
        return copy.deepcopy(patch["triplet"])
    for section in ("subject", "action", "condition"):
        if section in patch:
            t[section] = {**t.get(section, {}), **patch[section]}
    return t


def tag_negation(text: str, phen: dict, issues: list[str]) -> None:
    if "negation_not_tagged" in issues or DEONTIC_NEG_RE.search(text):
        if DEONTIC_NEG_RE.search(text) or re.search(
            r"\b(not resolved|BUT NOT TO|not to run-of|no longer)\b", text, re.I
        ):
            phen["negation"] = True


def fix_item(item: dict) -> dict:
    text = item["text"]
    issues = item["issues"]
    phen = copy.deepcopy(item["phenomena"])
    triplet = copy.deepcopy(item["triplet"])

    if GOVERN_RE.search(text):
        triplet = fix_govern_clause(text, triplet)
    elif "C-00809" == item["clause_id"]:
        triplet = fix_determine_laws(text, triplet)

    triplet = strip_without_consent(text, triplet)
    triplet = strip_procedural_subject_to(triplet)

    cid = item["clause_id"]
    note = ""
    if cid in MANUAL:
        patch = MANUAL[cid]
        triplet = apply_manual(triplet, patch)
        if "phenomena" in patch:
            phen.update(patch["phenomena"])
        note = patch.get("note", "")

    # Sync conditional flag with condition field (fixes annot/flag mismatch)
    sync_conditional(phen, triplet)

    # Explicit cond_flag_no_annot: no condition → conditional false
    if "cond_flag_no_annot" in issues:
        if not (triplet.get("condition") or {}).get("text", "").strip():
            phen["conditional"] = False

    # Explicit cond_annot_no_flag: has condition → conditional true
    if "cond_annot_no_flag" in issues:
        if (triplet.get("condition") or {}).get("text", "").strip():
            phen["conditional"] = True

    tag_negation(text, phen, issues)

    # Term / declarative role fixes
    pred = triplet["action"]["predicate"]
    subj = triplet["subject"]["text"]
    if pred in {"commence", "begin", "continue", "expire", "renew", "terminate", "valid", "survive"}:
        if re.search(r"\b(term|agreement|contract|this agreement)\b", subj, re.I):
            triplet["subject"]["role"] = "other"

    return {
        "clause_id": cid,
        "triplet": triplet,
        "phenomena": phen,
        "fix_notes": note,
    }


def main() -> None:
    items = [
        json.loads(line)
        for line in INPUT.read_text().splitlines()
        if line.strip()
    ]
    out = [fix_item(item) for item in items]
    with OUTPUT.open("w") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} records to {OUTPUT}")


if __name__ == "__main__":
    main()
