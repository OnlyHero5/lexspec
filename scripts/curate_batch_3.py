#!/usr/bin/env python3
"""Curate fix_batch_3.jsonl to LexSpec gold quality."""
import copy
import json
from pathlib import Path

INPUT = Path("data/processed/curated_500/fix_batch_3.jsonl")
OUTPUT = Path("data/processed/curated_500/fixed_batch_3.jsonl")

# Manual gold corrections keyed by clause_id
MANUAL = {
    "C-00700": {
        "action": {
            "object": (
                "a revocable, non-transferrable, non-assignable, non-sublicensable, "
                "non-exclusive and limited license to use the Newegg Marks"
            )
        },
        "note": "trim parenthetical from grant object",
    },
    "C-00666": {
        "action": {"object": "any pass-through rights or the HOF Entity Marks"},
        "note": "remove redundant 'use of' from object",
    },
    "C-00632": {
        "action": {"object": "this Agreement or any rights or obligations hereunder"},
        "condition": {
            "text": (
                "provided that either Party shall have the right, on notice to but without "
                "the other Party's consent, to assign this Agreement and its rights and "
                "obligations contained herein, to an affiliate or to a third party who is "
                "not a competitor of the other Party in connection with a sale of all or "
                "substantially all of the assigning Party's business or assets relating "
                "to this Agreement"
            ),
            "type": "exception",
        },
        "note": "fix nor→or object; without-consent is inline; provided-that is exception",
    },
    "C-00618": {
        "condition": {
            "text": (
                "Following the Initial Term, upon written notice to the other Party "
                "of at least 3 months"
            ),
            "type": "temporal",
        },
        "note": "include full temporal condition span",
    },
    "C-00616": {
        "subject": {"role": "other"},
        "note": "effective-date clause is declarative, not obligor duty",
    },
    "C-00604": {
        "condition": {
            "text": "Upon termination of this Agreement",
            "type": "temporal",
        },
        "note": "exclude procedural 'pursuant to' from condition",
    },
    "C-00601": {
        "condition": {
            "text": (
                "except that a Party may make such an assignment without the other Party's "
                "consent to its Affiliates or to a Third Party successor of, or transferee to, "
                "assets of such Party to which this Agreement relates, whether in a merger, "
                "sale of stock, sale of assets or other transaction"
            ),
            "type": "exception",
        },
        "note": "exception clause only; without-consent is inline prohibition frame",
    },
    "C-00567": {
        "action": {
            "object": (
                "an independent certified public accounting firm to examine "
                "the relevant books and records of the Audited Party"
            )
        },
        "note": "trim object to minimal complete NP",
    },
    "C-00559": {
        "condition": {
            "text": (
                "After the date that is eighteen (18) months after the Effective Date, "
                "upon six (6) months prior written notice to the other Party"
            ),
            "type": "temporal",
        },
        "note": "include notice requirement in temporal condition",
    },
    "C-00551": {
        "action": {
            "object": "a copy of all insurance policies maintained under this Article 15"
        },
        "note": "trim object to core NP",
    },
    "C-00545": {
        "action": {
            "object": (
                "a paid-up, royalty-free, non-exclusive license to Customer's "
                "Confidential Information and the Customer Technology"
            )
        },
        "note": "trim object to minimal complete NP",
    },
    "C-00536": {
        "subject": {"role": "other"},
        "note": "automatic stay-in-force is declarative, not party obligation",
    },
    "C-00535": {
        "subject": {"role": "other"},
        "note": "commencement term clause is declarative",
    },
    "C-00489": {
        "action": {
            "object": (
                "a non-sublicensable, non-transferable, exclusive right "
                "to distribute and sell the Products in the Territory"
            )
        },
        "note": "trim object to minimal complete NP",
    },
    "C-00455": {
        "action": {"object": "any liability under Section 8.5"},
        "note": "trim object to core NP",
    },
    "C-00412": {
        "subject": {
            "text": "All AT&T Affiliates and the federal government of the United States",
            "role": "other",
        },
        "note": "trim relative-clause subject; beneficiary status is declarative",
    },
    "C-00397": {
        "action": {
            "object": (
                "JHU a fee equal to one percent (1%) of the Aggregate Consideration"
            )
        },
        "note": "trim object to minimal fee NP",
    },
    "C-00379": {
        "condition": {"text": "", "type": "none"},
        "note": "without-consent is inline prohibition frame, not subordinate condition",
    },
    "C-00364": {
        "action": {
            "object": (
                "a worldwide, non-exclusive, limited, non-sublicenseable and "
                "non-assignable right and license to Exploit the PFHOF Works"
            )
        },
        "note": "trim object to minimal complete NP",
    },
    "C-00359": {
        "condition": {"text": "", "type": "none"},
        "note": "without-consent is inline prohibition frame, not trigger condition",
    },
    "C-00343": {
        "condition": {"text": "", "type": "none"},
        "note": "without-consent is inline prohibition frame, not exception condition",
    },
    "C-00304": {
        "action": {
            "object": (
                "an irrevocable, nonexclusive, worldwide, paid-up license to use, "
                "execute, reproduce, display, and perform copies of such Materials"
            )
        },
        "note": "trim object to minimal license NP",
    },
    "C-00285": {
        "subject": {"role": "other"},
        "action": {"object": ""},
        "note": "auto-renewal is declarative; empty object not 'none'",
    },
    "C-00243": {
        "condition": {"text": "During the Term", "type": "temporal"},
        "note": "temporal only; subject-to limitations is inline modifier",
    },
    "C-00217": {
        "subject": {"role": "other"},
        "condition": {
            "text": "unless sooner terminated",
            "type": "exception",
        },
        "note": "term clause is declarative; exclude procedural pursuant-to",
    },
    "C-00198": {
        "condition": {"text": "", "type": "none"},
        "note": "without-consent is inline prohibition frame",
    },
    "C-00194": {
        "subject": {"role": "other"},
        "note": "effective-date clause is declarative",
    },
    "C-00190": {
        "action": {
            "predicate": "liable",
            "object": "any special, indirect, exemplary or consequential damages",
        },
        "note": "shall not be liable → lemma liable; trim object",
    },
    "C-00180": {
        "subject": {"role": "other"},
        "note": "effective-date clause is declarative",
    },
    "C-00175": {
        "condition": {
            "text": "for the duration of this agreement",
            "type": "temporal",
        },
        "note": "temporal duration only; subject-to is inline trigger modifier",
    },
    "C-00426": {
        "subject": {"role": "other"},
        "note": "commencement term clause is declarative",
    },
    "C-00150": {
        "condition": {
            "text": "During the Term of this Agreement",
            "type": "temporal",
        },
        "note": "temporal frame only; without-approval is inline prohibition",
    },
}


def deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if k == "note":
            continue
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = copy.deepcopy(v)
    return out


def auto_fix(triplet: dict, text: str) -> tuple[dict, str | None]:
    """Apply automatic LexSpec rule fixes. Returns (triplet, note or None)."""
    t = copy.deepcopy(triplet)
    notes = []

    pred = t["action"]["predicate"]
    obj = t["action"]["object"] or ""

    # liable: shall not be liable → liable
    if pred == "be" and "liable" in text.lower() and "shall not be liable" in text.lower():
        t["action"]["predicate"] = "liable"
        notes.append("lemma liable from shall not be liable")

    # empty object literal "none"
    if obj.lower() == "none":
        t["action"]["object"] = ""
        notes.append("empty object not literal 'none'")

    # condition type consistency
    cond_text = (t["condition"].get("text") or "").strip()
    if not cond_text and t["condition"].get("type") != "none":
        t["condition"]["type"] = "none"
        notes.append("empty condition text → type none")
    if cond_text and t["condition"].get("type") == "none":
        # infer type from marker
        low = cond_text.lower()
        if any(m in low for m in ("unless", "except", "notwithstanding")):
            t["condition"]["type"] = "exception"
        elif any(
            m in low
            for m in (
                "when ",
                "upon ",
                "after ",
                "before ",
                "during ",
                "within ",
                "following ",
                "on or before",
            )
        ):
            t["condition"]["type"] = "temporal"
        elif any(
            m in low
            for m in ("if ", "in the event", "provided that", "subject to", "so long as")
        ):
            t["condition"]["type"] = "trigger"
        notes.append("infer condition type from marker")

    if notes:
        return t, "; ".join(notes)
    return t, None


def apply_patch(triplet: dict, patch: dict) -> dict:
    t = copy.deepcopy(triplet)
    for section in ("subject", "action", "condition"):
        if section in patch:
            t[section] = {**t.get(section, {}), **patch[section]}
    return t


def main():
    records = [
        json.loads(line)
        for line in INPUT.read_text().splitlines()
        if line.strip()
    ]
    changed = 0
    out_records = []

    for rec in records:
        cid = rec["clause_id"]
        orig = rec["triplet"]
        triplet = copy.deepcopy(orig)
        note_parts = []

        triplet, auto_note = auto_fix(triplet, rec["text"])
        if auto_note:
            note_parts.append(auto_note)

        if cid in MANUAL:
            patch = MANUAL[cid]
            triplet = apply_patch(triplet, patch)
            if patch.get("note"):
                note_parts.append(patch["note"])

        changed_flag = triplet != orig
        if changed_flag:
            changed += 1

        out = {**rec}
        out["triplet"] = triplet
        out["curated"] = True
        out["curation_changed"] = changed_flag
        out["curation_note"] = "; ".join(note_parts) if note_parts else ""
        out_records.append(out)

    with OUTPUT.open("w") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_records)} records to {OUTPUT}")
    print(f"Changed: {changed}")


if __name__ == "__main__":
    main()
