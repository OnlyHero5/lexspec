#!/usr/bin/env python3
"""Final gold polish: fix hard annotation errors and normalize auxiliary predicates.

Targets gold_triplets_500.jsonl for publication-quality lemma frames:
  - repair truncated subjects/objects
  - map be/have/not → single content-verb lemmas
  - re-extract truncated objects from source text (max 200 chars)
"""

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
    TERM_PREDICATES,
    fix_role_modality,
    fix_triplet,
    normalize_condition,
    strip_without_consent,
)
from polish_gold_500 import (  # noqa: E402
    extract_subject_near_predicate,
    fix_subject_field,
    is_polluted_subject,
    model_consensus_subject,
)

MAX_OBJECT = 200
AUX_PREDS = frozenset({"be", "have", "not"})
KEEP_PREDICATES = frozenset({"subject"})  # lexical predicate for is-subject-to frames

# ---------------------------------------------------------------------------
# Explicit gold patches for records that resist safe generalization
# ---------------------------------------------------------------------------
CLAUSE_PATCHES: dict[str, dict] = {
    "C-00038": {
        "subject": {
            "text": "any merger, consolidation or reorganization involving Licensee "
            "(regardless of whether Licensee is a surviving or disappearing entity)",
            "role": "other",
        },
        "action": {
            "predicate": "deem",
            "object": "a transfer of rights, obligations or performance under this Agreement "
            "for which Licensor's prior written consent is required",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00048": {
        "subject": {"text": "Neither Party", "role": "prohibited_party"},
        "action": {
            "predicate": "assign",
            "object": "its rights, duties or obligations under this Agreement to any third party in whole or in part",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00367": {
        "subject": {"text": "The Village Media Company", "role": "prohibited_party"},
        "action": {
            "predicate": "challenge",
            "object": "(a) the rights of PFHOF in and to any PFHOF Work, (b) the validity of any PFHOF Work, "
            "(c) PFHOF's right to grant rights or licenses relating to the PFHOF Works or "
            "(d) the validity, legality, or enforceability of this Agreement",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00585": {
        "subject": {"text": "THE FOREGOING SENTENCE", "role": "prohibited_party"},
        "action": {
            "predicate": "limit",
            "object": "(1) indemnification obligations under Sections 11.1-11.2 and (2) damages for breach of Article 9 confidentiality and non-use obligations",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00094": {
        "subject": {"text": "Endorser", "role": "other"},
        "action": {
            "predicate": "receive",
            "object": "ten percent (10%) of Net Sales from the sale of any Products other than the Licensed Products featured and sold in conjunction with the Training Video",
        },
        "condition": {
            "text": "In the event that Endorser agrees to produce the Training Video and Products (other than the Licensed Products) are featured and sold in connection with such Training Video",
            "type": "trigger",
        },
    },
    "C-00259": {
        "subject": {"text": "\"Exclusivity\"", "role": "other"},
        "action": {
            "predicate": "mean",
            "object": "Franchisor shall not grant further Trademark licenses for Smaaash Centres in the Territory, and Franchisee shall not enter into competing gaming centre arrangements in the Territory",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00260": {
        "subject": {"text": "\"Exclusivity\"", "role": "other"},
        "action": {
            "predicate": "mean",
            "object": "Franchisor shall not grant further Trademark licenses for Smaaash Centres in the Territory, and Franchisee shall not enter into competing gaming centre arrangements in the Territory",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00471": {
        "subject": {"text": "Customer", "role": "other"},
        "action": {
            "predicate": "consider",
            "object": "in good faith any alternative dates of inspection or audit proposed by Manufacturer within five (5) days of Manufacturer's receipt of such notice",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00560": {
        "subject": {"text": "Exact", "role": "obligor"},
        "action": {
            "predicate": "notify",
            "object": "Pfizer of its intent to grant Ex-US Commercial Rights to a Third Party outside the Territory",
        },
        "condition": {
            "text": "During the Term, if Exact enters a formal process or intends to grant an exclusive commercial license to a Third Party solely to promote or sell the Product outside the Territory",
            "type": "trigger",
        },
    },
    "C-00095": {
        "subject": {"text": "the Minimum Royalty and Timing of Payment", "role": "other"},
        "action": {"predicate": "specify", "object": "the Contract Year minimum royalty and payment schedule"},
        "condition": {
            "text": "In the event that the Second Renewal Threshold is achieved in the Sixth Contract Year",
            "type": "trigger",
        },
    },
    "C-00092": {
        "subject": {"text": "MusclePharm", "role": "right_holder"},
        "action": {"predicate": "assign", "object": "this Agreement"},
        "condition": {
            "text": "provided that the acquirer of MusclePharm shall have financial resources substantially similar or greater than MusclePharm and shall specifically assume the obligations of MusclePharm under this Agreement in writing prior to the consummation of the change of control transaction",
            "type": "trigger",
        },
    },
    "C-00096": {
        "subject": {"text": "MusclePharm", "role": "other"},
        "action": {
            "predicate": "prepare",
            "object": "all such works based upon the Trademarks and/or Name and Appearance Rights",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00112": {
        "subject": {"text": "Online BVI", "role": "right_holder"},
        "action": {"predicate": "receive", "object": "50% of all Adjusted Net Revenue"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00119": {
        "subject": {"text": "each Party or its independent auditor", "role": "right_holder"},
        "action": {"predicate": "audit", "object": "the other Party's books and records"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00142": {
        "subject": {"text": "ETON", "role": "right_holder"},
        "action": {
            "predicate": "dispose",
            "object": "any existing inventory of any Products then in ETON's possession",
        },
        "condition": {
            "text": "If this Agreement is terminated by Aucta under Section 11.2 or 11.3, then",
            "type": "trigger",
        },
    },
    "C-00154": {
        "subject": {"text": "ExxonMobil", "role": "right_holder"},
        "action": {"predicate": "assign", "object": "this Agreement to its Affiliates"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00167": {
        "subject": {"text": "Company", "role": "prohibited_party"},
        "action": {
            "predicate": "assign",
            "object": "its rights or obligations under this Agreement",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00189": {
        "subject": {"text": "Company", "role": "other"},
        "action": {"predicate": "perform", "object": "any further responsibilities to Distributor"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00201": {
        "subject": {"text": "JRVS", "role": "right_holder"},
        "action": {"predicate": "audit", "object": "the Distributor's books and records"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00205": {
        "subject": {"text": "this Agreement", "role": "other"},
        "action": {"predicate": "continue", "object": ""},
        "condition": {"text": "for an initial term of three (3) years", "type": "temporal"},
    },
    "C-00212": {
        "subject": {"text": "Erchonia", "role": "other"},
        "action": {"predicate": "enforce", "object": "all the rights and remedies provided for herein upon a breach of this agreement"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00231": {
        "subject": {"text": "The term of this Agreement", "role": "other"},
        "action": {"predicate": "continue", "object": ""},
        "condition": {"text": "for one (1) year commencing on the Effective Date", "type": "temporal"},
    },
    "C-00237": {
        "subject": {"text": "ESSI", "role": "right_holder"},
        "action": {"predicate": "use", "object": "the name"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00244": {
        "subject": {"text": "Wade", "role": "obligor"},
        "action": {"predicate": "provide", "object": "continued endorsement by Athlete of the Naked Products"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00306": {
        "subject": {"text": "IBM", "role": "other"},
        "action": {
            "predicate": "limit",
            "object": "its liability to indemnification payments under Section 8.1 and damages for bodily injury",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00308": {
        "subject": {"text": "all counterparts of this Agreement", "role": "other"},
        "action": {"predicate": "constitute", "object": "one and the same agreement"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00317": {
        "subject": {"text": "none of Nuance, SpinCo or any other member of either Group", "role": "other"},
        "action": {
            "predicate": "incur",
            "object": "any Liability for any indirect, special, punitive or consequential damages",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00326": {
        "subject": {"text": "none of Honeywell, SpinCo or any other member of either Group", "role": "other"},
        "action": {
            "predicate": "incur",
            "object": "any Liability for any indirect, special, punitive or consequential damages",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00331": {
        "subject": {"text": "the license grants set forth in Articles 3.00 and 3.01", "role": "other"},
        "action": {"predicate": "become", "object": "exclusive to Investor for a perpetual term"},
        "condition": {
            "text": "If the Option is exercised before the expiration of the Option Period",
            "type": "trigger",
        },
    },
    "C-00346": {
        "subject": {"text": "the University and ArTara", "role": "other"},
        "action": {
            "predicate": "own",
            "object": "all intellectual property or patentable inventions arising out of or in connection with the Project",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00360": {
        "subject": {"text": "PFHOF", "role": "obligor"},
        "action": {
            "predicate": "credit",
            "object": "the Youth Sports License Fee against the Annual Guarantee on the Closing Date and each anniversary of the Closing Date during the Term",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00365": {
        "subject": {"text": "The Village Media Company", "role": "right_holder"},
        "action": {
            "predicate": "sublicense",
            "object": "the production and creation of the HOFV Works and Exploit the HOFV Works",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00366": {
        "subject": {"text": "the Village Media Company and its permitted licensees", "role": "other"},
        "action": {"predicate": "exploit", "object": "the HOFV Works"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00371": {
        "subject": {"text": "Vyera", "role": "other"},
        "action": {
            "predicate": "distribute",
            "object": "the Licensed Product to customers throughout the Territory",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00403": {
        "subject": {"text": "Neither Party", "role": "prohibited_party"},
        "action": {
            "predicate": "assign",
            "object": "any of its duties or obligations under this Agreement",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00408": {
        "subject": {"text": "AT&T", "role": "other"},
        "action": {"predicate": "conduct", "object": "an audit of Vendor's Subcontractors"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00415": {
        "subject": {"text": "This Agreement and all of the provisions hereof", "role": "other"},
        "action": {
            "predicate": "bind",
            "object": "the Parties and their respective successors and permitted assigns",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00452": {
        "subject": {
            "text": "any and all rights, title and interest in any Intellectual Property Rights resulting from any development made by Dexcel which is related to the Product",
            "role": "other",
        },
        "action": {"predicate": "own", "object": "Dexcel and Kitov jointly and equally (50%/50%)"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00474": {
        "subject": {"text": "Manufacturer", "role": "right_holder"},
        "action": {
            "predicate": "provide",
            "object": "the total limits required by any combination of self-insurance, primary and excess coverage",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00485": {
        "subject": {"text": "This Agreement", "role": "other"},
        "action": {"predicate": "continue", "object": ""},
        "condition": {"text": "for a period of five (5) years from the Effective Date", "type": "temporal"},
    },
    "C-00490": {
        "subject": {"text": "The Reseller", "role": "other"},
        "action": {
            "predicate": "include",
            "object": "any additional products developed, manufactured or acquired by Supplier within this Agreement",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00497": {
        "subject": {"text": "EITHER PARTY", "role": "prohibited_party"},
        "action": {
            "predicate": "liable",
            "object": "any incidental, consequential, indirect, special, or punitive damages",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00517": {
        "subject": {"text": "this Agreement", "role": "other"},
        "action": {
            "predicate": "bind",
            "object": "the Parties hereto and their respective successors and permitted assigns",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00520": {
        "subject": {"text": "Calm", "role": "right_holder"},
        "action": {"predicate": "hire", "object": "personnel of its choosing to be present in any Store(s)"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00521": {
        "subject": {"text": "such Product Collateral IP (or aspect thereof)", "role": "other"},
        "action": {"predicate": "deem", "object": "works made for hire for Calm within the meaning of the U.S. Copyright Law"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00537": {
        "subject": {"text": "This Agreement", "role": "other"},
        "action": {"predicate": "expire", "object": ""},
        "condition": {"text": "in accordance with Section 2.1", "type": "temporal"},
    },
    "C-00539": {
        "subject": {"text": "Customer", "role": "right_holder"},
        "action": {
            "predicate": "terminate",
            "object": "any Scope of Work and corresponding Purchase Order for Services",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00547": {
        "subject": {"text": "Supplier", "role": "right_holder"},
        "action": {
            "predicate": "restrict",
            "object": "observation access to prevent undue interference with Supplier's operations",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00584": {
        "subject": {"text": "Valeant", "role": "other"},
        "action": {"predicate": "inspect", "object": "the applicable books and records of the other Party"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00611": {
        "subject": {"text": "the Reseller", "role": "other"},
        "action": {"predicate": "commission", "object": "any Registered Referrals"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00623": {
        "subject": {"text": "Reseller", "role": "other"},
        "action": {"predicate": "sell", "object": "additional Product units in its inventory to Customers and/or End Users"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00632": {
        "subject": {"text": "a Party", "role": "prohibited_party"},
        "action": {
            "predicate": "assign",
            "object": "this Agreement and its rights and obligations hereunder without the prior written consent of the other Party",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00642": {
        "subject": {"text": "This Sub-Reseller Agreement", "role": "other"},
        "action": {"predicate": "become", "object": "effective"},
        "condition": {"text": "as of the later of the dates beneath the Parties' signatures below", "type": "temporal"},
    },
    "C-00660": {
        "subject": {"text": "this Section 2.2", "role": "prohibited_party"},
        "action": {
            "predicate": "apply",
            "object": "to agreements executed prior to the date of this Agreement between the parties",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00741": {
        "subject": {"text": "A change of control", "role": "other"},
        "action": {"predicate": "deem", "object": "an assignment requiring consent hereunder"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00747": {
        "subject": {"text": "The parties' rights and obligations hereunder", "role": "other"},
        "action": {"predicate": "construe", "object": "under the laws of the State of Texas"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00752": {
        "subject": {"text": "Seller", "role": "prohibited_party"},
        "action": {
            "predicate": "exceed",
            "object": "an amount equal to the purchase price of the products to which any such claims relate",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00746": {
        "subject": {"text": "Skype", "role": "obligor"},
        "action": {
            "predicate": "grant",
            "object": "a limited, non-exclusive license to use, market, provide access to, promote, reproduce and display the Skype Intellectual Property",
        },
        "condition": {"text": "during the Term", "type": "temporal"},
    },
    "C-00778": {
        "subject": {"text": "Licensee", "role": "other"},
        "action": {
            "predicate": "sublicense",
            "object": "its rights under Section 1.1 to a current or future wholly owned subsidiary of Licensee",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00814": {
        "subject": {"text": "Shipper", "role": "other"},
        "action": {
            "predicate": "audit",
            "object": "Carrier's applicable books and records for the limited purpose of determining compliance with this Agreement",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00846": {
        "subject": {"text": "the auditing party", "role": "other"},
        "action": {"predicate": "inspect", "object": "the applicable books and records"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00861": {
        "subject": {"text": "HCI", "role": "other"},
        "action": {
            "predicate": "serve",
            "object": "as the exclusive health content partner in the health section of the Sympatico web site",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00869": {
        "subject": {"text": "Either party", "role": "other"},
        "action": {
            "predicate": "liable",
            "object": "any indirect, incidental, consequential, special or exemplary damages",
        },
        "condition": {
            "text": "EXCEPT WITH RESPECT TO THE INDEMNITY OBLIGATIONS IN SECTION 14, THE CONFIDENTIALITY OBLIGATIONS UNDER SECTION 16, AND THE BREACH OF SECTION 15",
            "type": "exception",
        },
    },
}

TRUNCATION_ENDINGS = (
    " its rights and",
    " for a",
    " OR (2) DAMAGES AVAILABLE FOR A",
    " (the \"Joint",
    " and FCE Background",
    " for Generation",
    " approved in",
    " relating to",
    " under the",
    " in connection with",
    " not enter into any arrangement",
)


def trim_object(triplet: dict, max_len: int = MAX_OBJECT) -> dict:
    obj = (triplet.get("action") or {}).get("object", "") or ""
    if len(obj) <= max_len:
        return triplet
    cut = obj[:max_len].rsplit(" ", 1)[0]
    return {**triplet, "action": {**triplet["action"], "object": cut}}


def looks_truncated(obj: str) -> bool:
    obj = (obj or "").strip()
    if not obj:
        return False
    if any(obj.endswith(suffix) for suffix in TRUNCATION_ENDINGS):
        return True
    if len(obj) >= 170 and not re.search(r"[.)\"']$", obj):
        return True
    if obj.endswith(")") and len(obj) < 25 and "(" in obj and obj.count("(") < obj.count(")"):
        return True
    return False


def complete_object_from_text(text: str, triplet: dict) -> dict:
    """Re-extract object span from clause text when the stored object looks cut off."""
    action = triplet.get("action") or {}
    pred = (action.get("predicate") or "").strip().lower()
    obj = (action.get("object") or "").strip()
    if not pred or not obj or not looks_truncated(obj):
        return triplet

    anchor = obj[: min(40, len(obj))]
    idx = text.find(anchor)
    if idx < 0:
        anchor = obj[: min(25, len(obj))]
        idx = text.find(anchor)
    if idx < 0:
        return triplet

    patterns = [
        rf"\b{re.escape(pred)}\s+(.+?)(?:\.|;|$|\s+provided\b|\s+PROVIDED\b)",
        rf"\b{re.escape(pred)}\s+(.+?)(?:\.|;|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text[idx : idx + 800], re.I | re.DOTALL)
        if m:
            candidate = m.group(1).strip().rstrip(",")
            if len(candidate) > len(obj):
                return {**triplet, "action": {**action, "object": candidate}}
    return triplet


def normalize_content_predicate(text: str, triplet: dict) -> dict:
    """Rule-based auxiliary predicate normalization (fallback after explicit patches)."""
    pred = ((triplet.get("action") or {}).get("predicate") or "").strip().lower()
    if pred not in AUX_PREDS:
        return triplet
    t = copy.deepcopy(triplet)

    # shall not ... challenge
    if re.search(r"\bshall not\b[^.]{0,120}\bchallenge\b", text, re.I):
        mobj = re.search(r"\bchallenge\s+(\(.+?\))\.", text, re.I | re.DOTALL)
        subj_m = re.search(r"^([^.]{5,80}?)\s+shall not\b", text.strip(), re.I)
        obj = mobj.group(1).strip() if mobj else (t["action"].get("object") or "")
        subj = subj_m.group(1).strip() if subj_m else t["subject"]["text"]
        return {
            **t,
            "subject": {"text": subj, "role": "prohibited_party"},
            "action": {"predicate": "challenge", "object": obj},
        }

    # SHALL NOT LIMIT (complete object)
    m = re.search(
        r"\bSHALL NOT LIMIT\s+(\(.+?\))\.\s*$",
        text.strip(),
        re.I | re.DOTALL,
    )
    if m:
        return {
            **t,
            "subject": {"text": "THE FOREGOING SENTENCE", "role": "prohibited_party"},
            "action": {"predicate": "limit", "object": m.group(1).strip()},
        }

    # shall be entitled to VERB
    m = re.search(
        r"\b([A-Z][A-Za-z0-9&'() ]{1,60}?)\s+shall be entitled\s+to\s+(\w+)\b([^.;]*)",
        text,
        re.I,
    )
    if m:
        return {
            **t,
            "subject": {"text": m.group(1).strip(), "role": t["subject"].get("role", "right_holder")},
            "action": {"predicate": m.group(2).lower(), "object": m.group(3).strip(" ,")},
        }

    # shall have the right to VERB
    m = re.search(
        r"\b([A-Z][A-Za-z0-9&'() ]{1,80}?)\s+shall have the (?:exclusive )?right to\s+(\w+)\b([^.;]*)",
        text,
        re.I,
    )
    if m:
        return {
            **t,
            "subject": {"text": m.group(1).strip(), "role": "right_holder"},
            "action": {"predicate": m.group(2).lower(), "object": m.group(3).strip(" ,")},
        }

    # shall be liable / BE LIABLE
    if re.search(r"\b(?:shall not be liable|BE LIABLE|shall be liable)\b", text, re.I):
        subj_m = re.search(r"\b((?:NEITHER PARTY|Neither party|either party|EITHER PARTY|Either party))\b", text, re.I)
        subj = subj_m.group(1) if subj_m else t["subject"]["text"]
        obj_m = re.search(
            r"\b(?:liable|LIABLE)\s+(?:to\s+[^.;]+?\s+)?for\s+([^.;]+?)(?:\.|;|$|\s+REGARDLESS|\s+ARISING|\s+HOWEVER)",
            text,
            re.I | re.DOTALL,
        )
        obj = obj_m.group(1).strip() if obj_m else (t["action"].get("object") or "")
        exc_m = re.search(r"\b(EXCEPT[^.;]+?)(?:\.|;|$)", text, re.I)
        condition = (
            {"text": exc_m.group(1).strip(), "type": "exception"}
            if exc_m
            else t.get("condition") or {"text": "", "type": "none"}
        )
        return {
            **t,
            "subject": {"text": subj, "role": "prohibited_party"},
            "action": {"predicate": "liable", "object": obj},
            "condition": condition,
        }

    # shall be binding
    if re.search(r"\bshall be binding\b", text, re.I):
        subj_m = re.search(r"^(.+?)\s+shall be binding\b", text.strip(), re.I | re.DOTALL)
        obj_m = re.search(r"\bbinding upon and inure to the benefit of\s+([^.;]+)", text, re.I)
        return {
            **t,
            "subject": {"text": (subj_m.group(1).strip() if subj_m else t["subject"]["text"]), "role": "other"},
            "action": {
                "predicate": "bind",
                "object": obj_m.group(1).strip() if obj_m else "the Parties and their successors",
            },
        }

    # shall be effective / continue in effect
    if re.search(r"\b(?:shall be effective|is effective|shall continue in effect)\b", text, re.I):
        subj_m = re.search(r"\b((?:This |The )?[A-Za-z ]+?(?:Agreement|Sub-Reseller Agreement))\b", text)
        subj = subj_m.group(1).strip() if subj_m else t["subject"]["text"]
        term_m = re.search(r"\bfor a period of[^.;]+|\bfor \d+[^.;]+|\bas of[^.;]+", text, re.I)
        condition = (
            {"text": term_m.group(0).strip(), "type": "temporal"}
            if term_m
            else t.get("condition") or {"text": "", "type": "none"}
        )
        return {
            **t,
            "subject": {"text": subj, "role": "other"},
            "action": {"predicate": "continue", "object": ""},
            "condition": condition,
        }

    # shall be deemed
    if re.search(r"\bshall be deemed\b", text, re.I):
        subj_m = re.search(r"\b([^,;]{5,80}?)\s+shall be deemed\b", text, re.I)
        obj_m = re.search(r"\bshall be deemed\s+(.+?)(?:\.|;|$|\s+within\b|\s+and/or\b)", text, re.I | re.DOTALL)
        return {
            **t,
            "subject": {"text": subj_m.group(1).strip() if subj_m else t["subject"]["text"], "role": "other"},
            "action": {"predicate": "deem", "object": obj_m.group(1).strip() if obj_m else t["action"].get("object", "")},
        }

    # shall be construed/enforced under
    if re.search(r"\bshall be construed\b|\bshall be enforced under\b", text, re.I):
        subj_m = re.search(r"^(.+?)\s+shall be (?:construed|enforced)\b", text.strip(), re.I)
        obj_m = re.search(r"\bunder\s+([^.;]+)", text, re.I)
        return {
            **t,
            "subject": {"text": subj_m.group(1).strip() if subj_m else t["subject"]["text"], "role": "other"},
            "action": {"predicate": "construe", "object": f"under {obj_m.group(1).strip()}" if obj_m else ""},
        }

    # none of ... shall have any Liability
    if re.search(r"\bnone of\b[^.]{0,120}\bshall[^.]{0,40}\bhave any Liability\b", text, re.I):
        subj_m = re.search(r"\b(none of[^,]+(?:Group|either Group))\b", text, re.I)
        obj_m = re.search(r"\bhave any Liability\s+([^.;]+)", text, re.I)
        return {
            **t,
            "subject": {"text": subj_m.group(1).strip() if subj_m else t["subject"]["text"], "role": "prohibited_party"},
            "action": {
                "predicate": "incur",
                "object": f"any Liability {obj_m.group(1).strip()}" if obj_m else "any Liability",
            },
        }

    # shall not be assignable
    if re.search(r"\bshall not be assignable\b|\bshall be assignable by a Party without\b", text, re.I):
        return {
            **t,
            "subject": {"text": "a Party", "role": "prohibited_party"},
            "action": {
                "predicate": "assign",
                "object": "this Agreement and its rights and obligations hereunder without the prior written consent of the other Party",
            },
        }

    return t


def repair_subject(text: str, triplet: dict, record: dict) -> dict:
    subj = ((triplet.get("subject") or {}).get("text") or "").strip()
    if subj == "entity)" or (subj.endswith(")") and len(subj) < 30):
        candidate = model_consensus_subject(record)
        if not candidate:
            candidate = extract_subject_near_predicate(
                text, (triplet.get("action") or {}).get("predicate", "")
            )
        if candidate and not is_polluted_subject(candidate):
            return {**triplet, "subject": {**triplet["subject"], "text": candidate}}
    if is_polluted_subject(subj):
        return fix_subject_field(text, triplet, record)
    return triplet


def finalize_triplet(text: str, triplet: dict, record: dict) -> tuple[dict, bool]:
    orig = json.dumps(triplet, sort_keys=True)
    t = copy.deepcopy(triplet)

    cid = record.get("clause_id", "")
    if cid in CLAUSE_PATCHES:
        t = copy.deepcopy(CLAUSE_PATCHES[cid])
    else:
        pred = ((t.get("action") or {}).get("predicate") or "").strip().lower()
        if pred in AUX_PREDS:
            t = normalize_content_predicate(text, t)

    t = repair_subject(text, t, record)
    t = complete_object_from_text(text, t)
    t, _ = fix_triplet(text, t)
    t = strip_without_consent(text, t)
    if cid not in CLAUSE_PATCHES:
        t = fix_role_modality(text, t)
    t = normalize_condition(t)
    t = trim_object(t)

    # Explicit patches are authoritative (preserve lemma frames and roles).
    if cid in CLAUSE_PATCHES:
        t = copy.deepcopy(CLAUSE_PATCHES[cid])
        t = trim_object(t)

    return t, json.dumps(t, sort_keys=True) != orig


def write_testset(rows: list[dict]) -> None:
    with TEST.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(
                json.dumps(
                    {
                        "clause_id": rec["clause_id"],
                        "text": rec["text"],
                        "phenomena": rec.get("phenomena") or {},
                    },
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
    aux_before = 0
    aux_after = 0

    for rec in rows:
        pred = ((rec.get("triplet") or {}).get("action") or {}).get("predicate", "").lower()
        if pred in AUX_PREDS:
            aux_before += 1
        triplet, did_change = finalize_triplet(rec["text"], rec.get("triplet") or {}, rec)
        new_pred = (triplet.get("action") or {}).get("predicate", "").lower()
        if new_pred in AUX_PREDS and new_pred not in KEEP_PREDICATES:
            aux_after += 1
        if did_change:
            changed.append(rec["clause_id"])
            rec["triplet"] = triplet
            rec["finalize_changed"] = True

    with GOLD.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    write_testset(rows)

    print(f"finalize_gold_500: changed {len(changed)}/{len(rows)} records")
    print(f"auxiliary predicates: {aux_before} -> {aux_after}")
    if changed:
        print("sample changed:", ", ".join(changed[:20]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
