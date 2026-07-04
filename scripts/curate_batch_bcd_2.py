#!/usr/bin/env python3
"""Produce fixed_batch_bcd_2.jsonl from fix_batch_bcd_2.jsonl."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data/processed/curated_500/fix_batch_bcd_2.jsonl"
OUTPUT = ROOT / "data/processed/curated_500/fixed_batch_bcd_2.jsonl"

WITHOUT_CONSENT_RE = re.compile(
    r"\bwithout\s+(?:the\s+)?(?:prior\s+)?(?:written\s+)?(?:express\s+)?"
    r"(?:consent|approval)\b",
    re.I,
)
NEGATION_RE = re.compile(
    r"\b(shall not|may not|must not|will not|neither\b|nor\b.*\bshall\b|"
    r"does not|do not|not assign|not be assigned|excludes|excludes or limits)\b",
    re.I,
)
SUBJECT_TO_RE = re.compile(
    r"^Subject to (?:the terms(?: and conditions)? of this [Aa]greement|Section\s+\d)",
    re.I,
)


def has_negation(text: str, triplet: dict) -> bool:
    role = (triplet.get("subject") or {}).get("role", "")
    if role == "prohibited_party":
        return True
    return bool(NEGATION_RE.search(text))


def sync_conditional(phen: dict, triplet: dict) -> dict:
    phen = dict(phen)
    has_cond = bool((triplet.get("condition") or {}).get("text", "").strip())
    phen["conditional"] = has_cond
    return phen


def strip_without_consent(text: str, triplet: dict) -> dict:
    t = copy.deepcopy(triplet)
    cond = t.get("condition") or {}
    ctext = (cond.get("text") or "").strip()
    if not ctext:
        return t
    if re.search(r"\bshall not\b|\bmay not\b|\bneither\b|\bwill not\b", text, re.I):
        if WITHOUT_CONSENT_RE.search(ctext):
            t["condition"] = {"text": "", "type": "none"}
    return t


def fix_liable(triplet: dict, text: str) -> dict:
    t = copy.deepcopy(triplet)
    pred = t["action"]["predicate"]
    if pred == "be" and re.search(r"\bshall not be liable\b|\bbe liable\b", text, re.I):
        t["action"]["predicate"] = "liable"
    return t


def trim_object(obj: str, max_len: int = 175) -> str:
    if len(obj) <= max_len:
        return obj
    cut = obj[:max_len].rsplit(" ", 1)[0]
    return cut


MANUAL: dict[str, dict] = {
    "C-00637": {
        "triplet": {
            "subject": {"text": "CHT", "role": "obligor"},
            "action": {
                "predicate": "provide",
                "object": (
                    "reasonable access to all facilities, systems and assets used by CHT, "
                    "to CHT personnel and subcontractors and to all relevant CHT books and records"
                ),
            },
            "condition": {
                "text": "upon ten (10) Business Days prior written notice",
                "type": "temporal",
            },
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "complete provide-object NP; upon-notice is temporal condition",
    },
    "C-00638": {
        "triplet": {
            "subject": {"text": "neither Party", "role": "prohibited_party"},
            "action": {"predicate": "exclude", "object": "any liability"},
            "condition": {
                "text": "Notwithstanding Sections 17(a) and 17(b)",
                "type": "exception",
            },
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "trim object to any liability; tag neither/excludes negation",
    },
    "C-00652": {
        "triplet": {
            "subject": {"text": "Channel Partner", "role": "obligor"},
            "action": {
                "predicate": "grant",
                "object": "a royalty-free, non-exclusive, non-transferable, limited license right",
            },
            "condition": {
                "text": "Subject to the terms and conditions of this agreement",
                "type": "trigger",
            },
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "add Subject-to trigger condition",
    },
    "C-00749": {
        "triplet": {
            "subject": {"text": "Buyer or Seller", "role": "prohibited_party"},
            "action": {"predicate": "assign", "object": "This Agreement"},
            "condition": {
                "text": (
                    "except that Seller may assign all of its rights and obligations hereunder "
                    "to any entity of which Exxon Mobil Corporation owns, directly or indirectly, "
                    "at least fifty percent (50%) of the shares or other indicia of equity having "
                    "the right to elect such entity's board of directors or other governing body"
                ),
                "type": "exception",
            },
        },
        "phenomena": {
            "passive": True,
            "conditional": True,
            "relative_clause": True,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "passive assign by Buyer/Seller; except-carve-out is condition; tag negation",
    },
    "C-00775": {
        "triplet": {
            "subject": {"text": "Licensee", "role": "prohibited_party"},
            "action": {
                "predicate": "assign",
                "object": "this Agreement or its right to use the Brand",
            },
            "condition": {
                "text": (
                    "except for an assignment outside of bankruptcy to a successor organization "
                    "that is solely the result of a name change by Licensee"
                ),
                "type": "exception",
            },
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": True,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "may-not prohibition; except-carve-out is condition; without-consent inline",
    },
    "C-00794": {
        "triplet": {
            "subject": {"text": "Licensor", "role": "obligor"},
            "action": {
                "predicate": "grant",
                "object": "a personal, non-exclusive, royalty-free right and license to use the Licensed Mark",
            },
            "condition": {
                "text": "Subject to the terms and conditions of this Agreement",
                "type": "trigger",
            },
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": True,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "add Subject-to trigger condition",
    },
    "C-00801": {
        "triplet": {
            "subject": {"text": "either Party", "role": "prohibited_party"},
            "action": {"predicate": "assign", "object": "This Agreement"},
            "condition": {"text": "except as provided below", "type": "exception"},
        },
        "phenomena": {
            "passive": True,
            "conditional": True,
            "relative_clause": True,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "passive may-not assign; except-as-provided is condition; tag negation",
    },
    "C-00847": {
        "triplet": {
            "subject": {"text": "NEITHER PARTY", "role": "prohibited_party"},
            "action": {
                "predicate": "liable",
                "object": "SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES OR LOST PROFITS",
            },
            "condition": {
                "text": "EXCEPT IN THE EVENT OF A BREACH OF SECTION 11",
                "type": "exception",
            },
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "shall not be liable → lemma liable; tag neither negation",
    },
    "C-00884": {
        "triplet": {
            "subject": {"text": "About", "role": "obligor"},
            "action": {"predicate": "make", "object": "commercially reasonable efforts"},
            "condition": {
                "text": "in twelve (12) months or less from the Effective Date",
                "type": "temporal",
            },
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "frame second sentence About/make; temporal effort deadline",
    },
    "C-00006": {
        "triplet": {
            "subject": {"text": "MA", "role": "prohibited_party"},
            "action": {
                "predicate": "assign",
                "object": "any of the rights granted pursuant to this Agreement",
            },
            "condition": {"text": "", "type": "none"},
        },
        "phenomena": {
            "passive": False,
            "conditional": False,
            "relative_clause": False,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "remove without-approval spurious condition; tag may-not negation",
    },
    "C-00020": {
        "triplet": {
            "subject": {"text": "Licensor", "role": "obligor"},
            "action": {"predicate": "make", "object": "not less than ten (10) Licensed Programs"},
            "condition": {"text": "at all times during the Term", "type": "temporal"},
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "sync phenomena.conditional with during-Term temporal scope",
    },
    "C-00026": {
        "triplet": {
            "subject": {"text": "License Term", "role": "other"},
            "action": {"predicate": "be", "object": "Perpetual, unlimited runs"},
            "condition": {"text": "Commencing: November 15, 2012", "type": "temporal"},
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "sync phenomena.conditional with Commencing temporal",
    },
    "C-00027": {
        "triplet": {
            "subject": {"text": "the laws of the State of Florida", "role": "other"},
            "action": {"predicate": "govern", "object": "This Agreement"},
            "condition": {"text": "", "type": "none"},
        },
        "phenomena": {
            "passive": True,
            "conditional": False,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "govern clause: laws govern Agreement",
    },
    "C-00031": {
        "triplet": {
            "subject": {"text": "Producer", "role": "obligor"},
            "action": {
                "predicate": "grant",
                "object": (
                    "the right and license to Distribute the Program on any ConvergTV channel, "
                    "and/or other distribution outlets, that exists today or that is created "
                    "or developed in the future"
                ),
            },
            "condition": {"text": "", "type": "none"},
        },
        "phenomena": {
            "passive": False,
            "conditional": False,
            "relative_clause": True,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "active grant frame; relative clause on outlets",
    },
    "C-00033": {
        "triplet": {
            "subject": {"text": "Each of the Parties", "role": "right_holder"},
            "action": {
                "predicate": "audit",
                "object": "the other Party's compliance with this Agreement",
            },
            "condition": {"text": "", "type": "none"},
        },
        "phenomena": {
            "passive": False,
            "conditional": False,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "at-own-expense is manner; no subordinate condition",
    },
    "C-00035": {
        "triplet": {
            "subject": {"text": "this Agreement", "role": "other"},
            "action": {"predicate": "renew", "object": ""},
            "condition": {
                "text": (
                    "unless either party provides the other with written notice of non-renewal "
                    "at least ninety (90) days before the expiration of the Initial Term"
                ),
                "type": "exception",
            },
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "auto-renew unless notice; declarative term subject",
    },
    "C-00039": {
        "triplet": {
            "subject": {"text": "Licensee", "role": "prohibited_party"},
            "action": {"predicate": "assign", "object": "any of its rights"},
            "condition": {"text": "", "type": "none"},
        },
        "phenomena": {
            "passive": False,
            "conditional": False,
            "relative_clause": False,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "remove without-consent spurious condition; tag shall-not negation",
    },
    "C-00057": {
        "triplet": {
            "subject": {"text": "the parties", "role": "obligor"},
            "action": {
                "predicate": "grant",
                "object": "a limited license to use each other's proprietary marks",
            },
            "condition": {"text": "Throughout the Term of this Agreement", "type": "temporal"},
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "sync phenomena.conditional with Throughout-Term temporal",
    },
    "C-00060": {
        "triplet": {
            "subject": {"text": "This agreement", "role": "other"},
            "action": {"predicate": "renew", "object": "additional successive terms of twelve (12) months each"},
            "condition": {
                "text": (
                    "unless either party notifies the other in writing at least sixty (60) days "
                    "prior to the end of the Initial Term"
                ),
                "type": "exception",
            },
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "auto-renew unless notice exception",
    },
    "C-00073": {
        "triplet": {
            "subject": {"text": "This agreement", "role": "other"},
            "action": {"predicate": "commence", "object": ""},
            "condition": {"text": "as of date first above written", "type": "temporal"},
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "sync phenomena.conditional with commencement temporal",
    },
    "C-00079": {
        "triplet": {
            "subject": {"text": "TL", "role": "obligor"},
            "action": {
                "predicate": "purchase",
                "object": "a minimum of ten thousand (10,000) units of each recorded Product",
            },
            "condition": {"text": "during the first thirty-two (32) months of release", "type": "temporal"},
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "sync phenomena.conditional with during-release temporal",
    },
    "C-00084": {
        "triplet": {
            "subject": {"text": "TL", "role": "prohibited_party"},
            "action": {
                "predicate": "do",
                "object": "any act or thing which will in any way impair Integrity's rights in and to the Integrity Trademarks",
            },
            "condition": {"text": "at any time", "type": "temporal"},
        },
        "phenomena": {
            "passive": True,
            "conditional": True,
            "relative_clause": True,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "will-not prohibition; at-any-time temporal scope; tag negation",
    },
    "C-00086": {
        "triplet": {
            "subject": {"text": "this Agreement", "role": "other"},
            "action": {"predicate": "renew", "object": "an additional term of three (3) years"},
            "condition": {
                "text": (
                    "In the event that MusclePharm shall achieve Net Sales (as defined below) "
                    "of $20 million (the \"First Renewal Threshold\") in the aggregate during "
                    "the Third Contract Year"
                ),
                "type": "trigger",
            },
        },
        "phenomena": {
            "passive": True,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "renew object is additional term not Agreement duplicate",
    },
    "C-00087": {
        "triplet": {
            "subject": {"text": "the laws of the State of California", "role": "other"},
            "action": {"predicate": "construe", "object": "This Agreement"},
            "condition": {"text": "", "type": "none"},
        },
        "phenomena": {
            "passive": True,
            "conditional": False,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "construed/enforced → laws construe Agreement",
    },
    "C-00090": {
        "triplet": {
            "subject": {"text": "Endorser and the Lender", "role": "prohibited_party"},
            "action": {"predicate": "enter", "object": "any other endorsement agreement"},
            "condition": {
                "text": "During the term of this Agreement, or any extensions of this Agreement",
                "type": "temporal",
            },
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "During-Term temporal; tag will-not negation",
    },
    "C-00096": {
        "triplet": {
            "subject": {"text": "an employee-for-hire of MusclePharm or a third party", "role": "obligor"},
            "action": {
                "predicate": "prepare",
                "object": "All such works based upon the Trademarks and/or Name and Appearance Rights",
            },
            "condition": {"text": "", "type": "none"},
        },
        "phenomena": {
            "passive": True,
            "conditional": False,
            "relative_clause": True,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "passive prepare-by agent; provided-however is separate frame",
    },
    "C-00101": {
        "triplet": {
            "subject": {"text": "MusclePharm", "role": "obligor"},
            "action": {
                "predicate": "obtain",
                "object": (
                    "a commercial general liability insurance policy including coverage for "
                    "contractual liability, product liability, personal injury liability, "
                    "and advertiser's liability"
                ),
            },
            "condition": {
                "text": (
                    "throughout the Term of the Agreement and for a period of not less than "
                    "four years thereafter"
                ),
                "type": "temporal",
            },
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "sync phenomena.conditional with throughout-Term temporal",
    },
    "C-00113": {
        "triplet": {
            "subject": {"text": "each", "role": "obligor"},
            "action": {
                "predicate": "assign",
                "object": (
                    "all copyrights, patents, trade marks, service marks, rights of publicity, "
                    "authors' rights, contract and licensing rights, goodwill and all other "
                    "intellectual property rights"
                ),
            },
            "condition": {
                "text": (
                    "if such rights comprise (i) intellectual property that constitutes "
                    "predominantly communication software or related communication hardware "
                    "or other technology, including without limitation, any upgrades and "
                    "Improvements thereof, or (ii) any \"user\" names, and other \"user profile\" "
                    "information included within the Company-Skype Branded Application"
                ),
                "type": "trigger",
            },
        },
        "phenomena": {
            "passive": True,
            "conditional": True,
            "relative_clause": True,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "trim assign object; tag neither/will-not grant negation in frame",
    },
    "C-00128": {
        "triplet": {
            "subject": {"text": "Each Party", "role": "obligor"},
            "action": {
                "predicate": "assign",
                "object": "an undivided one-half right, title and interest in and to all Joint IP",
            },
            "condition": {"text": "", "type": "none"},
        },
        "phenomena": {
            "passive": False,
            "conditional": False,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "to-whom-vests is subject modifier not condition",
    },
    "C-00138": {
        "triplet": {
            "subject": {"text": "The Parties", "role": "prohibited_party"},
            "action": {"predicate": "assign", "object": "this Agreement or any part of it"},
            "condition": {"text": "", "type": "none"},
        },
        "phenomena": {
            "passive": False,
            "conditional": False,
            "relative_clause": False,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "remove without-consent spurious condition; tag shall-not negation",
    },
    "C-00150": {
        "triplet": {
            "subject": {"text": "FCE", "role": "prohibited_party"},
            "action": {
                "predicate": "conduct",
                "object": (
                    "any Work using Generation 1 Technology in Carbon Capture Applications "
                    "or any Work using Generation 2 Technology"
                ),
            },
            "condition": {"text": "During the Term of this Agreement", "type": "temporal"},
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "During-Term temporal; without-approval inline; tag will-not negation",
    },
    "C-00156": {
        "triplet": {
            "subject": {"text": "ExxonMobil", "role": "obligor"},
            "action": {
                "predicate": "grant",
                "object": (
                    "a worldwide, royalty-bearing, non-exclusive, sub-licensable, right and license "
                    "to practice ExxonMobil Background Information and ExxonMobil Background Patents "
                    "for Generation 2 Technology in any application outside of Power Applications "
                    "and Hydrogen Applications"
                ),
            },
            "condition": {
                "text": (
                    "In the event ExxonMobil notifies FCE that it has formally decided not to pursue "
                    "Generation 2 Technology for Carbon Capture Applications, then upon FCE's "
                    "written request"
                ),
                "type": "trigger",
            },
        },
        "phenomena": {
            "passive": True,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "complete truncated grant object",
    },
    "C-00158": {
        "triplet": {
            "subject": {"text": "FCE", "role": "obligor"},
            "action": {
                "predicate": "grant",
                "object": (
                    "a worldwide, non-exclusive, royalty-free, irrevocable, perpetual, sub-licensable, "
                    "non-transferable right and license to practice FCE Background Information and "
                    "FCE Background Patents for Generation 2 Technology in Carbon Capture Applications "
                    "and Hydrogen Applications"
                ),
            },
            "condition": {
                "text": "To the extent not already granted pursuant to the License Agreement",
                "type": "trigger",
            },
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "complete truncated grant object; sync conditional with To-the-extent trigger",
    },
    "C-00178": {
        "triplet": {
            "subject": {"text": "the Supplier", "role": "obligor"},
            "action": {"predicate": "maintain", "object": "product liability insurance"},
            "condition": {"text": "During the Term", "type": "temporal"},
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "sync phenomena.conditional with During-Term temporal",
    },
    "C-00180": {
        "triplet": {
            "subject": {"text": "This Agreement", "role": "other"},
            "action": {"predicate": "become", "object": "effective"},
            "condition": {"text": "on the date first written above", "type": "temporal"},
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "effective-date declarative; sync conditional with on-date temporal",
    },
    "C-00185": {
        "triplet": {
            "subject": {"text": "The Company", "role": "prohibited_party"},
            "action": {"predicate": "contact", "object": "any of Distributor's Customer's"},
            "condition": {"text": "", "type": "none"},
        },
        "phenomena": {
            "passive": False,
            "conditional": False,
            "relative_clause": False,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "remove without-approval spurious condition; tag shall-not negation",
    },
    "C-00186": {
        "triplet": {
            "subject": {"text": "Neither Party", "role": "prohibited_party"},
            "action": {"predicate": "assign", "object": "any of its rights, interest or obligations hereunder"},
            "condition": {"text": "", "type": "none"},
        },
        "phenomena": {
            "passive": False,
            "conditional": False,
            "relative_clause": False,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "remove without-consent spurious condition; tag neither/shall-not negation",
    },
    "C-00191": {
        "triplet": {
            "subject": {"text": "Distributor", "role": "obligor"},
            "action": {"predicate": "notify", "object": "Company of any shortages, defects, non-conformance"},
            "condition": {"text": "Within Seven (7) days of receipt of such Products", "type": "temporal"},
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "sync phenomena.conditional with Within-days temporal",
    },
    "C-00194": {
        "triplet": {
            "subject": {"text": "This Agreement", "role": "other"},
            "action": {"predicate": "become", "object": "effective"},
            "condition": {"text": "upon the date first written above", "type": "temporal"},
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "effective-date declarative; upon-date temporal condition",
    },
    "C-00198": {
        "triplet": {
            "subject": {"text": "The Distributor", "role": "prohibited_party"},
            "action": {"predicate": "assign", "object": "any of its rights, obligations or privileges"},
            "condition": {"text": "", "type": "none"},
        },
        "phenomena": {
            "passive": False,
            "conditional": False,
            "relative_clause": False,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "without-consent inline prohibition; tag shall-not negation",
    },
    "C-00206": {
        "triplet": {
            "subject": {"text": "This agreement", "role": "other"},
            "action": {"predicate": "renew", "object": "this agreement"},
            "condition": {
                "text": (
                    "upon the parties mutual agreement on new minimum performance goals "
                    "for the renewal period"
                ),
                "type": "trigger",
            },
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "upon mutual agreement is trigger not temporal",
    },
    "C-00211": {
        "triplet": {
            "subject": {"text": "Distributor", "role": "prohibited_party"},
            "action": {
                "predicate": "assign",
                "object": "any duties or obligations arising under this Agreement",
            },
            "condition": {"text": "", "type": "none"},
        },
        "phenomena": {
            "passive": True,
            "conditional": False,
            "relative_clause": True,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "remove without-consent spurious condition; tag may-not negation",
    },
    "C-00217": {
        "triplet": {
            "subject": {"text": "The initial term of this Agreement", "role": "other"},
            "action": {"predicate": "commence", "object": ""},
            "condition": {"text": "unless sooner terminated", "type": "exception"},
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "term clause declarative; unless-sooner-terminated exception only",
    },
    "C-00225": {
        "triplet": {
            "subject": {"text": "Performance Benchmarks", "role": "other"},
            "action": {
                "predicate": "mean",
                "object": (
                    "the following requirements necessary for Distributor to maintain "
                    "the exclusivity granted in Section 2.1 hereof"
                ),
            },
            "condition": {"text": "", "type": "none"},
        },
        "phenomena": {
            "passive": False,
            "conditional": False,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": True,
        },
        "fix_notes": "definition clause shall mean; no subordinate condition on main frame",
    },
    "C-00233": {
        "triplet": {
            "subject": {"text": "Talent", "role": "prohibited_party"},
            "action": {
                "predicate": "endorse",
                "object": "any other product which is directly competitive to ESSI's products",
            },
            "condition": {"text": "during the Term and in the Territories", "type": "temporal"},
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": True,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "During-Term temporal; trim endorse object; tag will-not negation",
    },
    "C-00234": {
        "triplet": {
            "subject": {"text": "either party", "role": "prohibited_party"},
            "action": {
                "predicate": "assign",
                "object": "this Agreement or any of the rights or obligations contained herein",
            },
            "condition": {"text": "", "type": "none"},
        },
        "phenomena": {
            "passive": True,
            "conditional": False,
            "relative_clause": False,
            "long_distance": False,
            "negation": True,
            "is_definition": False,
        },
        "fix_notes": "remove without-consent spurious condition; tag may-not negation",
    },
    "C-00236": {
        "triplet": {
            "subject": {"text": "ESSI and Talent", "role": "obligor"},
            "action": {"predicate": "negotiate", "object": "additional compensation"},
            "condition": {
                "text": "In the event any Production Session exceeds eight (8) hours in duration",
                "type": "trigger",
            },
        },
        "phenomena": {
            "passive": False,
            "conditional": True,
            "relative_clause": False,
            "long_distance": False,
            "negation": False,
            "is_definition": False,
        },
        "fix_notes": "sync phenomena.conditional with In-the-event trigger",
    },
}


def validate(triplet: dict) -> list[str]:
    issues = []
    pred = triplet["action"]["predicate"]
    if " " in pred:
        issues.append("multiword_pred")
    ctext = (triplet["condition"].get("text") or "").strip()
    ctype = triplet["condition"].get("type", "none")
    if ctext and ctype == "none":
        issues.append("cond_mismatch")
    if not ctext and ctype != "none":
        issues.append("cond_empty")
    obj = triplet["action"].get("object", "")
    if len(obj) > 185:
        issues.append("long_obj")
    return issues


def main() -> None:
    records = [json.loads(line) for line in INPUT.read_text().splitlines() if line.strip()]
    out_rows = []
    validation_failures = []

    for rec in records:
        cid = rec["clause_id"]
        if cid not in MANUAL:
            raise SystemExit(f"Missing manual gold for {cid}")

        fix = MANUAL[cid]
        triplet = copy.deepcopy(fix["triplet"])
        triplet["action"]["object"] = trim_object(triplet["action"].get("object", ""))
        phen = sync_conditional(fix["phenomena"], triplet)

        row = {
            "clause_id": cid,
            "triplet": triplet,
            "phenomena": phen,
            "fix_notes": fix["fix_notes"],
        }
        issues = validate(triplet)
        if issues:
            validation_failures.append((cid, issues))
        out_rows.append(row)

    out_rows.sort(key=lambda r: r["clause_id"])
    with OUTPUT.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_rows)} records to {OUTPUT}")
    if validation_failures:
        print("Validation failures:", validation_failures)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
