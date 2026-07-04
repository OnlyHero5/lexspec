#!/usr/bin/env python3
"""Repair ~20% of LexSpec-500 triplets flagged by full-record quality audit.

Only modifies records in PROBLEM_IDS. Other records are left untouched.
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
    GOVERN_RE,
    _agreement_np,
    _extract_laws,
    fix_govern_semantic,
    normalize_condition,
    strip_without_consent,
)

PROBLEM_IDS = frozenset({
    "C-00002", "C-00003", "C-00016", "C-00035", "C-00037", "C-00038", "C-00047",
    "C-00068", "C-00074", "C-00080", "C-00092", "C-00093", "C-00094", "C-00108",
    "C-00110", "C-00114", "C-00116", "C-00124", "C-00131", "C-00137", "C-00139",
    "C-00140", "C-00145", "C-00153", "C-00174", "C-00183", "C-00205", "C-00206",
    "C-00218", "C-00221", "C-00234", "C-00238", "C-00252", "C-00262", "C-00265",
    "C-00280", "C-00285", "C-00291", "C-00305", "C-00308", "C-00311", "C-00320",
    "C-00354", "C-00355", "C-00363", "C-00381", "C-00384", "C-00402", "C-00409",
    "C-00415", "C-00427", "C-00434", "C-00452", "C-00457", "C-00468", "C-00486",
    "C-00505", "C-00541", "C-00560", "C-00564", "C-00595", "C-00597", "C-00603",
    "C-00614", "C-00615", "C-00619", "C-00632", "C-00645", "C-00654", "C-00662",
    "C-00663", "C-00666", "C-00670", "C-00672", "C-00696", "C-00697", "C-00699",
    "C-00715", "C-00731", "C-00748", "C-00755", "C-00764", "C-00773", "C-00781",
    "C-00790", "C-00830", "C-00840", "C-00843", "C-00846", "C-00862", "C-00883",
})

MAX_CONDITION_RATIO = 0.45
MAX_SUBJECT_LEN = 90

CLAUSE_PATCHES: dict[str, dict] = {
    "C-00002": {
        "subject": {"text": "This agreement", "role": "other"},
        "action": {"predicate": "renew", "object": ""},
        "condition": {
            "text": "unless otherwise terminated according to the cancellation or termination provisions contained in paragraph 18 of this Agreement",
            "type": "exception",
        },
    },
    "C-00003": {
        "subject": {"text": "either party", "role": "right_holder"},
        "action": {"predicate": "terminate", "object": "This Agreement"},
        "condition": {
            "text": "at the expiration of its term or any renewal term",
            "type": "temporal",
        },
    },
    "C-00035": {
        "subject": {"text": "this Agreement", "role": "other"},
        "action": {"predicate": "renew", "object": ""},
        "condition": {
            "text": "unless either party provides the other with written notice of non-renewal at least ninety (90) days before the expiration of the Initial Term",
            "type": "exception",
        },
    },
    "C-00016": {
        "subject": {"text": "Rogers", "role": "right_holder"},
        "action": {"predicate": "incorporate", "object": "such term or terms into this Agreement"},
        "condition": {
            "text": "If Licensor enters, or has entered, into an agreement or series of agreements with a third party on terms that are more favourable than those contained in this Agreement",
            "type": "trigger",
        },
    },
    "C-00037": {
        "subject": {"text": "Licensor", "role": "obligor"},
        "action": {
            "predicate": "grant",
            "object": "an exclusive, non-transferable and non-sublicensable license to reproduce, perform, display, transmit and distribute the Licensed Content",
        },
        "condition": {
            "text": "Subject to Licensee's on\u00adgoing compliance with Section 3.2 and all other terms and conditions of this Agreement",
            "type": "trigger",
        },
    },
    "C-00038": {
        "subject": {"text": "any merger, consolidation or reorganization involving Licensee", "role": "other"},
        "action": {
            "predicate": "deem",
            "object": "a transfer of rights, obligations or performance under this Agreement for which Licensor's prior written consent is required",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00047": {
        "subject": {"text": "Licensor", "role": "obligor"},
        "action": {"predicate": "give", "object": "Licensee the first right of negotiation for each Additional Title"},
        "condition": {
            "text": "If, during the Term, Licensor develops or obtains the rights to license any live action or animated feature-length motion picture (each an \"Additional Title\")",
            "type": "trigger",
        },
    },
    "C-00068": {
        "subject": {"text": "Women.com", "role": "obligor"},
        "action": {"predicate": "deliver", "object": "an amount equal to the under-delivery within the same campaign elements"},
        "condition": {
            "text": "If Women.com does not deliver at least 80% of the Quarterly Impression Guarantee for Advertsing Promotions",
            "type": "trigger",
        },
    },
    "C-00074": {
        "subject": {"text": "the laws of the State of Tennessee", "role": "other"},
        "action": {"predicate": "govern", "object": "This Agreement"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00080": {
        "subject": {"text": "TL", "role": "right_holder"},
        "action": {"predicate": "purchase", "object": "limited quantities of the Product"},
        "condition": {
            "text": "provided the quantity of such purchases does not exceed seven percent (7%) of the total royalty bearing units of such Product title purchased by TL",
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
    "C-00093": {
        "subject": {"text": "MusclePharm", "role": "right_holder"},
        "action": {"predicate": "terminate", "object": "This Agreement"},
        "condition": {
            "text": "if death, or physical disability, physical injury, or other incapacity lasting more than eight (8) weeks, causes Endorser to be unable to perform a material amount of the personal or consulting services described in this Agreement",
            "type": "trigger",
        },
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
    "C-00108": {
        "subject": {"text": "the foregoing restrictions", "role": "prohibited_party"},
        "action": {"predicate": "apply", "object": ""},
        "condition": {"text": "In the case of Skype and its Affiliates", "type": "trigger"},
    },
    "C-00110": {
        "subject": {"text": "Skype or Skype Holding", "role": "right_holder"},
        "action": {"predicate": "assign", "object": "this Agreement"},
        "condition": {
            "text": "in the event of a merger, reorganization or sale of all or substantially all of Skype's or Skype Holding's assets or voting securities",
            "type": "trigger",
        },
    },
    "C-00114": {
        "subject": {"text": "the Parties", "role": "right_holder"},
        "action": {"predicate": "own", "object": "such rights"},
        "condition": {
            "text": "if such rights comprise (i) analysis prepared for or on behalf of the Parties as participants in the Company-Skype Branded Application",
            "type": "trigger",
        },
    },
    "C-00116": {
        "subject": {"text": "Skype", "role": "obligor"},
        "action": {"predicate": "license", "object": "that software product"},
        "condition": {
            "text": "in the event that, prior to such time as the Company-Skype Branded Application is updated or upgraded to include the Mobile Technology, Skype or any of its Affiliates makes available a software product",
            "type": "trigger",
        },
    },
    "C-00124": {
        "subject": {"text": "such Party", "role": "right_holder"},
        "action": {"predicate": "notify", "object": "the other Party"},
        "condition": {
            "text": "in the event that the applicable Party decides not to file at all or not to file a continuing or other application to maintain the viability of the U.S part of a family of patents",
            "type": "trigger",
        },
    },
    "C-00131": {
        "subject": {"text": "Such termination, together with the provisions of Section 5.2 of the License Agreement", "role": "other"},
        "action": {"predicate": "constitute", "object": "Stryker's sole remedy and Conformis' exclusive liability"},
        "condition": {
            "text": "in the event of any such rejection or failure by Conformis to deliver materially conforming products",
            "type": "trigger",
        },
    },
    "C-00137": {
        "subject": {"text": "each Party", "role": "right_holder"},
        "action": {"predicate": "assign", "object": "the rights and obligations under this Agreement"},
        "condition": {
            "text": "in connection with the transfer or sale of all or substantially all of its business or in connection with a merger or consolidation",
            "type": "trigger",
        },
    },
    "C-00139": {
        "subject": {"text": "Aucta", "role": "right_holder"},
        "action": {"predicate": "receive", "object": "15% of Net Sales Royalty"},
        "condition": {
            "text": "for as long as ETON is selling the Product(s) in the Territory",
            "type": "temporal",
        },
    },
    "C-00140": {
        "subject": {"text": "ETON", "role": "obligor"},
        "action": {"predicate": "pay", "object": "Aucta the difference between royalty payments in Sections 6.3.1 and 6.3.2"},
        "condition": {
            "text": "If the amount of royalty payment under Section 6.3.1 is less than the amount of royalty payment under Section 6.3.2",
            "type": "trigger",
        },
    },
    "C-00145": {
        "subject": {"text": "the Parties", "role": "obligor"},
        "action": {"predicate": "maintain", "object": "general liability insurance"},
        "condition": {
            "text": "At all times from the first commercial sale of any Product(s) or after the Effective Date through the date which is five (5) years after the final sale of such Product(s)",
            "type": "temporal",
        },
    },
    "C-00153": {
        "subject": {"text": "FCE", "role": "obligor"},
        "action": {"predicate": "provide", "object": "notice to ExxonMobil"},
        "condition": {"text": "Subject to requirements of applicable law", "type": "trigger"},
    },
    "C-00174": {
        "subject": {"text": "the Supplier", "role": "right_holder"},
        "action": {"predicate": "give", "object": "notice"},
        "condition": {
            "text": "if the Distributor purports to assign its rights or obligations under this agreement to any third party without the Supplier's consent",
            "type": "trigger",
        },
    },
    "C-00183": {
        "subject": {"text": "Company", "role": "obligor"},
        "action": {"predicate": "increase", "object": "Such Prices and Volume Discount Prices"},
        "condition": {
            "text": "provided Company provides Distributor with at least Ninety (90) days prior written notice",
            "type": "trigger",
        },
    },
    "C-00205": {
        "subject": {"text": "this Agreement", "role": "other"},
        "action": {"predicate": "continue", "object": ""},
        "condition": {
            "text": "Unless terminated earlier as provided in this agreement",
            "type": "exception",
        },
    },
    "C-00206": {
        "subject": {"text": "This agreement", "role": "other"},
        "action": {"predicate": "renew", "object": ""},
        "condition": {"text": "for a period of three (3) years", "type": "temporal"},
    },
    "C-00218": {
        "subject": {"text": "this Agreement", "role": "other"},
        "action": {"predicate": "renew", "object": ""},
        "condition": {
            "text": "unless either Party provides the other Party written notice of its desire to terminate at least one hundred twenty (120) days prior to the end of the Initial Term or any renewal",
            "type": "exception",
        },
    },
    "C-00221": {
        "subject": {"text": "Hydraspin", "role": "obligor"},
        "action": {"predicate": "enter", "object": "a contract with Distributor regarding the new territory"},
        "condition": {
            "text": "If the Parties are unable to reach an agreement on the terms of exclusivity within ten (10) business days of the date the opportunity is presented to Distributor",
            "type": "trigger",
        },
    },
    "C-00234": {
        "subject": {"text": "either party", "role": "prohibited_party"},
        "action": {"predicate": "assign", "object": "this Agreement or any of the rights or obligations contained herein"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00238": {
        "subject": {"text": "any party", "role": "prohibited_party"},
        "action": {"predicate": "sell", "object": "Such usage"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00252": {
        "subject": {"text": "the exclusive appointment and license set out in Sections 2(a) and 2(b)", "role": "other"},
        "action": {"predicate": "become", "object": "non-exclusive"},
        "condition": {
            "text": "if at any time during the Term hereof, CHT breaches Section 2(d) as determined by arbitration",
            "type": "trigger",
        },
    },
    "C-00262": {
        "subject": {"text": "The Franchisee", "role": "prohibited_party"},
        "action": {"predicate": "assign", "object": "any of its rights and or obligations under this Agreement"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00265": {
        "subject": {"text": "Franchisee", "role": "obligor"},
        "action": {"predicate": "act", "object": "as an agent and for the benefit of Franchisor"},
        "condition": {
            "text": "If Franchisee has obtained or obtains in the future, in any country, any right, title or interest in any Franchisor Property",
            "type": "trigger",
        },
    },
    "C-00280": {
        "subject": {"text": "Limitation of liability as described in this article", "role": "other"},
        "action": {"predicate": "apply", "object": ""},
        "condition": {
            "text": "in case the damage or loss is caused by a Party's willful misconduct (including fraud) or gross negligence",
            "type": "exception",
        },
    },
    "C-00285": {
        "subject": {"text": "This Agreement", "role": "other"},
        "action": {"predicate": "renew", "object": ""},
        "condition": {
            "text": "unless a Party provides the other with written notice of termination at least one hundred twenty (120) days before the end of the Initial Term",
            "type": "exception",
        },
    },
    "C-00291": {
        "subject": {"text": "Licensee", "role": "obligor"},
        "action": {"predicate": "transfer", "object": "all rights, title and interest in and to the VOTOCAST Materials"},
        "condition": {
            "text": "To the extent, if any, that ownership of the VOTOCAST Materials does not automatically vest in VOTOCAST by virtue of this Agreement or otherwise",
            "type": "trigger",
        },
    },
    "C-00305": {
        "subject": {"text": "either party", "role": "prohibited_party"},
        "action": {"predicate": "liable", "object": "special, incidental, or indirect damages or any consequential damages"},
        "condition": {
            "text": "provided that this Section 10.0 does not apply to Customer's failure to pay any amounts owed to IBM",
            "type": "exception",
        },
    },
    "C-00308": {
        "subject": {"text": "all counterparts of this Agreement", "role": "other"},
        "action": {"predicate": "constitute", "object": "one and the same agreement"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00311": {
        "subject": {"text": "any Party to this Agreement (or any of its successors or permitted assigns)", "role": "prohibited_party"},
        "action": {"predicate": "enter", "object": "a consolidation or merger transaction in which such Party is not the surviving entity"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00320": {
        "subject": {"text": "Honeywell", "role": "right_holder"},
        "action": {"predicate": "terminate", "object": "This Agreement"},
        "condition": {"text": "at any time, prior to the Distribution", "type": "temporal"},
    },
    "C-00354": {
        "subject": {"text": "the agreement", "role": "other"},
        "action": {"predicate": "renew", "object": ""},
        "condition": {
            "text": "unless either Party gives written notice to the other Party of intent not to renew at least six (6) months prior to the end of the then-current term",
            "type": "exception",
        },
    },
    "C-00355": {
        "subject": {"text": "the laws of the State of Ohio", "role": "other"},
        "action": {"predicate": "govern", "object": "This Agreement"},
        "condition": {"text": "without regard to conflicts of law provisions", "type": "exception"},
    },
    "C-00363": {
        "subject": {"text": "nothing in this Agreement", "role": "prohibited_party"},
        "action": {
            "predicate": "grant",
            "object": "Village Media Company or its Affiliates the right or license to any live (or near live) rights to Exploit any events or other content",
        },
        "condition": {"text": "For the avoidance of doubt", "type": "exception"},
    },
    "C-00381": {
        "subject": {"text": "an independent certified public accountant selected by CytoDyn", "role": "obligor"},
        "action": {"predicate": "audit", "object": "such records of Vyera and its Affiliates"},
        "condition": {"text": "Upon reasonable prior notice, but not more than once per Calendar Year", "type": "temporal"},
    },
    "C-00384": {
        "subject": {"text": "CytoDyn", "role": "right_holder"},
        "action": {"predicate": "terminate", "object": "this Agreement in its entirety"},
        "condition": {"text": "upon written notice to Vyera on the occurrence of any of the following", "type": "trigger"},
    },
    "C-00402": {
        "subject": {"text": "AT&T", "role": "right_holder"},
        "action": {"predicate": "terminate", "object": "this Agreement"},
        "condition": {
            "text": "in the event that Vendor, prior to Location Acceptance at all Cell Sites and without the prior written consent of AT&T, consummates a sale, assignment, transfer, license or change of control",
            "type": "trigger",
        },
    },
    "C-00409": {
        "subject": {"text": "Neither Party", "role": "prohibited_party"},
        "action": {"predicate": "liable", "object": "any special, consequential, incidental or punitive damages"},
        "condition": {
            "text": "except to the extent such damages are payable by such Party pursuant to its indemnification obligations under Section 3",
            "type": "exception",
        },
    },
    "C-00415": {
        "subject": {"text": "This Agreement and all of the provisions hereof", "role": "other"},
        "action": {"predicate": "bind", "object": "the Parties and their respective successors and permitted assigns"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00427": {
        "subject": {"text": "This Agreement", "role": "other"},
        "action": {"predicate": "renew", "object": ""},
        "condition": {
            "text": "unless either party request in writing, at least ninety (90) days prior to the anniversary date, that this Agreement not be renewed",
            "type": "exception",
        },
    },
    "C-00434": {
        "subject": {"text": "Customer", "role": "obligor"},
        "action": {"predicate": "compensate", "object": "Contractor"},
        "condition": {
            "text": "In the event of termination of this Agreement or a cancellation of a Purchase Order, and/or discontinuance of a Product",
            "type": "trigger",
        },
    },
    "C-00452": {
        "subject": {
            "text": "any and all rights, title and interest in any Intellectual Property Rights resulting from any development made by Dexcel which is related to the Product",
            "role": "other",
        },
        "action": {"predicate": "own", "object": "Dexcel and Kitov jointly and equally (50%/50%)"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00457": {
        "subject": {"text": "Kitov", "role": "obligor"},
        "action": {"predicate": "provide", "object": "written notification of any shortfalls in shipment quantity"},
        "condition": {"text": "within thirty (30) Working Days of receipt of the Product at Kitov's warehouse", "type": "temporal"},
    },
    "C-00468": {
        "subject": {"text": "Customer", "role": "obligor"},
        "action": {"predicate": "cause", "object": "its Personnel to irrevocably transfer, assign and convey all rights, title and interest in and to each of the Inventions"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00486": {
        "subject": {"text": "the Agreement", "role": "other"},
        "action": {"predicate": "renew", "object": ""},
        "condition": {
            "text": "unless one party provides the other party with prior written notice of non-renewal at least sixty (60) days prior to the expiration of the then-current term",
            "type": "exception",
        },
    },
    "C-00505": {
        "subject": {"text": "the Party", "role": "obligor"},
        "action": {"predicate": "forward", "object": "the Nomination"},
        "condition": {"text": "if a Party receives a Nomination", "type": "trigger"},
    },
    "C-00541": {
        "subject": {"text": "Supplier", "role": "prohibited_party"},
        "action": {"predicate": "assign", "object": "this Agreement"},
        "condition": {
            "text": "except to a Third Party which acquires all, or substantially all, of Supplier's business or assets",
            "type": "exception",
        },
    },
    "C-00560": {
        "subject": {"text": "Exact", "role": "obligor"},
        "action": {"predicate": "notify", "object": "Pfizer of its intent to grant Ex-US Commercial Rights to a Third Party outside the Territory"},
        "condition": {
            "text": "During the Term, if Exact enters a formal process or intends to grant an exclusive commercial license to a Third Party solely to promote or sell the Product outside the Territory",
            "type": "trigger",
        },
    },
    "C-00564": {
        "subject": {"text": "Pfizer", "role": "obligor"},
        "action": {"predicate": "invest", "object": "its portion of Shared M&P Expense"},
        "condition": {
            "text": "subject to Exact spending at least twelve million dollars ($12,000,000) in Baseline M&P Expense each Calendar Year",
            "type": "trigger",
        },
    },
    "C-00595": {
        "subject": {"text": "This Agreement", "role": "other"},
        "action": {"predicate": "renew", "object": ""},
        "condition": {
            "text": "unless either Party provides the other Party written notice of non-renewal no later than one hundred twenty (120) days before the end of the Initial Term",
            "type": "exception",
        },
    },
    "C-00597": {
        "subject": {"text": "MMT", "role": "prohibited_party"},
        "action": {"predicate": "commercialize", "object": "any Competing Product in the Field in any country in the Territory"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00603": {
        "subject": {"text": "MMT", "role": "prohibited_party"},
        "action": {"predicate": "grant", "object": "sublicenses of the rights and licenses granted to it in Section 2.1(a)"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00614": {
        "subject": {"text": "Company", "role": "obligor"},
        "action": {
            "predicate": "grant",
            "object": "the right to use and display the Company trademarks, tradenames and other designations of source",
        },
        "condition": {"text": "Subject to the terms of this Agreement", "type": "trigger"},
    },
    "C-00615": {
        "subject": {"text": "THE RESELLER AND ITS AFFILIATES", "role": "prohibited_party"},
        "action": {"predicate": "liable", "object": "to the Company"},
        "condition": {"text": "EXCEPT FOR IN THE EVENT OF WILLFUL MISCONDUCT OR GROSS NEGLIGENCE", "type": "exception"},
    },
    "C-00619": {
        "subject": {"text": "the laws of the State of New York", "role": "other"},
        "action": {"predicate": "govern", "object": "This Agreement"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00632": {
        "subject": {"text": "a Party", "role": "prohibited_party"},
        "action": {"predicate": "assign", "object": "this Agreement and its rights and obligations hereunder without the prior written consent of the other Party"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00645": {
        "subject": {"text": "The Agreement", "role": "other"},
        "action": {"predicate": "renew", "object": ""},
        "condition": {
            "text": "unless either party provides the other party written notification of its intent to terminate the Agreement at least ninety (90) days prior to the end of the then-current term",
            "type": "exception",
        },
    },
    "C-00654": {
        "subject": {"text": "these limitations of liability", "role": "other"},
        "action": {
            "predicate": "exclude",
            "object": "CHANNEL PARTNER's payment obligations, early termination fees, confidentiality breaches, misappropriation of intellectual property, and indemnification obligations",
        },
        "condition": {"text": "", "type": "none"},
    },
    "C-00662": {
        "subject": {"text": "Each of the HOF Entities or Constellation", "role": "right_holder"},
        "action": {"predicate": "terminate", "object": "this Agreement"},
        "condition": {
            "text": "if association with another Party could, in such Party's reasonable opinion, materially damage its brand or reputation",
            "type": "trigger",
        },
    },
    "C-00663": {
        "subject": {"text": "either Party", "role": "prohibited_party"},
        "action": {"predicate": "assign", "object": "this Agreement nor any right or obligation hereunder"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00666": {
        "subject": {"text": "any third party", "role": "prohibited_party"},
        "action": {"predicate": "use", "object": "the HOF Entity Marks"},
        "condition": {
            "text": "except to Constellation's subsidiaries and brands for use in a manner consistent with this Agreement",
            "type": "exception",
        },
    },
    "C-00670": {
        "subject": {"text": "This Agreement", "role": "other"},
        "action": {"predicate": "continue", "object": ""},
        "condition": {
            "text": "provided such continuance is specifically approved at least annually by the Fund's Board of Trustees",
            "type": "trigger",
        },
    },
    "C-00672": {
        "subject": {"text": "This Agreement", "role": "other"},
        "action": {"predicate": "terminate", "object": ""},
        "condition": {"text": "in the event of its assignment (as defined in the 1940 Act)", "type": "trigger"},
    },
    "C-00696": {
        "subject": {"text": "Allied", "role": "obligor"},
        "action": {
            "predicate": "communicate",
            "object": "that Newegg is the exclusive sponsor of the Arena for the technology e-commerce and online retailer categories",
        },
        "condition": {"text": "where possible and appropriate and where reasonably practicable", "type": "trigger"},
    },
    "C-00697": {
        "subject": {"text": "Each Party", "role": "prohibited_party"},
        "action": {"predicate": "make", "object": "any defamatory, misleading or disparaging remarks, comments or statements"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00699": {
        "subject": {"text": "Neither Newegg nor Allied", "role": "prohibited_party"},
        "action": {"predicate": "assign", "object": "any part of its rights or obligations under this Agreement"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00715": {
        "subject": {"text": "The Manufacturer", "role": "obligor"},
        "action": {"predicate": "grant", "object": "exclusive rights to the Customer"},
        "condition": {
            "text": "for the term of ten (10) years from the date of the signing of this agreement",
            "type": "temporal",
        },
    },
    "C-00731": {
        "subject": {"text": "any party", "role": "prohibited_party"},
        "action": {"predicate": "place", "object": "The Association Marks"},
        "condition": {"text": "", "type": "none"},
    },
    "C-00748": {
        "subject": {"text": "the other party", "role": "right_holder"},
        "action": {"predicate": "terminate", "object": "this Agreement"},
        "condition": {
            "text": "In the event that the other party concludes in its sole discretion that such Change of Control, if it is implemented, may result in it and/or its Affiliates being subjected to any fact, matter or circumstance that would materially and adversely affect its business",
            "type": "trigger",
        },
    },
    "C-00755": {
        "subject": {"text": "This Agreement and all rights and licenses granted under this Agreement", "role": "other"},
        "action": {"predicate": "terminate", "object": ""},
        "condition": {"text": "as soon as practicable, but no longer than thirty (30) days, after Licensee is acquired by a third party", "type": "temporal"},
    },
    "C-00764": {
        "subject": {"text": "Licensor", "role": "prohibited_party"},
        "action": {"predicate": "require", "object": "prior written consent for a Change of Control assignment"},
        "condition": {
            "text": "in the event of a Change of Control of Licensee so long as the resulting Person assumes all obligations of Licensee",
            "type": "trigger",
        },
    },
    "C-00773": {
        "subject": {"text": "Licensor", "role": "right_holder"},
        "action": {"predicate": "terminate", "object": "this Agreement"},
        "condition": {"text": "upon written notice for any reason", "type": "temporal"},
    },
    "C-00781": {
        "subject": {"text": "this Agreement", "role": "other"},
        "action": {"predicate": "expire", "object": ""},
        "condition": {
            "text": "if the Investment Advisor or one of its affiliates ceases to serve as investment adviser to the Licensee",
            "type": "trigger",
        },
    },
    "C-00790": {
        "subject": {"text": "This Agreement", "role": "other"},
        "action": {"predicate": "expire", "object": ""},
        "condition": {
            "text": "if the Investment Advisor or one of its affiliates ceases to serve as investment adviser to the Licensee",
            "type": "trigger",
        },
    },
    "C-00830": {
        "subject": {"text": "Company", "role": "obligor"},
        "action": {"predicate": "grant", "object": "a non-exclusive, non-transferable license"},
        "condition": {"text": "during the Term", "type": "temporal"},
    },
    "C-00840": {
        "subject": {"text": "either 2TheMart or i-Escrow, Inc.", "role": "right_holder"},
        "action": {"predicate": "terminate", "object": "this Agreement"},
        "condition": {
            "text": "If a majority of the equity securities of either 2TheMart or i-Escrow, Inc. are acquired by another company during the term of this Agreement",
            "type": "trigger",
        },
    },
    "C-00843": {
        "subject": {"text": "neither party", "role": "prohibited_party"},
        "action": {"predicate": "use", "object": "the Domain Name"},
        "condition": {"text": "to the extent that the Domain Name is deemed a combination mark", "type": "trigger"},
    },
    "C-00846": {
        "subject": {"text": "i-Escrow", "role": "obligor"},
        "action": {"predicate": "pay", "object": "all cost of such audit"},
        "condition": {
            "text": "if the audit reveals overdue payments in excess of ten percent (10%) of the payments owed to date",
            "type": "trigger",
        },
    },
    "C-00862": {
        "subject": {"text": "Either party", "role": "right_holder"},
        "action": {"predicate": "terminate", "object": "this Agreement"},
        "condition": {
            "text": "if either Party's corporate structure has undergone a material ownership change such that its corporate interests are then in conflict with the other Party",
            "type": "trigger",
        },
    },
    "C-00883": {
        "subject": {"text": "NEITHER PARTY", "role": "prohibited_party"},
        "action": {
            "predicate": "liable",
            "object": "ANY INDIRECT, SPECIAL, INCIDENTAL, CONSEQUENTIAL OR EXEMPLARY DAMAGES",
        },
        "condition": {"text": "NOTWITHSTANDING ANYTHING TO THE CONTRARY HEREIN", "type": "exception"},
    },
}


def _match_span(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text, re.I | re.DOTALL)
    return m.group(0).strip().rstrip(",;") if m else None


def trim_condition_span(text: str, triplet: dict) -> dict:
    cond = triplet.get("condition") or {}
    ctext = (cond.get("text") or "").strip()
    ctype = (cond.get("type") or "none").strip().lower()
    if not ctext or len(ctext) <= MAX_CONDITION_RATIO * max(len(text), 1):
        return triplet

    for pat in (
        r"\bunless\b[^.;]+",
        r"\bIf\b[^,]+(?:,\s*[^,]+){0,3}(?=,\s*then\b|\bthen\b)",
        r"\bif\b[^,]+(?:,\s*[^,]+){0,3}(?=,\s*then\b|\bthen\b)",
        r"\bIn the event that [^,]+(?:,\s*[^,]+){0,2}",
        r"\bFor so long as [^,]+",
        r"\bSubject to [^,]+(?:,\s*Section [^,]+)?",
        r"\bprovided(?:\s+that|\s*,\s*however\s*,\s*that)\s+[^.;]+",
        r"\bin the event of [^,]+",
        r"\b(?:at the expiration of|upon \d|within \d|during the Term|for a period of|until )[^,;.]+",
        r"\b(?:Except|EXCEPT) [^,;.]+",
    ):
        span = _match_span(text, pat)
        if span and span.lower() in text.lower() and len(span) < len(ctext):
            return {**triplet, "condition": {"text": span, "type": ctype if ctype != "none" else "trigger"}}
    return triplet


def clean_subject_span(text: str, triplet: dict) -> dict:
    subj = ((triplet.get("subject") or {}).get("text") or "").strip()
    pred = ((triplet.get("action") or {}).get("predicate") or "").strip().lower()
    if not subj:
        return triplet

    if pred in ("renew", "continue", "expire", "terminate"):
        subj = re.sub(r"^(Thereafter,\s*|Upon expiration of [^,]+,\s*)", "", subj, flags=re.I).strip()
    if re.search(r"\bshall\b", subj):
        m = re.match(r"^(This [Aa]greement|the [Aa]greement|this agreement)", subj)
        if m:
            subj = "this Agreement" if "this" in m.group(1).lower() else "The Agreement"
    if pred == "govern" or GOVERN_RE.search(text):
        laws = _extract_laws(text)
        if laws and len(laws) < len(subj):
            subj = laws
    if len(subj) > MAX_SUBJECT_LEN:
        cut = subj[:MAX_SUBJECT_LEN].rsplit(",", 1)[0].strip()
        if len(cut) >= 20:
            subj = cut

    if subj != (triplet.get("subject") or {}).get("text"):
        return {**triplet, "subject": {**triplet["subject"], "text": subj}}
    return triplet


def repair_triplet(text: str, triplet: dict, clause_id: str) -> tuple[dict, bool]:
    if clause_id not in PROBLEM_IDS:
        return triplet, False

    orig = json.dumps(triplet, sort_keys=True)
    if clause_id in CLAUSE_PATCHES:
        t = copy.deepcopy(CLAUSE_PATCHES[clause_id])
    else:
        t = copy.deepcopy(triplet)
        t = clean_subject_span(text, t)
        t = trim_condition_span(text, t)
        t = fix_govern_semantic(text, t)
        t = clean_subject_span(text, t)
        t = strip_without_consent(text, t)
        t = normalize_condition(t)

    changed = json.dumps(t, sort_keys=True) != orig
    return t, changed


def write_testset(rows: list[dict]) -> None:
    with TEST.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(
                json.dumps(
                    {"clause_id": rec["clause_id"], "text": rec["text"], "phenomena": rec.get("phenomena") or {}},
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
        triplet, did = repair_triplet(rec["text"], rec.get("triplet") or {}, rec["clause_id"])
        if did:
            rec["triplet"] = triplet
            rec["quality_fix_changed"] = True
            changed.append(rec["clause_id"])

    with GOLD.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    write_testset(rows)

    print(f"fix_quality_20pct_500: changed {len(changed)}/{len(rows)} (target {len(PROBLEM_IDS)})")
    if changed:
        print("changed:", ", ".join(changed))
    missing = PROBLEM_IDS - set(changed)
    if missing:
        print("unchanged problem ids:", ", ".join(sorted(missing)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
