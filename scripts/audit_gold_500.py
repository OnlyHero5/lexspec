#!/usr/bin/env python3
"""Exhaustive quality audit for gold_triplets_500.jsonl — read-only."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/processed/gold_triplets_500.jsonl"

sys.path.insert(0, str(ROOT / "scripts"))
from build_gold_500 import GOVERN_RE, is_bad_text, TERM_PREDICATES  # noqa: E402
from fix_annotations_500 import (  # noqa: E402
    AUTO_RENEW_RE,
    DEFINITION_RE,
    fix_definition_triplet,
    fix_multi_shall_primary,
    fix_role_modality,
)
from polish_gold_500 import is_polluted_subject, fix_permission_without_consent  # noqa: E402
from validate_quality_95 import (  # noqa: E402
    check_govern_frame,
    check_condition_consistency,
    check_role_modality,
    validate_record,
)

MANNER_IN_COND = re.compile(
    r"\b(upon giving|by giving|in writing only|via overnight|with prior written notice to)\b",
    re.I,
)
SUBJECT_TO_BOILER = re.compile(r"^Subject to the terms(?: and conditions)? of this Agreement\b", re.I)
IF_THEN_RE = re.compile(r"\b(?:If|In the event that)\b.+?\bthen\b", re.I | re.DOTALL)
THEN_MAIN_RE = re.compile(
    r"\bthen\b[^.]*?\b(?:this Agreement|the Agreement|[A-Z][A-Za-z0-9&]+)\s+shall\s+(?:automatically\s+)?(\w+)",
    re.I,
)
IN_EVENT_THEN_RE = re.compile(
    r"\bIn the event that\b.+?\bthen\b[^.]*?\bshall\b", re.I | re.DOTALL,
)
PASSIVE_BY = re.compile(
    r"\bby\s+([^.;,\n]+?)(?:\.|,|;|$|\s+without|\s+except|\s+which|\s+at\s+the|\s+upon)",
    re.I,
)
MAY_BE_PASSIVE = re.compile(r"\bmay be (\w+)(?:ed|en)?\s+by\b", re.I)
SHALL_BE_PASSIVE = re.compile(r"\bshall be (\w+)(?:ed|en)?\s+by\b", re.I)
NEITHER_MAY = re.compile(r"\b(?:Neither [Pp]arty|NEITHER PARTY)\s+may\s+(?:assign|transfer|delegate)\b")
PERM_WITHOUT_CONSENT = re.compile(
    r"\b([A-Z][A-Za-z0-9&]+)\s+may,\s*without consent,\s*(assign|transfer)\b", re.I
)
PROHIBITION_LEAD = re.compile(
    r"\b(?:shall not|may not|must not|will not|neither\s+party\s+shall|neither\s+party\s+may|"
    r"in no event shall|nothing herein shall)\b",
    re.I,
)
PERMISSION_LEAD = re.compile(
    r"\b(?:is entitled to|has the right to|shall have the right|may elect to)\b|\bmay\b(?!\s+not\b)",
    re.I,
)
OBLIGATION_LEAD = re.compile(r"\b(?:shall|must|agrees to|undertakes to|will)\b(?!\s+not\b)", re.I)


@dataclass
class Finding:
    clause_id: str
    category: str
    detail: str
    severity: str  # error | warn
    suggested: dict | None = None


@dataclass
class AuditReport:
    findings: list[Finding] = field(default_factory=list)

    def add(self, clause_id: str, category: str, detail: str, severity: str = "error", suggested=None):
        self.findings.append(Finding(clause_id, category, detail, severity, suggested))


def load_rows() -> list[dict]:
    return [json.loads(line) for line in GOLD.open(encoding="utf-8") if line.strip()]


def norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def lead_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0] if parts else text


def frame_sentence(text: str, subj: str, pred: str) -> str:
    """Return the sentence most likely containing the gold frame."""
    subj_l, pred_l = subj.lower(), pred.lower()
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        if pred_l in sent.lower() and (not subj_l or subj_l in sent.lower()):
            return sent
    if IF_THEN_RE.search(text) and pred_l in text.lower():
        then_m = re.search(r"\bthen\b.+", text, re.I | re.DOTALL)
        if then_m:
            return then_m.group(0)
    return lead_sentence(text)


def expected_role_for_frame(sent: str, pred: str, phen: dict) -> str | None:
    if phen.get("is_definition") or pred in ("mean", "govern", "construe", "interpret"):
        return "other"
    if pred in TERM_PREDICATES and re.search(r"\b(term|agreement)\b", sent, re.I):
        return "other"
    if NEITHER_MAY.search(sent):
        return "prohibited_party"
    if PERM_WITHOUT_CONSENT.search(sent):
        return "right_holder"
    if PROHIBITION_LEAD.search(sent):
        return "prohibited_party"
    if PERMISSION_LEAD.search(sent):
        return "right_holder"
    if OBLIGATION_LEAD.search(sent):
        return "obligor"
    return None


def audit_record(record: dict, report: AuditReport) -> None:
    cid = record["clause_id"]
    text = record["text"]
    triplet = record.get("triplet") or {}
    phen = record.get("phenomena") or {}
    s = triplet.get("subject") or {}
    a = triplet.get("action") or {}
    c = triplet.get("condition") or {}

    subj = (s.get("text") or "").strip()
    role = (s.get("role") or "").strip()
    pred = (a.get("predicate") or "").strip()
    obj = (a.get("object") or "").strip()
    ctext = (c.get("text") or "").strip()
    ctype = (c.get("type") or "none").strip()

    # --- structural (validate_quality_95) ---
    for code in validate_record(record):
        if code.startswith("role_modality"):
            report.add(cid, "role_modality", code, "error")
        else:
            report.add(cid, "structural", code, "error")

    gov = check_govern_frame(text, triplet)
    if gov:
        report.add(cid, "govern_frame", gov, "error")

    cond = check_condition_consistency(triplet)
    if cond:
        report.add(cid, "condition_schema", cond, "error")

    if is_bad_text(text):
        report.add(cid, "bad_text", is_bad_text(text) or "bad", "error")

    if is_polluted_subject(subj):
        report.add(cid, "polluted_subject", subj[:100], "error")

    if subj.lower().startswith("that ") and not re.search(rf"\b{re.escape(subj)}\b", text, re.I):
        report.add(cid, "bad_subject_prefix", f"subject={subj!r}", "error")

    if subj and obj and subj.lower() == obj.lower() and pred not in ("mean", "be", *TERM_PREDICATES):
        if not (role == "other" and pred in TERM_PREDICATES):
            report.add(cid, "subj_eq_obj", f"subj=obj={subj!r}", "error")

    if len(obj) > 185:
        report.add(cid, "long_object", f"len={len(obj)}", "error")

    if " " in pred:
        report.add(cid, "multiword_predicate", pred, "error")

    # --- span grounding ---
    if subj and subj.lower() not in text.lower():
        report.add(cid, "subject_not_in_text", subj, "error")
    if ctext and norm(ctext) not in norm(text):
        report.add(cid, "condition_not_in_text", ctext[:80], "error")

    # --- lead-frame role (smarter than whole-text) ---
    fsent = frame_sentence(text, subj, pred)
    exp_role = expected_role_for_frame(fsent, pred, phen)
    if exp_role and role not in {exp_role, "other", "indemnifying_party"}:
        if not (exp_role == "obligor" and role == "indemnifying_party"):
            report.add(
                cid,
                "role_frame",
                f"gold role={role}, expected={exp_role} on frame: {fsent[:120]}...",
                "error",
            )

    # --- if-then main action inversion ---
    if IN_EVENT_THEN_RE.search(text) or IF_THEN_RE.search(text):
        then_m = THEN_MAIN_RE.search(text)
        if then_m:
            main_pred = then_m.group(1).lower()
            if pred != main_pred and pred in norm(text.split("then")[0] if "then" in text.lower() else ""):
                # gold predicate appears in if-clause but not as main then-predicate
                if main_pred in TERM_PREDICATES or main_pred in (
                    "renew", "terminate", "grant", "deliver", "pay", "assign",
                ):
                    report.add(
                        cid,
                        "if_then_inversion",
                        f"gold pred={pred} from if-clause; main then-pred={main_pred}",
                        "error",
                        suggested=fix_multi_shall_primary(text, triplet),
                    )

    # --- auto-renew should use renew predicate ---
    if AUTO_RENEW_RE.search(text) and pred not in ("renew", "mean", "govern", "construe", "interpret"):
        if not DEFINITION_RE.search(text.strip()):
            report.add(cid, "auto_renew_frame", f"pred={pred}, expected renew", "error")

    # --- definition consistency ---
    is_def_text = bool(DEFINITION_RE.search(text.strip()) or re.search(r'\bshall mean\b', text, re.I))
    if phen.get("is_definition") and not is_def_text:
        report.add(cid, "phenomenon_definition", "is_definition=true but no shall mean", "error")
    if is_def_text and pred != "mean":
        report.add(cid, "definition_predicate", f"pred={pred}, expected mean", "error")
    if is_def_text and role != "other":
        report.add(cid, "definition_role", f"role={role}, expected other", "error")

    # --- passive agent ---
    if phen.get("passive"):
        by_m = PASSIVE_BY.search(text)
        passive_m = MAY_BE_PASSIVE.search(text) or SHALL_BE_PASSIVE.search(text)
        if passive_m and by_m:
            agent = by_m.group(1).strip()
            # If gold predicate matches passive verb and subject should be agent
            pverb = passive_m.group(1).lower()
            if pred == pverb or pred + "e" == pverb or pred.rstrip("e") + "ed" == pverb:
                if norm(subj) != norm(agent) and agent.lower() not in subj.lower():
                    report.add(
                        cid,
                        "passive_agent",
                        f"gold subj={subj!r}, by-phrase agent={agent!r}",
                        "warn",
                    )

    # --- manner in condition (guideline violation) ---
    if ctext and MANNER_IN_COND.search(ctext):
        if not re.search(r"\b(if|unless|except|provided that|in the event)\b", ctext, re.I):
            report.add(cid, "manner_condition", ctext[:100], "error")

    if ctext and SUBJECT_TO_BOILER.match(ctext):
        report.add(cid, "procedural_condition", "Subject to boilerplate in condition", "error")

    # --- permission without consent should be right_holder ---
    if PERM_WITHOUT_CONSENT.search(text):
        sug = fix_permission_without_consent(text, triplet)
        if role != sug["subject"]["role"] or pred != sug["action"]["predicate"]:
            report.add(
                cid,
                "permission_without_consent",
                f"gold role={role} pred={pred}; expected right_holder {sug['action']['predicate']}",
                "error",
                suggested=sug,
            )

    # --- neither party may assign ---
    if NEITHER_MAY.search(text) and role not in ("prohibited_party", "other"):
        report.add(cid, "neither_party_may", f"role={role}, expected prohibited_party", "error")

    # --- rule pipeline would fix role ---
    fixed_role = fix_role_modality(text, triplet)
    fixed_subj = (fixed_role.get("subject") or {}).get("role", "")
    if fixed_subj and fixed_subj != role:
        # only if lead-frame agrees with fix_role_modality
        if exp_role and fixed_subj == exp_role:
            report.add(cid, "role_fix_pipeline", f"{role} -> {fixed_subj}", "error")

    # --- definition triplet template ---
    if is_def_text:
        def_t = fix_definition_triplet(text, triplet)
        if json.dumps(def_t, sort_keys=True) != json.dumps(triplet, sort_keys=True):
            ds, da, dc = def_t["subject"], def_t["action"], def_t["condition"]
            report.add(
                cid,
                "definition_frame",
                f"expected subj={ds['text'][:40]} pred=mean",
                "error",
                suggested=def_t,
            )


def consolidate(report: AuditReport) -> dict[str, list[Finding]]:
    by_id: dict[str, list[Finding]] = {}
    for f in report.findings:
        by_id.setdefault(f.clause_id, []).append(f)
    return by_id


def main() -> int:
    rows = load_rows()
    report = AuditReport()
    for r in rows:
        audit_record(r, report)

    by_id = consolidate(report)
    errors = {cid: fs for cid, fs in by_id.items() if any(x.severity == "error" for x in fs)}

    print(f"TOTAL RECORDS: {len(rows)}")
    print(f"RECORDS WITH ERRORS: {len(errors)}")
    print()

    for cid in sorted(errors, key=lambda x: (errors[x][0].category, x)):
        fs = errors[cid]
        r = next(x for x in rows if x["clause_id"] == cid)
        t = r["triplet"]
        print("=" * 88)
        print(f"{cid} | {r['text'][:160]}...")
        print(
            f"GOLD: ({t['subject']['role']}) {t['subject']['text']} | "
            f"{t['action']['predicate']} | {t['action']['object'][:60]}..."
            if t["action"]["object"]
            else f"GOLD: ({t['subject']['role']}) {t['subject']['text']} | {t['action']['predicate']} | (empty obj)"
        )
        c = t["condition"]
        if c.get("text"):
            print(f"COND: [{c['type']}] {c['text'][:100]}")
        cats = sorted({f.category for f in fs})
        print(f"ISSUES ({', '.join(cats)}):")
        for f in fs:
            if f.severity == "error":
                print(f"  - [{f.category}] {f.detail}")
        print()

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
