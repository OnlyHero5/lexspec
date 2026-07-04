#!/usr/bin/env python3
"""Comprehensive LexSpec-500 quality validator (target score >= 95/100).

Extends validate_testset_500.py with annotation-frame checks, phenomenon
quotas, role-modality heuristics, and a weighted quality score estimate.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/processed/gold_triplets_500.jsonl"
TEST = ROOT / "data/processed/gold_testset_500.jsonl"

sys.path.insert(0, str(ROOT / "scripts"))
from build_gold_500 import GOVERN_RE, is_bad_text  # noqa: E402

PHENOMENA_KEYS = (
    "passive",
    "conditional",
    "relative_clause",
    "long_distance",
    "negation",
    "is_definition",
)

EXPECTED_LEN = 500
MIN_PASSIVE = 100
MIN_LONG_DISTANCE = 75
MIN_NEGATION = 75
MIN_RELATIVE = 75
MIN_IS_DEFINITION = 5
MAX_ZERO_PHEN = 5
COND_RATIO_MIN = 0.25
COND_RATIO_MAX = 0.55
COND_RATIO_HARD_MAX = 0.60
TARGET_SCORE = 95

AGREEMENT_NP_RE = re.compile(r"^(This Agreement|this Agreement|the Agreement|The Agreement)\b")
LAWS_NP_RE = re.compile(
    r"\b(?:substantive )?laws of\b|\b(?:state|federal|applicable) law\b|\blaw of\b",
    re.I,
)
PROHIBITION_RE = re.compile(
    r"\b(?:shall\s+not|may\s+not|must\s+not|will\s+not|"
    r"neither\b[^.;]{0,160}?\b(?:shall|may)\b|"
    r"neither\s+party\s+may\b|"
    r"no\s+party\s+shall|"
    r"in\s+no\s+event|"
    r"nothing\s+(?:in\s+)?(?:herein|this\s+Agreement))\b",
    re.I | re.DOTALL,
)
PERMISSION_RE = re.compile(
    r"\b(?:may\b(?!\s+not\b)|is\s+entitled\s+to|has\s+the\s+right\s+to|shall\s+have\s+the\s+right)\b",
    re.I,
)
OBLIGATION_RE = re.compile(
    r"\b(?:shall\b(?!\s+not\b)|must\b(?!\s+not\b)|agrees?\s+to|undertakes?\s+to)\b",
    re.I,
)
DEFINITION_ROLE_OK = frozenset({"other"})
TERM_PREDICATES = frozenset({
    "commence", "begin", "start", "end", "expire", "renew", "continue",
    "terminate", "become", "remain", "extend", "mean", "include", "refer",
})
ROLE_MODALITY_MAP = {
    "obligor": {"obligor", "indemnifying_party", "other"},
    "right_holder": {"right_holder", "other"},
    "prohibited_party": {"prohibited_party", "other"},
}


@dataclass
class QualityReport:
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    per_record: dict[str, list[str]] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    score_breakdown: dict[str, float] = field(default_factory=dict)

    def add(self, clause_id: str, code: str, *, warn: bool = False) -> None:
        self.per_record.setdefault(clause_id, []).append(code)
        if warn:
            self.warnings.append(f"{clause_id}: {code}")
        else:
            self.issues.append(f"{clause_id}: {code}")


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def count_zero_phenomena(rows: list[dict]) -> int:
    return sum(
        1 for r in rows
        if not any((r.get("phenomena") or {}).get(k) for k in PHENOMENA_KEYS)
    )


def phenomenon_counts(rows: list[dict]) -> dict[str, int]:
    return {
        k: sum(1 for r in rows if (r.get("phenomena") or {}).get(k))
        for k in PHENOMENA_KEYS
    }


def expected_role_from_text(text: str) -> str | None:
    """Heuristic expected deontic role from surface modality markers."""
    if PROHIBITION_RE.search(text):
        return "prohibited_party"
    if PERMISSION_RE.search(text):
        return "right_holder"
    if OBLIGATION_RE.search(text):
        return "obligor"
    return None


def check_role_modality(text: str, role: str, pred: str, phen: dict) -> str | None:
    if role in DEFINITION_ROLE_OK and phen.get("is_definition"):
        return None
    if pred in TERM_PREDICATES and role == "other":
        return None
    if GOVERN_RE.search(text) or pred in ("govern", "construe", "interpret"):
        if re.search(r"\bgoverned by the terms\b", text, re.I) or pred == "terminate":
            return None
        return None if role == "other" else "role_modality_govern"
    expected = expected_role_from_text(text)
    if expected is None:
        return None
    allowed = ROLE_MODALITY_MAP.get(expected, {expected, "other"})
    if role not in allowed:
        return f"role_modality:{role}_vs_{expected}"
    return None


def check_govern_frame(text: str, triplet: dict) -> str | None:
    pred = (triplet.get("action") or {}).get("predicate", "")
    subj = ((triplet.get("subject") or {}).get("text") or "").strip()
    if re.search(r"\bgoverned by the terms\b", text, re.I):
        return None
    if pred != "govern" and not GOVERN_RE.search(text):
        return None
    if pred != "govern":
        return "govern_pred_missing"
    if AGREEMENT_NP_RE.match(subj):
        return "govern_agreement_subject"
    if not LAWS_NP_RE.search(subj):
        return "govern_subject_not_laws"
    obj = ((triplet.get("action") or {}).get("object") or "").strip()
    if obj and AGREEMENT_NP_RE.match(obj):
        return None
    if obj and LAWS_NP_RE.search(obj) and not AGREEMENT_NP_RE.search(obj):
        return "govern_object_should_be_agreement"
    return None


def check_condition_consistency(triplet: dict) -> str | None:
    cond = triplet.get("condition") or {}
    ctext = (cond.get("text") or "").strip()
    ctype = (cond.get("type") or "none").strip().lower()
    if ctext and ctype == "none":
        return "cond_text_with_type_none"
    if not ctext and ctype != "none":
        return "cond_type_without_text"
    if ctype not in ("none", "trigger", "temporal", "exception"):
        return f"cond_invalid_type:{ctype}"
    return None


def validate_record(record: dict) -> list[str]:
    codes: list[str] = []
    text = record.get("text", "")
    triplet = record.get("triplet") or {}
    phen = record.get("phenomena") or {}
    cid = record.get("clause_id", "?")

    if is_bad_text(text):
        codes.append("bad_text")

    pred = (triplet.get("action") or {}).get("predicate", "")
    role = (triplet.get("subject") or {}).get("role", "")

    if " " in pred:
        codes.append("multiword_predicate")

    gov_issue = check_govern_frame(text, triplet)
    if gov_issue:
        codes.append(gov_issue)

    cond_issue = check_condition_consistency(triplet)
    if cond_issue:
        codes.append(cond_issue)

    rm_issue = check_role_modality(text, role, pred, phen)
    if rm_issue:
        codes.append(rm_issue)

    subj = ((triplet.get("subject") or {}).get("text") or "").strip()
    obj = ((triplet.get("action") or {}).get("object") or "").strip()
    if subj and obj and subj.lower() == obj.lower() and pred not in TERM_PREDICATES:
        codes.append("subj_eq_obj")

    return codes


def quota_score(actual: float, target: float, *, higher_is_better: bool = True) -> float:
    if higher_is_better:
        if actual >= target:
            return 1.0
        return max(0.0, actual / target) if target else 0.0
    if actual <= target:
        return 1.0
    return max(0.0, target / actual) if actual else 1.0


def ratio_in_band(ratio: float, lo: float, hi: float) -> float:
    if lo <= ratio <= hi:
        return 1.0
    if ratio > COND_RATIO_HARD_MAX:
        return 0.0
    if ratio < lo:
        return max(0.0, ratio / lo)
    return max(0.0, 1.0 - (ratio - hi) / max(hi, 1e-9))


def compute_score(rows: list[dict], report: QualityReport) -> float:
    n = len(rows)
    phen = report.metrics["phenomena"]
    cond_ratio = phen["conditional"] / n if n else 0.0
    zp = report.metrics["zero_phenomena"]

    breakdown = {
        "structure": 0.0,
        "phenomena": 0.0,
        "annotation": 0.0,
        "text": 0.0,
    }

    structure_pts = 0.0
    structure_pts += 5.0 if n == EXPECTED_LEN else 5.0 * quota_score(n, EXPECTED_LEN)
    structure_pts += 5.0 if report.metrics["unique_clause_ids"] == n else 0.0
    structure_pts += 5.0 if report.metrics["unique_texts"] == n else 0.0
    breakdown["structure"] = structure_pts

    phen_pts = 0.0
    phen_pts += 8.0 * quota_score(phen["passive"], MIN_PASSIVE)
    phen_pts += 8.0 * ratio_in_band(cond_ratio, COND_RATIO_MIN, COND_RATIO_MAX)
    phen_pts += 7.0 * quota_score(phen["long_distance"], MIN_LONG_DISTANCE)
    phen_pts += 7.0 * quota_score(phen["negation"], MIN_NEGATION)
    phen_pts += 7.0 * quota_score(phen["relative_clause"], MIN_RELATIVE)
    phen_pts += 4.0 * quota_score(phen["is_definition"], MIN_IS_DEFINITION)
    phen_pts += 4.0 * quota_score(zp, MAX_ZERO_PHEN, higher_is_better=False)
    breakdown["phenomena"] = phen_pts

    gov_violations = sum(
        1 for codes in report.per_record.values()
        if any(c.startswith("govern_") for c in codes)
    )
    multiword = sum(1 for codes in report.per_record.values() if "multiword_predicate" in codes)
    cond_bad = sum(
        1 for codes in report.per_record.values()
        if any(c.startswith("cond_") for c in codes)
    )
    role_bad = sum(
        1 for codes in report.per_record.values()
        if any(c.startswith("role_modality") for c in codes)
    )

    annotation_pts = 30.0
    annotation_pts -= min(8.0, gov_violations * 1.5)
    annotation_pts -= min(6.0, multiword * 2.0)
    annotation_pts -= min(6.0, cond_bad * 1.0)
    annotation_pts -= min(10.0, role_bad * 0.5)
    annotation_pts = max(0.0, annotation_pts)
    breakdown["annotation"] = annotation_pts

    bad_text_n = report.metrics["bad_text_count"]
    breakdown["text"] = 10.0 if bad_text_n == 0 else max(0.0, 10.0 - bad_text_n * 2.0)

    report.score_breakdown = breakdown
    return round(sum(breakdown.values()), 1)


def validate(rows: list[dict]) -> QualityReport:
    report = QualityReport()
    n = len(rows)
    report.metrics["count"] = n

    ids = [r.get("clause_id") for r in rows]
    texts = [r.get("text", "") for r in rows]
    report.metrics["unique_clause_ids"] = len(set(ids))
    report.metrics["unique_texts"] = len(set(texts))

    if n != EXPECTED_LEN:
        report.issues.append(f"count={n}, expected {EXPECTED_LEN}")
    if len(set(ids)) != n:
        dup = [k for k, v in Counter(ids).items() if v > 1]
        report.issues.append(f"duplicate clause_ids: {dup[:10]}")
    if len(set(texts)) != n:
        dup_t = sum(1 for _, c in Counter(texts).items() if c > 1)
        report.issues.append(f"duplicate texts: {dup_t} groups")

    phen = phenomenon_counts(rows)
    zp = count_zero_phenomena(rows)
    report.metrics["phenomena"] = phen
    report.metrics["zero_phenomena"] = zp
    report.metrics["conditional_ratio"] = round(phen["conditional"] / n, 4) if n else 0.0

    if phen["passive"] < MIN_PASSIVE:
        report.issues.append(f"passive={phen['passive']}, need >={MIN_PASSIVE}")
    if phen["long_distance"] < MIN_LONG_DISTANCE:
        report.issues.append(f"long_distance={phen['long_distance']}, need >={MIN_LONG_DISTANCE}")
    if phen["negation"] < MIN_NEGATION:
        report.issues.append(f"negation={phen['negation']}, need >={MIN_NEGATION}")
    if phen["relative_clause"] < MIN_RELATIVE:
        report.issues.append(f"relative_clause={phen['relative_clause']}, need >={MIN_RELATIVE}")
    if phen["is_definition"] < MIN_IS_DEFINITION:
        report.issues.append(f"is_definition={phen['is_definition']}, need >={MIN_IS_DEFINITION}")
    if zp > MAX_ZERO_PHEN:
        report.issues.append(f"zero_phenomena={zp}, max {MAX_ZERO_PHEN}")

    cr = report.metrics["conditional_ratio"]
    if cr < COND_RATIO_MIN or cr > COND_RATIO_MAX:
        report.issues.append(
            f"conditional_ratio={cr:.3f}, expected [{COND_RATIO_MIN}, {COND_RATIO_MAX}]"
        )
    if cr > COND_RATIO_HARD_MAX:
        report.issues.append(f"conditional_ratio={cr:.3f}, exceeds hard cap {COND_RATIO_HARD_MAX}")

    bad_text_n = 0
    for record in rows:
        cid = record.get("clause_id", "?")
        codes = validate_record(record)
        if "bad_text" in codes:
            bad_text_n += 1
        for code in codes:
            report.add(cid, code)
    report.metrics["bad_text_count"] = bad_text_n

    report.metrics["govern_violations"] = sum(
        1 for codes in report.per_record.values()
        if any(c.startswith("govern_") for c in codes)
    )
    report.metrics["multiword_predicates"] = sum(
        1 for codes in report.per_record.values() if "multiword_predicate" in codes
    )
    report.metrics["condition_inconsistencies"] = sum(
        1 for codes in report.per_record.values()
        if any(c.startswith("cond_") for c in codes)
    )
    report.metrics["role_modality_mismatches"] = sum(
        1 for codes in report.per_record.values()
        if any(c.startswith("role_modality") for c in codes)
    )
    report.metrics["records_with_issues"] = len(report.per_record)

    report.metrics["quality_score"] = compute_score(rows, report)
    report.metrics["target_score"] = TARGET_SCORE
    report.metrics["passed"] = report.metrics["quality_score"] >= TARGET_SCORE
    return report


def print_report(source: Path, report: QualityReport) -> None:
    m = report.metrics
    phen = m["phenomena"]
    print(f"Source: {source.relative_to(ROOT)}")
    print(f"count={m['count']} unique_ids={m['unique_clause_ids']} unique_texts={m['unique_texts']}")
    print(
        f"phenomena: passive={phen['passive']} conditional={phen['conditional']} "
        f"({m['conditional_ratio']:.1%}) relative={phen['relative_clause']} "
        f"long_distance={phen['long_distance']} negation={phen['negation']} "
        f"is_definition={phen['is_definition']}"
    )
    print(f"zero_phenomena={m['zero_phenomena']} (max {MAX_ZERO_PHEN})")
    print(
        f"annotation: govern_violations={m['govern_violations']} "
        f"multiword_pred={m['multiword_predicates']} "
        f"cond_inconsistencies={m['condition_inconsistencies']} "
        f"role_modality_mismatches={m['role_modality_mismatches']}"
    )
    print(f"bad_text={m['bad_text_count']} records_with_issues={m['records_with_issues']}")
    print()
    print("Score breakdown:")
    for key, pts in report.score_breakdown.items():
        print(f"  {key}: {pts:.1f}")
    print(f"QUALITY SCORE: {m['quality_score']:.1f}/100 (target >= {TARGET_SCORE})")

    if report.issues:
        print("\nQuota / structural issues:")
        for item in report.issues:
            print(f"  - {item}")

    sample = sorted(report.per_record.items(), key=lambda x: len(x[1]), reverse=True)[:12]
    if sample:
        print("\nSample per-record issues:")
        for cid, codes in sample:
            print(f"  {cid}: {', '.join(codes[:6])}")


def main() -> int:
    rows = load_jsonl(GOLD)
    source = GOLD
    if not rows:
        rows = load_jsonl(TEST)
        source = TEST

    if not rows:
        print("FAIL: no gold_triplets_500.jsonl or gold_testset_500.jsonl")
        return 1

    report = validate(rows)
    print_report(source, report)

    if report.metrics["passed"]:
        print("\nPASS (quality >= 95 and quotas met)")
        return 0

    print("\nFAIL (quality below target or quota breach)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
