#!/usr/bin/env python3
"""Re-detect language phenomena for the curated 500-item gold set.

Input (required):
  data/processed/curated_500/stage2_annotations.jsonl

Also loads stage1_pool.jsonl for long_distance proxy preservation on swapped
records (ld_boost / zero_phenomena / duplicate_text).

Output:
  data/processed/curated_500/stage3_phenomena.jsonl

Runs ``detect_phenomena`` with StanzaParser, then syncs ``conditional``,
``negation``, and ``is_definition`` from triplet/text rules while keeping
passive/relative from the detector. Preserves stage-1 long_distance proxy tags
when Stanza MDD misses them.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.corpus.phenomena_detector import detect_phenomena
from src.extraction.schema import DependencyTree
from src.linguistic.ud_features._clause import _matches_marker
from src.utils.config import 加载模型配置, 构建Stanza解析器
from src.utils.constraints import get_validation_thresholds, load_constraints_config

CURATED_DIR = ROOT / "data/processed/curated_500"
STAGE2 = CURATED_DIR / "stage2_annotations.jsonl"
STAGE1 = CURATED_DIR / "stage1_pool.jsonl"
OUTPUT = CURATED_DIR / "stage3_phenomena.jsonl"

PHENOMENA_KEYS = (
    "passive",
    "conditional",
    "relative_clause",
    "long_distance",
    "negation",
    "is_definition",
)
LD_PRESERVE_SWAP_REASONS = frozenset({"ld_boost", "zero_phenomena", "duplicate_text"})

# Lexical conditional triggers (excludes temporal-only markers like upon/within/when).
LEXICAL_TRIGGER_RE = re.compile(
    r"(?<!\w)(?:"
    r"if|unless|provided\s+that|subject\s+to|"
    r"in\s+the\s+event(?:\s+(?:that|of))?|"
    r"so\s+long\s+as|conditioned\s+upon|on\s+condition\s+that|"
    r"except(?:\s+as|\s+that|\s+where|\s+when)?|other\s+than|notwithstanding"
    r")(?!\w)",
    re.I,
)

DEONTIC_NEG_RE = re.compile(
    r"\b(?:"
    r"shall\s+not|may\s+not|must\s+not|will\s+not|"
    r"agrees?\s+that\s+it\s+will\s+not|"
    r"nothing\s+in\s+this\s+Agreement\s+shall|"
    r"neither\b[^.;]{0,160}?\bshall\b|"
    r"no\s+party\s+shall|"
    r"in\s+no\s+event|"
    r"nothing\s+(?:in\s+)?(?:herein|this\s+Agreement)"
    r")\b",
    re.I | re.DOTALL,
)
WITHOUT_CONSENT_RE = re.compile(
    r"\bwithout\s+(?:the\s+)?(?:prior\s+)?(?:written\s+)?(?:express\s+)?(?:approval|consent)\b",
    re.I,
)
DEFINITION_QUOTED_MEAN_RE = re.compile(r'"[^"]+"\s+shall\s+mean\b', re.I)
DEFINITION_FOR_PURPOSES_RE = re.compile(
    r"\bFor\s+purposes\s+of\b[^.;]{0,200}?\bshall\s+mean\b",
    re.I | re.DOTALL,
)


def resolve_input_path() -> Path:
    if not STAGE2.is_file():
        raise FileNotFoundError(f"Required input missing: {STAGE2.relative_to(ROOT)}")
    return STAGE2


def load_stage1_lookup() -> dict[str, dict]:
    if not STAGE1.is_file():
        return {}
    lookup: dict[str, dict] = {}
    for line in STAGE1.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        lookup[rec["clause_id"]] = rec
    return lookup


def phen_count(phenomena: dict) -> int:
    return sum(1 for key in PHENOMENA_KEYS if phenomena.get(key))


def has_lexical_trigger(text: str) -> bool:
    return bool(text.strip() and LEXICAL_TRIGGER_RE.search(text))


def load_trigger_exception_markers(constraints: dict) -> list[str]:
    """Trigger/exception mark words from constraints.yaml (excludes temporal)."""
    markers: list[str] = []
    section = constraints.get("condition_markers") or {}
    for category in ("trigger", "exception"):
        category_data = section.get(category) or {}
        if isinstance(category_data, dict):
            words = category_data.get("mark_words") or []
        elif isinstance(category_data, list):
            words = category_data
        else:
            words = []
        markers.extend(word.lower() for word in words)
    return markers


def has_conditional_advcl_mark(tree: DependencyTree, markers: list[str]) -> bool:
    """True when an advcl subordinate is introduced by a trigger/exception mark."""
    for mark_token in tree.find_tokens_by_deprel("mark"):
        head = tree.get_token(mark_token.head)
        if head is None or head.deprel != "advcl":
            continue
        if _matches_marker(mark_token.text.lower().strip(), markers):
            return True
    return False


def sync_conditional(
    phenomena: dict,
    triplet: dict,
    text: str,
    *,
    has_conditional_advcl: bool,
) -> None:
    """Set conditional=true only for triggers/exceptions, not temporal-only scope."""
    cond = triplet.get("condition") or {}
    ctext = (cond.get("text") or "").strip()
    ctype = (cond.get("type") or "none").strip().lower()

    # Lexical triggers apply to annotated condition text only — not bare "Subject to
    # Section X" in the lead sentence unless extracted as a condition span.
    lexical = bool(ctext and has_lexical_trigger(ctext))

    phenomena["conditional"] = (
        has_conditional_advcl
        or lexical
        or ctype in ("trigger", "exception")
    )


def sync_is_definition(text: str, triplet: dict, detected: bool) -> bool:
    """True for definition clauses from predicate or definitional frames."""
    pred = ((triplet.get("action") or {}).get("predicate") or "").strip().lower()
    if pred == "mean":
        return True
    if DEFINITION_QUOTED_MEAN_RE.search(text):
        return True
    if DEFINITION_FOR_PURPOSES_RE.search(text):
        return True
    if re.search(r'\b(?:shall\s+)?means?\b', text, re.I) and re.search(
        r'(?:^|\s)["\']?\w+["\']?\s+(?:shall\s+)?mean',
        text,
        re.I,
    ):
        return True
    return text_is_definition(text) or detected


def text_is_definition(text: str) -> bool:
    """Text-only definition heuristic when Stanza detection fails."""
    text_lower = text.lower()
    if DEFINITION_QUOTED_MEAN_RE.search(text):
        return True
    if DEFINITION_FOR_PURPOSES_RE.search(text):
        return True
    if re.search(r"\bmeans\b", text_lower):
        if text.strip().startswith('"') or text.strip()[0:1].isdigit():
            return True
        if re.search(r'["\']?\w+["\']?\s+(?:shall\s+)?means?\b', text_lower):
            return True
    if re.match(r'^\d+\.\d+\s+["\']', text.strip()):
        return True
    return False


def sync_negation(text: str, detector_neg: bool) -> bool:
    """True for deontic prohibitions; ignore bare without-consent inline frames."""
    if DEONTIC_NEG_RE.search(text):
        return True
    if WITHOUT_CONSENT_RE.search(text):
        return False
    return detector_neg


def merge_phenomena(
    detected: dict,
    triplet: dict,
    text: str,
    *,
    prior_phen: dict | None,
    stage1_phen: dict | None,
    swap_reason: str | None,
    has_conditional_advcl: bool,
) -> dict:
    phenomena = dict(detected)
    sync_conditional(phenomena, triplet, text, has_conditional_advcl=has_conditional_advcl)
    phenomena["negation"] = sync_negation(text, bool(detected.get("negation", False)))
    phenomena["is_definition"] = sync_is_definition(
        text,
        triplet,
        bool(detected.get("is_definition", False)),
    )

    prior_ld = bool((prior_phen or {}).get("long_distance"))
    stage1_ld = bool((stage1_phen or {}).get("long_distance"))
    if swap_reason in LD_PRESERVE_SWAP_REASONS and stage1_ld:
        preserve_ld = True
    else:
        preserve_ld = False
    phenomena["long_distance"] = bool(
        detected.get("long_distance") or prior_ld or stage1_ld or preserve_ld
    )

    if swap_reason and phen_count(phenomena) == 0 and stage1_ld:
        phenomena["long_distance"] = True

    # Preserve non-conditional structural flags from stage-1 / detector.
    for key in ("passive", "relative_clause", "negation", "long_distance", "is_definition"):
        if not phenomena.get(key):
            if (stage1_phen or {}).get(key):
                phenomena[key] = True
            elif detected.get(key):
                phenomena[key] = True

    return phenomena


def count_phenomena(records: list[dict]) -> Counter:
    counts: Counter = Counter()
    for rec in records:
        phen = rec.get("phenomena") or {}
        for key in PHENOMENA_KEYS:
            if phen.get(key):
                counts[key] += 1
    return counts


def print_counts_table(counts: Counter, total: int) -> None:
    rows = [
        ("passive", counts.get("passive", 0)),
        ("conditional", counts.get("conditional", 0)),
        ("relative_clause", counts.get("relative_clause", 0)),
        ("long_distance", counts.get("long_distance", 0)),
        ("negation", counts.get("negation", 0)),
        ("is_definition", counts.get("is_definition", 0)),
    ]
    print("\nPhenomenon counts (stage3_phenomena.jsonl)")
    print(f"{'phenomenon':<18} {'count':>6} {'pct':>8}")
    print("-" * 34)
    for name, count in rows:
        pct = 100.0 * count / total if total else 0.0
        print(f"{name:<18} {count:>6} {pct:>7.1f}%")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    input_path = args.input if args.input else resolve_input_path()
    output_path = args.output if args.output else OUTPUT
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    constraints = load_constraints_config(str(ROOT / "configs/constraints.yaml"))
    thresholds = get_validation_thresholds(constraints)
    long_distance_mdd = thresholds["long_distance_mdd"]

    model_config = 加载模型配置(str(ROOT / "configs/model.yaml"))
    parser = 构建Stanza解析器(model_config)
    conditional_markers = load_trigger_exception_markers(constraints)

    stage1_lookup = load_stage1_lookup()
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise SystemExit(f"Input file is empty: {input_path}")

    out_records: list[dict] = []
    parse_failures = 0

    for i, rec in enumerate(records, start=1):
        text = rec["text"]
        triplet = rec.get("triplet") or {}
        clause_id = rec.get("clause_id", str(i))
        prior_phen = rec.get("phenomena") or {}
        stage1_rec = stage1_lookup.get(clause_id, {})
        stage1_phen = stage1_rec.get("phenomena") or {}
        swap_reason = rec.get("swap_reason")
        has_conditional_advcl = False
        try:
            tree = parser.parse(text)
            detected = detect_phenomena(tree, long_distance_mdd=long_distance_mdd)
            has_conditional_advcl = has_conditional_advcl_mark(tree, conditional_markers)
        except Exception as exc:
            parse_failures += 1
            detected = dict(prior_phen)
            detected["is_definition"] = sync_is_definition(text, triplet, False)
            print(f"WARN parse failed {clause_id}: {exc}", file=sys.stderr)

        updated = dict(rec)
        updated["phenomena"] = merge_phenomena(
            detected,
            triplet,
            text,
            prior_phen=prior_phen,
            stage1_phen=stage1_phen,
            swap_reason=swap_reason,
            has_conditional_advcl=has_conditional_advcl,
        )
        out_records.append(updated)

        if i % 50 == 0:
            print(f"Processed {i}/{len(records)}", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in out_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    counts = count_phenomena(out_records)
    print(
        json.dumps(
            {
                "input": str(input_path.relative_to(ROOT)),
                "output": str(output_path.relative_to(ROOT)),
                "records": len(out_records),
                "long_distance_mdd": long_distance_mdd,
                "parse_failures": parse_failures,
                "phenomenon_counts": dict(counts),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print_counts_table(counts, len(out_records))


if __name__ == "__main__":
    main()
