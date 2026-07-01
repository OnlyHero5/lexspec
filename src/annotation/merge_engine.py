"""
Merge engine for the annotation pipeline.

Reconciles dual-model annotations using field-level consensus voting,
augmented by cross-model review results to automatically resolve
disagreements where possible.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

from src.extraction.schema import LegalTriplet
from src.annotation.cli_utils import (
    dict_to_triplet, triplet_to_dict,
    GEMMA_ANNOT, QWEN_ANNOT, QWEN_REVIEW_GEMMA, GEMMA_REVIEW_QWEN,
    GOLD_OUT, DISAGREE_OUT,
)
from src.annotation.consensus import field_level_consensus
from src.annotation.normalization import normalize_text
from src.utils.io import read_jsonl, write_jsonl, ensure_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _index_by_id(records: List[dict]) -> Dict[str, dict]:
    """Index a list of records by clause_id."""
    return {r["clause_id"]: r for r in records if r.get("clause_id")}


def _field_value(t: LegalTriplet, field_name: str) -> str:
    """Extract the string value of a named field from a LegalTriplet."""
    mapping = {
        "subject.text": t.subject.text,
        "subject.role": t.subject.role.value,
        "action.predicate": t.action.predicate,
        "action.object": t.action.object,
        "condition.text": t.condition.text,
        "condition.type": t.condition.type.value,
    }
    return str(mapping.get(field_name, ""))


def _set_field(t: LegalTriplet, field_name: str, value: str) -> None:
    """Set a named field on a LegalTriplet."""
    from src.extraction.schema import LegalRole, ConditionType
    if field_name == "subject.text":
        t.subject.text = value
    elif field_name == "subject.role":
        t.subject.role = LegalRole(value)
    elif field_name == "action.predicate":
        t.action.predicate = value
    elif field_name == "action.object":
        t.action.object = value
    elif field_name == "condition.text":
        t.condition.text = value
    elif field_name == "condition.type":
        t.condition.type = ConditionType(value)


def _resolve_with_reviews(
    consensus: LegalTriplet,
    disagreements: List[dict],
    review_records: List[dict],
    reviewer_annotations: Dict[str, LegalTriplet],
) -> tuple[LegalTriplet, List[dict]]:
    """Use cross-review results to automatically resolve annotation disagreements.

    For each disagreement, try in order:
      1. If the reviewer rejected the field, adopt the reviewer's correction.
      2. If the reviewer's own annotation agrees with one side, adopt that side's value.

    Args:
        consensus:            Current consensus triplet (may be modified).
        disagreements:        List of field-level disagreements.
        review_records:       List of review records.
        reviewer_annotations: Reviewer's own independent annotations (clause_id -> LegalTriplet).

    Returns:
        (updated consensus triplet, still-unresolved disagreements list).
    """
    by_id = _index_by_id(review_records)
    resolved_notes: List[dict] = []
    c = consensus.model_copy(deep=True)
    remaining: List[dict] = []

    for dis in disagreements:
        field = dis["field"]
        clause_id = dis.get("clause_id", "")
        resolved = False

        # Strategy 1: Reviewer explicitly rejected the field -> adopt correction.
        rev = by_id.get(clause_id)
        if rev and rev.get("success") and rev.get("corrected_triplet"):
            corrected = dict_to_triplet(rev["corrected_triplet"])
            rejected_fields = {
                fj["field"]
                for fj in (rev.get("field_judgments") or [])
                if fj.get("judgment") == "reject"
            }
            if field in rejected_fields or rev.get("verdict") in ("partial", "reject"):
                val = _field_value(corrected, field)
                _set_field(c, field, val)
                dis = {
                    **dis,
                    "resolved": True,
                    "resolved_by": rev.get("reviewer_role", "review"),
                    "resolution": val,
                }
                resolved = True
                resolved_notes.append({
                    "clause_id": clause_id, "field": field, "via": "review correction"
                })

        # Strategy 2: Reviewer's own annotation agrees with one side -> adopt it.
        if not resolved and clause_id in reviewer_annotations:
            own = reviewer_annotations[clause_id]
            a_norm = normalize_text(dis["anno_a_value"])
            b_norm = normalize_text(dis["anno_b_value"])
            own_val = _field_value(own, field)
            is_text_field = (
                "text" in field or "predicate" in field or "object" in field
            )
            own_norm = normalize_text(own_val) if is_text_field else own_val
            if own_norm == a_norm:
                final_val = (
                    dis["anno_a_value"]
                    if "role" not in field and "type" not in field
                    else own_val
                )
                _set_field(c, field, final_val)
                dis = {
                    **dis,
                    "resolved": True,
                    "resolved_by": "reviewer own annotation",
                    "resolution": own_val,
                }
                resolved = True
            elif own_norm == b_norm:
                final_val = (
                    dis["anno_b_value"]
                    if "role" not in field and "type" not in field
                    else own_val
                )
                _set_field(c, field, final_val)
                dis = {
                    **dis,
                    "resolved": True,
                    "resolved_by": "reviewer own annotation",
                    "resolution": own_val,
                }
                resolved = True

        if not dis.get("resolved"):
            remaining.append(dis)

    return c, remaining


def cmd_merge(args) -> None:
    """Merge dual-model annotations and cross-review results into gold standard.

    Process:
      1. For each shared clause, field-level voting produces preliminary consensus.
      2. Cross-review results are used to automatically resolve disagreements.
      3. Unresolved disagreements are marked as needs_human_review.
      4. Output gold file and human-review file.
    """
    qwen_path = Path(args.qwen or QWEN_ANNOT)
    gemma_path = Path(args.gemma or GEMMA_ANNOT)
    qwen_rev_path = Path(args.qwen_review or QWEN_REVIEW_GEMMA)
    gemma_rev_path = Path(args.gemma_review or GEMMA_REVIEW_QWEN)
    gold_path = Path(args.output or GOLD_OUT)
    disagree_path = Path(args.disagreements or DISAGREE_OUT)

    # Pre-check: both annotation files must exist.
    if not qwen_path.exists() or not gemma_path.exists():
        logger.error("Both Qwen and Gemma annotation files are required.")
        logger.error(
            "  qwen:  %s (%s)",
            qwen_path, "exists" if qwen_path.exists() else "missing",
        )
        logger.error(
            "  gemma: %s (%s)",
            gemma_path, "exists" if gemma_path.exists() else "missing",
        )
        sys.exit(1)

    # Load all data.
    qwen_recs = _index_by_id(read_jsonl(str(qwen_path)))
    gemma_recs = _index_by_id(read_jsonl(str(gemma_path)))
    qwen_rev = read_jsonl(str(qwen_rev_path)) if qwen_rev_path.exists() else []
    gemma_rev = read_jsonl(str(gemma_rev_path)) if gemma_rev_path.exists() else []

    # Only process clauses annotated by both models.
    common_ids = sorted(set(qwen_recs) & set(gemma_recs))
    logger.info("Merging %d clauses annotated by both models", len(common_ids))

    gold_records: List[dict] = []
    human_review: List[dict] = []

    # Build reviewer own-annotation indices (for auto-resolution).
    qwen_own = {
        cid: dict_to_triplet(r["triplet"])
        for cid, r in qwen_recs.items()
        if r.get("success")
    }
    gemma_own = {
        cid: dict_to_triplet(r["triplet"])
        for cid, r in gemma_recs.items()
        if r.get("success")
    }

    for cid in common_ids:
        qr = qwen_recs[cid]
        gr = gemma_recs[cid]

        # Either side failed -> mark for human review.
        if not qr.get("success") or not gr.get("success"):
            human_review.append({"clause_id": cid, "reason": "annotation failed"})
            continue

        q_tri = dict_to_triplet(qr["triplet"])
        g_tri = dict_to_triplet(gr["triplet"])

        # Step 1: Field-level consensus.
        consensus, disagreements = field_level_consensus(q_tri, g_tri)
        for d in disagreements:
            d["clause_id"] = cid

        # Step 2: Auto-resolve using cross-reviews.
        # Qwen reviews Gemma -> can resolve fields where Gemma was wrong.
        consensus, remaining = _resolve_with_reviews(
            consensus, disagreements, qwen_rev, qwen_own,
        )
        # Gemma reviews Qwen -> can resolve fields where Qwen was wrong.
        consensus, remaining = _resolve_with_reviews(
            consensus, remaining, gemma_rev, gemma_own,
        )

        gold_records.append({
            "clause_id": cid,
            "text": qr.get("text") or gr.get("text"),
            "phenomena": qr.get("phenomena") or gr.get("phenomena", {}),
            "triplet": triplet_to_dict(consensus),
            "qwen_triplet": qr["triplet"],
            "gemma_triplet": gr["triplet"],
            "unresolved_disagreements": remaining,
            "needs_human_review": len(remaining) > 0,
        })
        if remaining:
            human_review.append({
                "clause_id": cid,
                "text": qr.get("text"),
                "disagreements": remaining,
            })

    # Write outputs.
    ensure_dir(gold_path.parent)
    write_jsonl(str(gold_path), gold_records)
    write_jsonl(str(disagree_path), human_review)

    n_human = sum(1 for g in gold_records if g.get("needs_human_review"))
    print(f"\nGold file: {gold_path} ({len(gold_records)} clauses)")
    print(f"Needs human review: {disagree_path} ({n_human} clauses)")
