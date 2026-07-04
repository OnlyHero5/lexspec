"""clause_id 对齐与 baseline 初抽复用的测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.alignment import (
    ClauseAlignmentError,
    align_predictions_to_gold,
    align_to_gold_order,
    records_to_triplets,
)
from src.evaluation.data_loading import (
    load_baseline_triplet_map,
    load_validations_aligned,
    require_baseline_triplet_map,
    skipped_validation_record,
    validation_result_record,
)
from src.extraction.schema import (
    LegalTriplet,
    Subject,
    Action,
    Condition,
    LegalRole,
    ValidationResult,
    ValidationStatus,
    LinguisticEvidence,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _sample_triplet_dict(text: str = "Seller") -> dict:
    return {
        "subject": {"text": text, "role": "obligor"},
        "action": {"predicate": "deliver", "object": "goods"},
        "condition": {"text": "", "type": "none"},
    }


class TestClauseIdAlignment:
    def test_align_to_gold_order_reorders_predictions(self):
        gold = [
            {"clause_id": "C-001", "triplet": _sample_triplet_dict("A")},
            {"clause_id": "C-002", "triplet": _sample_triplet_dict("B")},
        ]
        preds = [
            {"clause_id": "C-002", "triplet": _sample_triplet_dict("B")},
            {"clause_id": "C-001", "triplet": _sample_triplet_dict("A")},
        ]
        aligned = align_to_gold_order(gold, preds, other_label="predictions", strict=True)
        assert [r["clause_id"] for r in aligned] == ["C-001", "C-002"]

    def test_missing_clause_id_raises(self):
        gold = [{"clause_id": "C-001", "triplet": _sample_triplet_dict()}]
        preds = [{"triplet": _sample_triplet_dict()}]
        with pytest.raises(ClauseAlignmentError, match="missing clause_id"):
            align_predictions_to_gold(gold, preds, strict=True)

    def test_extra_prediction_id_raises_in_strict_mode(self):
        gold = [{"clause_id": "C-001", "triplet": _sample_triplet_dict()}]
        preds = [
            {"clause_id": "C-001", "triplet": _sample_triplet_dict()},
            {"clause_id": "C-999", "triplet": _sample_triplet_dict()},
        ]
        with pytest.raises(ClauseAlignmentError, match="extra clause_id"):
            align_predictions_to_gold(gold, preds, strict=True)


class TestBaselineReuse:
    def test_require_baseline_map_covers_testset(self, tmp_path: Path):
        baseline = tmp_path / "baseline.jsonl"
        _write_jsonl(
            baseline,
            [
                {
                    "clause_id": "C-001",
                    "text": "Seller shall deliver.",
                    "triplet": _sample_triplet_dict(),
                },
            ],
        )
        test_clauses = [{"clause_id": "C-001", "text": "Seller shall deliver."}]
        triplet_map = require_baseline_triplet_map(str(baseline), test_clauses)
        assert "C-001" in triplet_map
        assert isinstance(triplet_map["C-001"], LegalTriplet)

    def test_missing_baseline_clause_raises(self, tmp_path: Path):
        baseline = tmp_path / "baseline.jsonl"
        _write_jsonl(
            baseline,
            [{"clause_id": "C-001", "text": "x", "triplet": _sample_triplet_dict()}],
        )
        test_clauses = [
            {"clause_id": "C-001", "text": "x"},
            {"clause_id": "C-002", "text": "y"},
        ]
        with pytest.raises(ClauseAlignmentError, match="缺少"):
            require_baseline_triplet_map(str(baseline), test_clauses)

    def test_load_baseline_triplet_map(self, tmp_path: Path):
        baseline = tmp_path / "baseline.jsonl"
        _write_jsonl(
            baseline,
            [{"clause_id": "C-001", "triplet": _sample_triplet_dict("Seller")}],
        )
        triplet_map = load_baseline_triplet_map(str(baseline))
        assert triplet_map["C-001"].subject.text == "Seller"


class TestValidationRecords:
    def test_load_validations_aligned_by_clause_id(self, tmp_path: Path):
        gold = tmp_path / "gold.jsonl"
        vals = tmp_path / "vals.jsonl"
        _write_jsonl(
            gold,
            [
                {"clause_id": "C-001", "triplet": _sample_triplet_dict("A")},
                {"clause_id": "C-002", "triplet": _sample_triplet_dict("B")},
            ],
        )
        val_result = ValidationResult(
            status=ValidationStatus.VALID,
            original_prediction=LegalTriplet(
                subject=Subject(text="A", role=LegalRole.OBLIGOR),
                action=Action(predicate="deliver", object="goods"),
                condition=Condition(),
            ),
            linguistic_evidence=LinguisticEvidence(),
            feedback="",
        )
        _write_jsonl(
            vals,
            [
                validation_result_record("C-002", val_result),
                validation_result_record("C-001", val_result),
            ],
        )
        aligned = load_validations_aligned(str(gold), str(vals), strict=True)
        assert len(aligned) == 2
        assert aligned[0].status == ValidationStatus.VALID

    def test_skipped_validation_record(self):
        rec = skipped_validation_record("C-empty")
        assert rec["clause_id"] == "C-empty"
        assert rec["skipped"] is True


class TestRecordsToTriplets:
    def test_records_to_triplets_preserves_order(self):
        records = [
            {"clause_id": "C-001", "triplet": _sample_triplet_dict("A")},
            {"clause_id": "C-002", "triplet": _sample_triplet_dict("B")},
        ]
        triplets = records_to_triplets(records)
        assert triplets[0].subject.text == "A"
        assert triplets[1].subject.text == "B"
