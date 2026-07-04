"""消融实验不变量：Ours-Dep 在 VALID 时必须与 baseline 初抽一致。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from experiments.step_04_extract_dep import 运行Dep流水线
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


def _triplet(subject_text: str = "Seller") -> LegalTriplet:
    return LegalTriplet(
        subject=Subject(text=subject_text, role=LegalRole.OBLIGOR),
        action=Action(predicate="deliver", object="goods"),
        condition=Condition(),
    )


def _write_baseline(path: Path, clause_id: str, triplet: LegalTriplet) -> None:
    row = {
        "clause_id": clause_id,
        "text": "Seller shall deliver goods.",
        "triplet": triplet.model_dump(mode="json"),
        "success": True,
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


class TestAblationDepInvariant:
    def test_valid_status_preserves_baseline_triplet(self, tmp_path: Path):
        baseline_path = tmp_path / "baseline.jsonl"
        baseline_triplet = _triplet("Acme Corp")
        _write_baseline(baseline_path, "C-00001", baseline_triplet)

        test_clauses = [
            {"clause_id": "C-00001", "text": "Seller shall deliver goods."},
        ]

        val_result = ValidationResult(
            status=ValidationStatus.VALID,
            original_prediction=baseline_triplet,
            linguistic_evidence=LinguisticEvidence(),
            feedback="",
        )
        mock_validator = MagicMock()
        mock_validator.validate.return_value = val_result

        mock_parser = MagicMock()
        mock_parser.parse.return_value = MagicMock()

        with patch(
            "experiments.step_04_extract_dep.构建Stanza解析器",
            return_value=mock_parser,
        ), patch(
            "experiments.step_04_extract_dep.ConstraintValidator",
            return_value=mock_validator,
        ), patch(
            "experiments.step_04_extract_dep.加载模型配置",
            return_value={},
        ):
            results, _ = 运行Dep流水线(
                test_clauses=test_clauses,
                baseline_predictions_path=str(baseline_path),
            )

        assert len(results) == 1
        out = LegalTriplet.model_validate(results[0]["triplet"])
        assert out.subject.text == baseline_triplet.subject.text
        assert out.action.predicate == baseline_triplet.action.predicate
        assert results[0]["validation_status"] == "VALID"
        assert results[0]["used_correction"] is False
