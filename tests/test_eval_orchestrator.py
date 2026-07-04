"""评估编排器对齐与 fail-fast 行为测试。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.evaluation.eval_orchestrator import run_evaluation


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _triplet(subject: str = "Seller") -> dict:
    return {
        "subject": {"text": subject, "role": "obligor"},
        "action": {"predicate": "deliver", "object": "goods"},
        "condition": {"text": "", "type": "none"},
    }


class TestEvalOrchestratorAlignment:
    def test_misaligned_testset_raises(self, tmp_path: Path):
        gold = tmp_path / "gold.jsonl"
        testset = tmp_path / "testset.jsonl"
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        out_dir = tmp_path / "out"

        _write_jsonl(
            gold,
            [
                {"clause_id": "C-001", "triplet": _triplet("A")},
                {"clause_id": "C-002", "triplet": _triplet("B")},
            ],
        )
        _write_jsonl(
            testset,
            [{"clause_id": "C-999", "text": "only one wrong id"}],
        )
        for name in ("baseline", "ours_dep", "ours_reflexion"):
            _write_jsonl(
                pred_dir / f"{name}.jsonl",
                [{"clause_id": cid, "triplet": _triplet()} for cid in ("C-001", "C-002")],
            )

        with patch(
            "src.evaluation.eval_orchestrator.parse_trees_for_clauses",
            return_value=[],
        ):
            with pytest.raises(RuntimeError, match="not aligned with gold"):
                run_evaluation(
                    predictions_dir=str(pred_dir),
                    gold_path=str(gold),
                    testset_path=str(testset),
                    output_dir=str(out_dir),
                    config_path=str(Path(__file__).resolve().parents[1] / "configs/model.yaml"),
                    constraints_path=str(
                        Path(__file__).resolve().parents[1] / "configs/constraints.yaml"
                    ),
                )
