"""Reflexion 生成器与配置接入测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.correction.reflexion import ReflexionGenerator
from src.extraction.client import ClientConfig, LLMClient
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
from src.utils.config import 获取Reflexion参数, 加载模型配置


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS = str(PROJECT_ROOT / "configs" / "prompts.yaml")
CONSTRAINTS = str(PROJECT_ROOT / "configs" / "constraints.yaml")
MODEL = str(PROJECT_ROOT / "configs" / "model.yaml")


class TestReflexionGenerator:
    def test_correct_uses_complete_structured(self):
        client = LLMClient(
            ClientConfig(model="test-model", base_url="http://127.0.0.1:8080/v1")
        )
        gen = ReflexionGenerator(
            client,
            prompts_path=PROMPTS,
            constraints_path=CONSTRAINTS,
            reflexion_temperature=0.0,
            max_iterations=1,
        )
        triplet = LegalTriplet(
            subject=Subject(text="Seller", role=LegalRole.OBLIGOR),
            action=Action(predicate="deliver", object="goods"),
            condition=Condition(),
        )
        val_result = ValidationResult(
            status=ValidationStatus.REFLEXION_REQUIRED,
            original_prediction=triplet,
            linguistic_evidence=LinguisticEvidence(),
            feedback="",
        )
        payload = triplet.model_dump(mode="json")

        with patch.object(
            client,
            "complete_structured",
            return_value='{"subject":{"text":"Seller","role":"obligor"},'
            '"action":{"predicate":"deliver","object":"goods"},'
            '"condition":{"text":"","type":"none"}}',
        ) as mock_structured:
            result = gen.correct("Seller shall deliver goods.", val_result)

        mock_structured.assert_called_once()
        _, kwargs = mock_structured.call_args
        assert kwargs.get("temperature") == 0.0
        assert result is not None
        assert result.subject.text == payload["subject"]["text"]

    def test_reflexion_params_wired_from_model_yaml(self):
        config = 加载模型配置(MODEL)
        params = 获取Reflexion参数(config, MODEL)
        client = MagicMock(spec=LLMClient)
        gen = ReflexionGenerator(
            client,
            prompts_path=PROMPTS,
            constraints_path=CONSTRAINTS,
            reflexion_temperature=params["temperature"],
            max_iterations=params["max_iterations"],
        )
        assert gen.reflexion_temperature == params["temperature"]
        assert gen.max_iterations == params["max_iterations"]
