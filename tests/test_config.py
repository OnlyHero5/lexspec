"""YAML 驱动配置加载的测试。"""

from pathlib import Path

import pytest

from src.utils.constraints import (
    get_corpus_sampling_config,
    get_f1_weights,
    get_normalization_config,
    get_party_alias_mappings,
    get_phenomenon_quotas,
    get_validation_thresholds,
    load_constraints_config,
    normalize_for_comparison,
)
from src.utils.config import 获取Reflexion参数, 加载模型配置
from src.utils.prompt_loader import load_extraction_prompts, load_reflexion_config
from src.correction.reflexion import ReflexionGenerator
from src.extraction.client import ClientConfig, LLMClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONSTRAINTS = str(PROJECT_ROOT / "configs" / "constraints.yaml")
PROMPTS = str(PROJECT_ROOT / "configs" / "prompts.yaml")


class TestConstraintsYaml:
    def test_load_constraints(self):
        config = load_constraints_config(CONSTRAINTS)
        assert "f1_weights" in config
        assert "corpus_sampling" in config

    def test_f1_weights_sum_to_one(self):
        config = load_constraints_config(CONSTRAINTS)
        weights = get_f1_weights(config, CONSTRAINTS)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_normalization_flags(self):
        config = load_constraints_config(CONSTRAINTS)
        norm = get_normalization_config(config, CONSTRAINTS)
        assert norm.remove_articles is True
        assert norm.use_party_aliases is True

    def test_party_alias_mappings_structure(self):
        config = load_constraints_config(CONSTRAINTS)
        aliases = get_party_alias_mappings(config, CONSTRAINTS)
        assert "Seller" in aliases
        assert isinstance(aliases["Seller"], list)

    def test_phenomenon_quotas_from_proportions(self):
        config = load_constraints_config(CONSTRAINTS)
        quotas = get_phenomenon_quotas(config, target_count=100, config_path=CONSTRAINTS)
        assert quotas["passive"] >= 20
        assert quotas["conditional"] >= 25

    def test_validation_thresholds_present(self):
        config = load_constraints_config(CONSTRAINTS)
        thresholds = get_validation_thresholds(config, CONSTRAINTS)
        assert thresholds["long_distance_tokens"] == 3
        assert thresholds["long_distance_mdd"] == 6.0

    def test_corpus_sampling_defaults(self):
        config = load_constraints_config(CONSTRAINTS)
        sampling = get_corpus_sampling_config(config, CONSTRAINTS)
        assert sampling["target_count_default"] == 100
        assert sampling["random_seed"] == 42

    def test_normalize_for_comparison_uses_yaml(self):
        result = normalize_for_comparison("The Seller", config_path=CONSTRAINTS)
        assert "seller" in result


class TestPromptsYaml:
    def test_reflexion_config_has_default_hint(self):
        feedback, system, hints = load_reflexion_config(PROMPTS)
        assert feedback.strip()
        assert system.strip()
        assert "default" in hints
        assert "passive_subject" in hints

    def test_baseline_prompt_uses_lexspec_triplet_schema(self):
        prompts = load_extraction_prompts(PROMPTS)
        system = prompts["system"]
        assert '"action"' in system
        assert '"predicate"' in system
        assert "Return ONLY a JSON array" not in system
        assert "NO JSON arrays" in system
        assert "Return ONLY a single JSON object" in prompts["user"]

    def test_reflexion_params_from_model_yaml(self):
        config = 加载模型配置(str(PROJECT_ROOT / "configs" / "model.yaml"))
        params = 获取Reflexion参数(config)
        assert params["max_iterations"] == 1
        assert params["temperature"] == 0.0

    def test_reflexion_generator_initializes(self):
        client = LLMClient(
            ClientConfig(model="test-model", base_url="http://127.0.0.1:8080/v1")
        )
        gen = ReflexionGenerator(
            client,
            prompts_path=PROMPTS,
            constraints_path=CONSTRAINTS,
        )
        assert gen.long_distance_token_threshold == 3

    def test_missing_default_hint_raises(self, tmp_path):
        bad_yaml = tmp_path / "prompts.yaml"
        bad_yaml.write_text(
            "reflexion:\n  feedback_template: '{error_type}'\n  error_hints:\n    foo: bar\n",
            encoding="utf-8",
        )
        with pytest.raises(KeyError, match="default"):
            load_reflexion_config(str(bad_yaml))
