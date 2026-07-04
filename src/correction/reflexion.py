"""
用于纠正 LLM 抽取错误的 Reflexion 反馈生成器
==============================================
编排 Reflexion 自纠错循环：当 UD 约束校验器返回
REFLEXION_REQUIRED 时，本模块生成带语言学提示的
定向反馈提示词，调用 LLM 重新生成，并返回纠正后的三元组。

全部提示模板与错误提示从 configs/prompts.yaml 加载。
距离阈值从 configs/constraints.yaml 加载。
"""

from __future__ import annotations

import json
from typing import Optional, TYPE_CHECKING

from src.extraction.schema import LegalTriplet, ValidationResult
from src.correction.reflexion_error_mapper import determine_error_types
from src.correction.response_parser import parse_llm_response
from src.utils.constraints import get_validation_thresholds, load_constraints_config
from src.utils.prompt_loader import load_reflexion_config
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.extraction.client import LLMClient

logger = get_logger(__name__)


class ReflexionGenerator:
    """Reflexion 自纠错生成器 —— 根据 UD 校验结果生成反馈并重抽取三元组。

    当 ``ConstraintValidator`` 返回 ``REFLEXION_REQUIRED`` 时，本类：
      1. 将校验错误映射为结构化错误类型；
      2. 从 ``prompts.yaml`` 组装带语言学证据的反馈提示词；
      3. 调用实验模型重新生成并解析为 ``LegalTriplet``。

    属性:
        client: 实验用大语言模型客户端。
        reflexion_prompt_template: Reflexion 反馈模板（含占位符）。
        system_prompt: 重抽取阶段的系统提示词。
        error_hints: 错误类型 → 定向修正指引 的映射。
        long_distance_token_threshold: 判定长距离依存错误的最小词元距离。
    """

    def __init__(
        self,
        client: "LLMClient",
        prompts_path: str = "configs/prompts.yaml",
        constraints_path: str = "configs/constraints.yaml",
        reflexion_temperature: float = 0.0,
        max_iterations: int = 1,
    ) -> None:
        """初始化 Reflexion 生成器并加载提示词与阈值配置。

        参数:
            client: 已配置的 ``LLMClient``，用于执行纠错重生成。
            prompts_path: Reflexion 反馈模板与错误提示的 YAML 路径。
            constraints_path: 校验阈值配置路径，用于读取
                ``long_distance_tokens`` 等参数。

        异常:
            FileNotFoundError: 提示词或约束配置文件不存在。
            KeyError: YAML 缺少 Reflexion 必需的键（如 ``error_hints``）。
        """
        self.client = client
        self.prompts_path = prompts_path
        self.constraints_path = constraints_path
        self.reflexion_temperature = reflexion_temperature
        self.max_iterations = max(1, int(max_iterations))

        feedback, system_prompt, error_hints = load_reflexion_config(prompts_path)
        self.reflexion_prompt_template = feedback
        self.system_prompt = system_prompt
        self.error_hints = error_hints

        constraints = load_constraints_config(constraints_path)
        thresholds = get_validation_thresholds(constraints, constraints_path)
        self.long_distance_token_threshold = int(thresholds["long_distance_tokens"])

        logger.info(
            "ReflexionGenerator initialized (prompts=%s, constraints=%s, hints=%d, "
            "long_distance_tokens=%d, temperature=%.1f, max_iterations=%d)",
            prompts_path,
            constraints_path,
            len(self.error_hints),
            self.long_distance_token_threshold,
            self.reflexion_temperature,
            self.max_iterations,
        )

    def generate_feedback(
        self,
        validation_result: ValidationResult,
        clause: str = "",
    ) -> str:
        """根据校验结果生成 Reflexion 反馈提示词。

        参数:
            validation_result: UD 约束校验器的完整输出，须含
                ``original_prediction``、``linguistic_evidence`` 与
                ``corrections``，用于填充模板占位符。
            clause: 原始合同条款文本；为空时从 ``validation_result`` 中
                尽力推导摘要性描述。

        返回:
            已格式化的用户侧反馈提示词字符串，可直接作为
            ``LLMClient.complete()`` 的 ``user_prompt`` 传入。
        """
        error_types = determine_error_types(
            validation_result,
            self.long_distance_token_threshold,
        )
        primary_error = error_types[0] if error_types else "default"
        specific_hint = self.error_hints[primary_error]

        clause_text = clause if clause else self._derive_clause_text(validation_result)
        prediction_json = json.dumps(
            validation_result.original_prediction.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        )
        evidence_json = json.dumps(
            validation_result.linguistic_evidence.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        )

        feedback = self.reflexion_prompt_template.format(
            error_type=primary_error,
            text=clause_text,
            prediction=prediction_json,
            linguistic_evidence=evidence_json,
            specific_hint=specific_hint,
        )
        logger.debug(
            "Generated feedback prompt (%d chars) for error type=%s",
            len(feedback),
            primary_error,
        )
        return feedback

    def correct(
        self,
        clause: str,
        validation_result: ValidationResult,
    ) -> Optional[LegalTriplet]:
        """执行一轮 Reflexion 纠错（生成反馈 → 调用 LLM → 解析响应）。

        参数:
            clause: 原始合同条款文本，写入反馈模板供模型对照。
            validation_result: 触发 Reflexion 的校验结果，通常
                ``status == REFLEXION_REQUIRED``。

        返回:
            解析成功的纠正后 ``LegalTriplet``；若 LLM 响应无法解析为
            合法三元组则返回 ``None``（调用方应保留原预测或回退策略）。

        异常:
            RuntimeError: LLM 请求在重试耗尽后仍失败（由 ``LLMClient`` 抛出）。
        """
        feedback_prompt = self.generate_feedback(validation_result, clause)

        logger.info("Calling LLM for Reflexion correction on clause: %.80s...", clause)
        response = self.client.complete_structured(
            system_prompt=self.system_prompt,
            user_prompt=feedback_prompt,
            temperature=self.reflexion_temperature,
        )

        corrected_triplet = parse_llm_response(response)
        if corrected_triplet is not None:
            logger.info(
                "Reflexion correction succeeded: subject=%s, predicate=%s",
                corrected_triplet.subject.text,
                corrected_triplet.action.predicate,
            )
        else:
            logger.warning(
                "Reflexion correction failed: could not parse LLM response into LegalTriplet"
            )
        return corrected_triplet

    @staticmethod
    def _derive_clause_text(validation_result: ValidationResult) -> str:
        """从校验结果尽力推导条款文本。"""
        pred = validation_result.original_prediction
        if pred.action.predicate:
            subject_text = pred.subject.text or "unknown party"
            return (
                f"Clause involving '{subject_text}' performing "
                f"action '{pred.action.predicate}'"
            )
        return ""
