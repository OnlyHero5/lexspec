"""
法律三元组抽取器 —— 基于大语言模型的结构化信息抽取
====================================================

从合同条款中抽取 (subject, action, condition) 三元组，
使用结构化提示词 + JSON Schema 约束的大语言模型调用。

架构:
  1. 提示词加载 —— 从 configs/prompts.yaml 加载模板
  2. 大语言模型调用 —— LLMClient 发送格式化提示词
  3. 响应解析 —— 多策略 JSON 提取 + 回退
  4. 校验与规范化 —— Pydantic 模型校验 + 枚举值规范化

使用示例::

    from src.extraction.client import LLMClient, ClientConfig
    from src.extraction.extractor import LegalTripletExtractor

    config = ClientConfig(base_url="http://localhost:8080/v1", model="qwen3.5-9b")
    client = LLMClient(config)
    extractor = LegalTripletExtractor(client)
    triplet = extractor.extract("Seller shall deliver the goods within 30 days.")
"""

from __future__ import annotations

from typing import List

from src.extraction.schema import LegalTriplet
from src.extraction.client import LLMClient
from src.utils.logging import get_logger

from ._prompts import load_prompts
from ._parsing import parse_llm_response
from ._validation import (
    validate_and_normalize_triplet,
    build_fallback_triplet,
)

logger = get_logger(__name__)


# =============================================================================
# 法律三元组抽取器
# =============================================================================


class LegalTripletExtractor:
    """通过大语言模型从合同条款中抽取法律动作三元组。

    从 ``configs/prompts.yaml`` 加载提示词模板。若配置文件缺失则抛出错误
    （不使用硬编码默认值），将模板与目标条款格式化后发送给大语言模型，
    并将结构化 JSON 响应解析为经过校验的 ``LegalTriplet`` Pydantic 模型。

    抽取器设计为**对大语言模型故障具有鲁棒性**：若模型返回格式错误的 JSON、
    字段不完整或拒绝作答，将构建回退三元组以保证批量处理不中断。
    失败的抽取会记录完整上下文日志，便于事后调试。

    属性:
        client:             已配置的 ``LLMClient`` 实例。
        system_prompt:      系统提示词模板（从文件加载）。
        user_prompt_template:  用户提示词模板，含 ``{clause}`` 或
                               ``{sentence}`` 占位符。
        prompts_source:     提示词来源的可读描述（用于日志）。
    """

    def __init__(
        self,
        client: LLMClient,
        prompts_path: str = "configs/prompts.yaml",
    ):
        """使用大语言模型客户端和提示词配置初始化抽取器。

        从 ``prompts_path``（YAML）加载提示词模板。若配置文件缺失或不完整
        则抛出错误，不会静默回退到硬编码默认值。

        参数:
            client:        指向 llama.cpp 服务器的已配置 ``LLMClient`` 实例。
            prompts_path:  ``prompts.yaml`` 配置文件路径。
                           默认为 ``configs/prompts.yaml``。

        异常:
            FileNotFoundError: prompts_path 不存在。
            KeyError: YAML 缺少必需的键。
            yaml.YAMLError: YAML 格式错误。
        """
        self.client = client

        loaded = load_prompts(prompts_path)
        self.system_prompt = loaded["system"]
        self.user_prompt_template = loaded["user"]
        self.prompts_source = loaded["source"]

        logger.info(
            "LegalTripletExtractor initialized (prompts_source=%s, model=%s)",
            self.prompts_source,
            client.config.model,
        )

    # ---------------------------------------------------------------------
    # 主抽取入口
    # ---------------------------------------------------------------------

    def extract(self, clause: str) -> LegalTriplet:
        """从单条合同条款中抽取法律三元组。

        主入口方法，执行以下步骤：

        1. **格式化提示词** —— 将条款文本插入用户提示词模板
           （根据所加载模板处理 ``{clause}`` 或 ``{sentence}`` 占位符）。
        2. **调用大语言模型** —— 通过 ``LLMClient.complete_structured()`` 发送。
        3. **解析响应** —— 采用多种回退策略的鲁棒 JSON 解析。
        4. **校验** —— 对照 ``LegalTriplet`` Pydantic 模型校验解析后的字典。
        5. **规范化** —— 修剪空白、处理空条件、确保枚举值有效。

        参数:
            clause:  合同条款字符串，例如
                     ``"Seller shall deliver the goods within 30 days."``

        返回:
            包含已抽取主体、动作与条件的经过校验的 ``LegalTriplet``。

        异常:
            ValueError:  若在所有回退尝试后仍无法解析或校验大语言模型响应。
                         此情况较少见——方法在多数失败模式下会返回回退三元组。
        """
        # --- 步骤 1：用条款文本格式化用户提示词 ---
        # 需处理两种占位符格式：
        #   - {clause}   —— 默认模板使用此格式
        #   - {sentence} —— YAML 配置（prompts.yaml）使用此格式
        # 格式化器根据模板中出现的占位符选择正确的键。
        user_prompt = self._format_prompt(clause)

        logger.debug(
            "Extracting triplet from clause (len=%d): %s",
            len(clause),
            clause[:120] + ("..." if len(clause) > 120 else ""),
        )

        # --- 步骤 2：调用大语言模型 ---
        try:
            raw_response = self.client.complete_structured(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
            )
            logger.debug(
                "LLM raw response (len=%d): %s",
                len(raw_response),
                raw_response[:200] + ("..." if len(raw_response) > 200 else ""),
            )
        except RuntimeError as exc:
            # 大语言模型客户端已耗尽所有重试。构建回退三元组，
            # 以便调用方继续处理其他条款。
            logger.error(
                "LLM request failed for clause: %s",
                exc,
            )
            return build_fallback_triplet(
                clause=clause,
                error=f"LLM request failed: {exc}",
            )

        # --- 步骤 3：解析 JSON 响应 ---
        parsed = parse_llm_response(raw_response)

        # 若解析返回空字典，说明响应完全无法解析。构建回退三元组。
        if not parsed:
            logger.warning(
                "Could not parse LLM response into JSON dict. "
                "Raw response: %s",
                raw_response[:200],
            )
            return build_fallback_triplet(
                clause=clause,
                error="JSON parse failure: no valid JSON found in response",
            )

        # --- 步骤 4 与 5：校验与规范化 ---
        try:
            triplet = validate_and_normalize_triplet(parsed, clause)
            logger.debug(
                "Extracted triplet: subject=(text=%r, role=%s), "
                "action=(predicate=%r, object=%r), "
                "condition=(text=%r, type=%s)",
                triplet.subject.text,
                triplet.subject.role.value,
                triplet.action.predicate,
                triplet.action.object,
                triplet.condition.text,
                triplet.condition.type.value,
            )
            return triplet

        except (ValueError, TypeError, KeyError) as exc:
            # 对照 Pydantic 模型的校验失败。可能因大语言模型返回了
            # 有效 JSON 但不符合预期模式（如缺少必填字段、类型错误）。
            logger.warning(
                "Triplet validation failed for clause: %s. Parsed: %s",
                exc,
                parsed,
            )
            return build_fallback_triplet(
                clause=clause,
                error=f"Validation failure: {exc}",
            )

    def extract_batch(self, clauses: List[str]) -> List[LegalTriplet]:
        """顺序从多条条款中抽取三元组。

        每条条款通过 ``self.extract()`` 独立处理。
        错误立即向上传播，不做静默捕获。

        参数:
            clauses:  合同条款字符串列表。

        返回:
            ``LegalTriplet`` 对象列表，与输入条款一一对应、顺序一致。

        异常:
            RuntimeError: 大语言模型请求失败。
            ValueError: 解析或校验失败。
        """
        results: List[LegalTriplet] = []

        for clause in clauses:
            triplet = self.extract(clause)
            results.append(triplet)

        logger.info(
            "Batch extraction complete: %d/%d clauses processed",
            len(results),
            len(clauses),
        )
        return results

    # ---------------------------------------------------------------------
    # 提示词格式化
    # ---------------------------------------------------------------------

    def _format_prompt(self, clause: str) -> str:
        """将条款文本插入用户提示词模板。

        自动处理 ``{clause}``（默认模板）与 ``{sentence}``
        （prompts.yaml 模板）两种占位符，根据已加载模板字符串
        中出现的占位符进行判断。

        模板使用 Python 的 ``str.format()`` 格式化。默认模板中的
        ``{{`` 与 ``}}`` 转义序列在最终提示词中变为字面量 ``{`` 与 ``}``，
        这些是大语言模型所见 JSON 示例的一部分。

        参数:
            clause:  待插入的合同条款文本。

        返回:
            可供大语言模型使用的完整格式化提示词字符串。
        """
        template = self.user_prompt_template

        # --- 确定使用哪种占位符 ---
        # YAML 配置使用 {sentence}，默认模板使用 {clause}。
        # 从模板字符串本身检测格式，使抽取器无论使用哪种提示词来源均可工作。
        if "{sentence}" in template:
            return template.format(sentence=clause)
        elif "{clause}" in template:
            return template.format(clause=clause)
        else:
            # 若模板无已识别的占位符，作为最后手段在新行追加条款。
            # 标准模板不应出现此情况，但可防止自定义提示词配置错误。
            logger.warning(
                "Prompt template has no {clause} or {sentence} placeholder. "
                "Appending clause to the end."
            )
            return template + "\n\n" + clause
