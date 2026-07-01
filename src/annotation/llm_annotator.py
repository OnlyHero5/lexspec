"""
基于 LLM 的合同条款标注器（金标准测试集构建）
============================================
使用 LLM（Qwen3.6 27B 或 Gemma4 31B）为合同条款标注法律三元组。
每个标注模型独立标注每条条款。

**关键约束**：标注模型与第二、三阶段使用的实验模型（Qwen3.5 9B）完全隔离。
本模块仅用于第一阶段（测试集构建），不出现在第二、三阶段。

用法::

    from src.extraction.client import LLMClient, ClientConfig
    from src.annotation.llm_annotator import LLMAnnotator

    client = LLMClient(ClientConfig(
        base_url="http://10.0.16.254:8080/v1",
        model="gemma-4-31B-it-Q8_0.gguf",
        timeout=1200,
    ))
    annotator = LLMAnnotator(client)
    triplet = annotator.annotate("Seller shall deliver the Goods.")
"""

from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING

from src.extraction.schema import LegalTriplet
from src.utils.logging import get_logger

from src.annotation.prompts import load_annotation_prompts
from src.annotation.response_parser import parse_llm_response

if TYPE_CHECKING:
    from src.extraction.client import LLMClient

logger = get_logger(__name__)


class LLMAnnotator:
    """使用 LLM 为合同条款标注法律三元组。

    支持双模型标注 — 各模型独立标注，
    结果通过字段级投票共识调和
    （见 src/annotation/consensus.py）。

    提示词从 configs/prompts.yaml 加载（见 annotation.system
    与 annotation.user 字段）。配置不可用时抛出异常。

    属性：
        client: 为标注模型配置的 LLM 客户端。
        system_prompt: 标注调用的系统提示词。
        user_template: 含 {sentence} 占位符的用户提示词模板。
    """

    def __init__(
        self,
        client: "LLMClient",
        prompts_path: str = "configs/prompts.yaml",
    ) -> None:
        """初始化标注器。

        从 YAML 配置文件加载标注提示词模板。
        文件缺失或格式错误时抛出异常。

        参数：
            client: 为标注模型配置的 LLM 客户端
                    （如 Qwen3.6 27B 或 Gemma4 31B）。
            prompts_path: 提示词 YAML 配置文件路径。
                          默认为 "configs/prompts.yaml"。
        """
        self.client = client
        self.system_prompt, self.user_template = load_annotation_prompts(
            prompts_path
        )
        logger.info(
            "LLMAnnotator initialized (prompts_path=%s)", prompts_path
        )
        logger.debug(
            "System prompt length: %d chars, user template length: %d chars",
            len(self.system_prompt),
            len(self.user_template),
        )

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def annotate(self, clause: str) -> LegalTriplet:
        """标注单条合同条款。

        将条款文本注入提示词模板，发送给 LLM，
        并将响应解析为 LegalTriplet。

        参数：
            clause: 待标注的合同条款文本。

        返回：
            含主体、行为、条件提取结果的 LegalTriplet。

        异常:
            ValueError: LLM 响应无法解析为有效 LegalTriplet，或调用失败时。
        """
        logger.debug("Annotating clause: %.80s...", clause)

        # 步骤 1：将条款文本注入用户提示词模板。
        user_prompt = self.user_template.format(sentence=clause)

        # 步骤 2：调用 LLM。
        try:
            response = self.client.complete_structured(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            logger.error("LLM call failed during annotation: %s", exc)
            raise ValueError(
                f"LLM annotation call failed, clause: {clause[:100]}..."
            ) from exc

        logger.debug("LLM annotation response length: %d chars", len(response))

        # 步骤 3：将响应解析为 LegalTriplet。
        triplet = parse_llm_response(response)

        if triplet is None:
            logger.error(
                "Could not parse LLM annotation response. First 200 chars: %.200s",
                response,
            )
            raise ValueError(
                f"Cannot parse annotation response, clause: {clause[:100]}..."
            )

        if not self._is_substantive_triplet(triplet):
            logger.error(
                "LLM returned an empty or invalid triplet for clause: %.80s...",
                clause,
            )
            raise ValueError(
                f"Annotation triplet is empty or invalid, clause: {clause[:100]}..."
            )

        logger.debug(
            "Annotation complete: subject=%s (role=%s), predicate=%s, object=%s, condition=%s",
            triplet.subject.text,
            triplet.subject.role.value,
            triplet.action.predicate,
            triplet.action.object,
            triplet.condition.type.value if triplet.condition.text else "none",
        )

        return triplet

    @staticmethod
    def _is_substantive_triplet(triplet: LegalTriplet) -> bool:
        """检查三元组是否包含实质性内容。

        拒绝解析成功但核心字段均为空的外壳。

        规则：
          - 操作性条款：至少需要 subject.text + action.predicate
          - 定义性条款（「X means Y」）：即使无当事方，
            也需要 predicate + object

        参数：
            triplet: 待验证的 LegalTriplet。

        返回：
            三元组有有意义内容时为 True，否则为 False。
        """
        has_subject = bool(triplet.subject.text.strip())
        has_predicate = bool(triplet.action.predicate.strip())
        has_object = bool(triplet.action.object.strip())
        if has_subject and has_predicate:
            return True
        if has_predicate and has_object:
            return True
        return False

    def annotate_batch(
        self,
        clauses: List[str],
        show_progress: bool = True,
    ) -> List[LegalTriplet]:
        """批量标注合同条款。

        每条条款独立标注。默认显示 tqdm 进度条。
        单条失败不会中止整批；失败项用占位三元组填充，
        使输出列表长度始终与输入一致。

        参数：
            clauses: 待标注的条款文本列表。
            show_progress: 是否显示 tqdm 进度条。默认 True。

        返回：
            LegalTriplet 列表，长度与输入相同。失败项
            的 subject.text 为 "ERROR"。
        """
        iterator = clauses
        if show_progress:
            from src.utils.progress import progress_bar
            iterator = progress_bar(clauses, desc="Annotating", unit="clause")

        results: List[LegalTriplet] = []
        for i, clause in enumerate(iterator):
            try:
                triplet = self.annotate(clause)
                results.append(triplet)
            except Exception as exc:
                logger.error(
                    "Annotation failed for item %d: %s. Clause: %.80s...",
                    i, exc, clause,
                )
                from src.extraction.schema import Subject, Action, Condition
                from src.extraction.schema import LegalRole, ConditionType

                placeholder = LegalTriplet(
                    subject=Subject(text="ERROR", role=LegalRole.OTHER),
                    action=Action(predicate="", object=""),
                    condition=Condition(text="", type=ConditionType.NONE),
                )
                results.append(placeholder)

        logger.info(
            "Batch annotation complete: %d/%d succeeded",
            sum(1 for t in results if t.subject.text != "ERROR"),
            len(clauses),
        )
        return results
