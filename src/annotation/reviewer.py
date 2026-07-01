"""
跨模型标注审查器
================
让一个标注模型审查另一模型对同一条款的标注。

用于分阶段工作流：
  1. Gemma 标注 → 本地保存
  2. 切换到 Qwen → Qwen 标注 + Qwen 审查 Gemma
  3. 切换到 Gemma → Gemma 审查 Qwen
  4. merge → gold_triplets.jsonl
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING

from src.extraction.schema import LegalTriplet
from src.annotation.triplet_coercion import coerce_to_triplet
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.extraction.client import LLMClient

logger = get_logger(__name__)


@dataclass
class FieldJudgment:
    """审查器对每个字段的接受/拒绝判断。"""

    field: str
    judgment: str  # 接受或拒绝
    reason: str = ""


@dataclass
class ReviewResult:
    """跨模型审查的结构化输出。"""

    verdict: str  # 接受、部分接受或拒绝
    field_judgments: List[FieldJudgment] = field(default_factory=list)
    corrected_triplet: Optional[LegalTriplet] = None
    overall_reason: str = ""
    raw_response: str = ""


class CrossModelReviewer:
    """让一个 LLM 审查另一模型的三元组标注。"""

    def __init__(
        self,
        client: "LLMClient",
        reviewer_role: str,
        prompts_path: str = "configs/prompts.yaml",
    ) -> None:
        self.client = client
        self.reviewer_role = reviewer_role
        self.system_prompt, self.user_template = self._load_review_prompts(
            prompts_path
        )

    def review(
        self,
        sentence: str,
        source_triplet: LegalTriplet,
        source_model: str,
    ) -> ReviewResult:
        """审查单条标注。"""
        source_json = json.dumps(
            source_triplet.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        user_prompt = self.user_template.format(
            sentence=sentence,
            source_model=source_model,
            source_triplet_json=source_json,
        )

        response = self.client.complete(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
        )
        return self._parse_review_response(response)

    def review_batch(
        self,
        items: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> List[ReviewResult]:
        """批量审查标注记录。

        每项须含键：text、triplet（dict 或 LegalTriplet），
        以及可选的 source_model / model_role。
        """
        iterator = items
        if show_progress:
            from src.utils.progress import progress_bar
            iterator = progress_bar(items, desc="Reviewing", unit="clause")

        results: List[ReviewResult] = []
        for item in iterator:
            text = item["text"]
            triplet = item["triplet"]
            if isinstance(triplet, dict):
                triplet = LegalTriplet.model_validate(triplet)
            source_model = (
                item.get("source_model")
                or item.get("model_role")
                or item.get("model")
                or "other_model"
            )
            results.append(self.review(text, triplet, source_model))
        return results

    def _load_review_prompts(self, prompts_path: str) -> tuple[str, str]:
        """从 YAML 配置加载审查提示词。任何失败均抛出异常。"""
        import yaml
        with open(prompts_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        if not isinstance(config, dict):
            raise TypeError(
                f"Prompts config '{prompts_path}' is not a valid YAML mapping."
            )
        review = (config.get("annotation") or {}).get("review") or {}
        system = review.get("system", "").strip()
        user = review.get("user", "").strip()
        if not system:
            raise KeyError(
                f"Missing or empty 'annotation.review.system' in '{prompts_path}'."
            )
        if not user:
            raise KeyError(
                f"Missing or empty 'annotation.review.user' in '{prompts_path}'."
            )
        logger.info("Loaded review prompts from %s", prompts_path)
        return system, user

    def _parse_review_response(self, response: str) -> ReviewResult:
        if not response or not response.strip():
            return ReviewResult(verdict="reject", overall_reason="Empty response")

        cleaned = response.strip()
        fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()

        data: Optional[dict] = None
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            for match in re.finditer(
                r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned
            ):
                try:
                    data = json.loads(match.group())
                    break
                except json.JSONDecodeError:
                    continue

        if not isinstance(data, dict):
            return ReviewResult(
                verdict="reject",
                overall_reason="Could not parse review JSON",
                raw_response=response,
            )

        verdict = str(data.get("verdict", "partial")).lower()
        if verdict not in ("accept", "partial", "reject"):
            verdict = "partial"

        field_judgments: List[FieldJudgment] = []
        for fj in data.get("field_judgments") or []:
            if not isinstance(fj, dict):
                continue
            field_judgments.append(
                FieldJudgment(
                    field=str(fj.get("field", "")),
                    judgment=str(fj.get("judgment", "accept")).lower(),
                    reason=str(fj.get("reason", "")),
                )
            )

        corrected: Optional[LegalTriplet] = None
        raw_corrected = data.get("corrected_triplet")
        if isinstance(raw_corrected, dict):
            corrected = coerce_to_triplet(raw_corrected)

        return ReviewResult(
            verdict=verdict,
            field_judgments=field_judgments,
            corrected_triplet=corrected,
            overall_reason=str(data.get("overall_reason", "")),
            raw_response=response,
        )
