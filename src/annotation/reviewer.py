"""
Cross-Model Annotation Reviewer
=================================
One annotation model reviews another model's labels on the same clause.

Used in the phased workflow:
  1. Gemma annotates → save locally
  2. Switch to Qwen → Qwen annotates + Qwen reviews Gemma
  3. Switch to Gemma → Gemma reviews Qwen
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
    """Per-field accept/reject judgment from the reviewer."""

    field: str
    judgment: str  # "accept" | "reject"
    reason: str = ""


@dataclass
class ReviewResult:
    """Structured output of a cross-model review."""

    verdict: str  # "accept" | "partial" | "reject"
    field_judgments: List[FieldJudgment] = field(default_factory=list)
    corrected_triplet: Optional[LegalTriplet] = None
    overall_reason: str = ""
    raw_response: str = ""


class CrossModelReviewer:
    """Have one LLM review another model's triplet annotation."""

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
        """Review a single annotation."""
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
        """Review a batch of annotation records.

        Each item must have keys: text, triplet (dict or LegalTriplet),
        and optionally source_model / model_role.
        """
        iterator = items
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(items, desc="Reviewing", unit="clause")
            except ImportError:
                pass

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
        """Load review prompts from YAML config. Raises on any failure."""
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
