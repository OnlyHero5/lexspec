"""
校验步骤 6：角色校验。

根据情态/极性分析得到的 UD 推导法律角色，校验 subject.role。
"""

from __future__ import annotations

from typing import List

from src.extraction.schema import (
    DependencyTree,
    FieldCorrection,
    LegalRole,
)
from src.linguistic.polarity_detector import PolarityDetector
from src.utils.logging import get_logger

logger = get_logger(__name__)


def step6_validate_role(
    llm_role: LegalRole,
    tree: DependencyTree,
    predicate_idx: int,
    predicate_lemma: str,
    polarity_detector: PolarityDetector,
    corrections: List[FieldCorrection],
    feedback_parts: List[str],
) -> bool:
    """步骤 6：根据 UD 推导角色校验 subject.role。

    法律角色由情态助动词（shall/may/must）与
    否定（not/no/never）模式确定。此为 PolarityDetector 的主要功能。

    角色校验通常是最可靠的步骤，因规则是确定性的：
    给定助动词与否定状态，角色唯一确定
    （"will" 除外，可能在义务与将来时之间歧义）。

    参数：
        llm_role: 大语言模型分配的角色。
        tree: 依存树。
        predicate_idx: 谓词的 1 基索引。
        predicate_lemma: 谓词词元（用于词项覆盖）。
        polarity_detector: PolarityDetector 实例。
        corrections: 追加 FieldCorrection 的列表。
        feedback_parts: 追加反馈字符串的列表。

    返回：
        角色匹配时返回 True。
    """
    ud_role, polarity = polarity_detector.detect(
        tree, predicate_idx, predicate_lemma
    )

    if ud_role == LegalRole.OTHER:
        # UD 无法确定角色（无情态、无清晰模式）。
        # 接受大语言模型所赋角色 —— 无证据反驳。
        logger.debug("UD role is OTHER — accepting LLM role %s", llm_role.value)
        return True

    if llm_role == ud_role:
        return True

    # 角色不匹配：UD 有明确角色，大语言模型分配不同。
    # 几乎总是可修正错误，因角色规则是确定性的。
    corrections.append(FieldCorrection(
        field="subject.role",
        original=llm_role.value,
        corrected=ud_role.value,
        reason=(
            f"Legal role derived from UD parse: "
            f"modal='{polarity_detector.detect_modality(tree, predicate_idx)[0]}', "
            f"polarity={polarity}, predicate='{predicate_lemma}' -> "
            f"{ud_role.value}. The LLM assigned {llm_role.value} which "
            f"conflicts with the syntactic evidence."
        ),
    ))
    feedback_parts.append(
        f"Role mismatch: LLM assigned {llm_role.value} but UD modality "
        f"analysis indicates {ud_role.value} (polarity={polarity})."
    )
    return False
