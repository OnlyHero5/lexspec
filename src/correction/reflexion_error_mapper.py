"""
Reflexion 错误类型判定
======================
将校验修正与语言学证据映射到 Reflexion 错误提示键。
"""

from __future__ import annotations

from typing import List

from src.extraction.schema import ValidationResult, FieldCorrection
from src.utils.logging import get_logger

logger = get_logger(__name__)


def determine_error_types(
    validation_result: ValidationResult,
    long_distance_token_threshold: int,
) -> List[str]:
    """分析修正项以判定语言学错误类型。

    参数：
        validation_result: 含修正与证据的校验结果。
        long_distance_token_threshold: 标记长距离依存错误所需的
            最小谓词–论元距离（来自 constraints.yaml
            validation.long_distance_tokens）。

    返回：
        按优先级排序的错误提示键列表。
    """
    corrections: List[FieldCorrection] = validation_result.corrections
    evidence = validation_result.linguistic_evidence

    if validation_result.status.value == "VALID":
        return []

    error_types: List[str] = []
    corrected_fields = {c.field for c in corrections}

    if evidence.passive_detected:
        subject_corrected = any(
            f in corrected_fields for f in ("subject.text", "subject.role")
        )
        object_corrected = "action.object" in corrected_fields
        if subject_corrected:
            error_types.append("passive_subject")
        if object_corrected:
            error_types.append("passive_object")

    condition_corrected = any(
        f in corrected_fields for f in ("condition.text", "condition.type")
    )
    if condition_corrected:
        error_types.append("condition_boundary")

    if evidence.polarity == "negative" and "subject.role" in corrected_fields:
        error_types.append("negation_role")

    if "subject.role" in corrected_fields and "negation_role" not in error_types:
        if "passive_subject" not in error_types:
            error_types.append("role_mismatch")

    action_corrected = any(
        f in corrected_fields for f in ("action.object", "action.predicate")
    )
    if (
        action_corrected
        and "passive_object" not in error_types
        and evidence.max_argument_distance > long_distance_token_threshold
    ):
        error_types.append("long_distance_object")

    if not error_types and corrections:
        error_types.append("default")

    logger.debug(
        "Error type analysis: corrected_fields=%s, max_arg_dist=%d, threshold=%d -> %s",
        corrected_fields,
        evidence.max_argument_distance,
        long_distance_token_threshold,
        error_types,
    )
    return error_types
