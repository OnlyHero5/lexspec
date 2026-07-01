"""
双模型标注的字段级投票共识
==========================
当两个标注模型（Qwen3.6 27B 与 Gemma4 31B）输出不一致时，
本模块逐字段解决分歧，并标识需人工审查的字段。

共识方式：
  - 在最细粒度比较字段（6 个）：
      1. subject.text
      2. subject.role
      3. action.predicate
      4. action.object
      5. condition.text
      6. condition.type
  - 文本字段使用规范化比较（小写、去冠词、
    尽可能词形还原），避免表面形式分歧。
  - 角色/类型字段使用枚举值精确匹配。
  - 两模型一致 -> 采用一致取值。
  - 不一致 -> 标记人工审查，暂用 anno_a（Qwen）
    作为暂定值（Qwen3.6 27B 为较大主模型）。

输出：
  - 共识 LegalTriplet（尽力合并）
  - 分歧记录列表，供下游人工审查与
    日志记录（见 src/annotation/disagreement_logger.py）
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple

from src.extraction.schema import (
    LegalTriplet,
    Subject,
    Action,
    Condition,
)
from src.utils.logging import get_logger

from src.annotation.normalization import normalize_text
from src.annotation.field_helpers import (
    FIELD_SPEC,
    _extract_field_values,
    _parse_role,
    _parse_condition_type,
)

logger = get_logger(__name__)


# =============================================================================
# 公开 API
# =============================================================================


def field_level_consensus(
    anno_a: LegalTriplet,
    anno_b: LegalTriplet,
) -> Tuple[LegalTriplet, List[Dict[str, Any]]]:
    """逐字段比较两份标注并产生共识。

    在最细粒度比较 6 个字段：
      - subject.text、subject.role
      - action.predicate、action.object
      - condition.text、condition.type

    匹配规则：
      - 文本字段：规范化比较（小写、去冠词、
        基本词形还原）。仅冠词、大小写或
        轻微屈折差异视为一致。
      - 角色/类型字段：枚举值字符串精确匹配。

    共识逻辑：
      - 两模型一致 -> 采用一致取值（因一致，
        取值取自 anno_a）。
      - 不一致 -> 字段标记为 needs_human_review，
        暂用 anno_a（Qwen，主模型）作为暂定值。

    参数：
        anno_a: 第一模型标注（通常为 Qwen，主模型）。
        anno_b: 第二模型标注（通常为 Gemma，次模型）。

    返回：
        (consensus_triplet, disagreements_list) 元组：
        - consensus_triplet: 尽力合并的 LegalTriplet。一致字段
          采用一致值；不一致字段用 anno_a 作暂定占位。
        - disagreements: 分歧记录列表，每项为 dict：
            {
                "field": str,          # 如 "subject.text"
                "anno_a_value": str,   # Qwen 取值
                "anno_b_value": str,   # Gemma 取值
                "resolved": bool,      # 初始 False；人工审查后设为 True
                "resolved_by": str,    # 初始 ""；设为 "human" 或 "auto"
                "resolution": str,     # 初始 ""；设为解决后的值
            }
    """
    # 为各标注的字段值建立查找表。
    a_values = _extract_field_values(anno_a)
    b_values = _extract_field_values(anno_b)

    disagreements: List[Dict[str, Any]] = []

    # 逐字段构建共识三元组。
    # 默认以 anno_a 取值开始。
    consensus_subject_text = anno_a.subject.text
    consensus_subject_role = anno_a.subject.role
    consensus_action_predicate = anno_a.action.predicate
    consensus_action_object = anno_a.action.object
    consensus_condition_text = anno_a.condition.text
    consensus_condition_type = anno_a.condition.type

    for field_name, _, _ in FIELD_SPEC:
        a_val = a_values[field_name]
        b_val = b_values[field_name]

        # 按该字段类型选用合适的比较策略，
        # 判断两值是否一致。
        is_text_field = field_name in (
            "subject.text", "action.predicate", "action.object", "condition.text"
        )

        if is_text_field:
            # 文本字段规范化比较。
            agreed = normalize_text(str(a_val)) == normalize_text(str(b_val))
        else:
            # 角色/类型枚举字段精确比较。
            agreed = str(a_val) == str(b_val)

        if agreed:
            # 两模型一致 — 无需操作。
            logger.debug("Field '%s': AGREED (a='%s', b='%s')", field_name, a_val, b_val)
        else:
            # 不一致 — 记录供人工审查。
            disagreement_record: Dict[str, Any] = {
                "field": field_name,
                "anno_a_value": str(a_val),
                "anno_b_value": str(b_val),
                "resolved": False,
                "resolved_by": "",
                "resolution": "",
            }
            disagreements.append(disagreement_record)
            logger.info(
                "Field '%s': DISAGREE (qwen='%s', gemma='%s')",
                field_name, a_val, b_val,
            )

    # 用（可能部分覆盖后的）字段值构建共识三元组。
    consensus = LegalTriplet(
        subject=Subject(
            text=consensus_subject_text,
            role=_parse_role(consensus_subject_role),
        ),
        action=Action(
            predicate=consensus_action_predicate,
            object=consensus_action_object,
        ),
        condition=Condition(
            text=consensus_condition_text,
            type=_parse_condition_type(consensus_condition_type),
        ),
    )

    total_fields = len(FIELD_SPEC)
    agreed_count = total_fields - len(disagreements)
    logger.info(
        "Consensus: %d/%d fields agreed, %d disagreements",
        agreed_count, total_fields, len(disagreements),
    )

    return consensus, disagreements


def resolve_disagreement(
    disagreement: Dict[str, Any],
    human_choice: str,
) -> Dict[str, Any]:
    """将人工裁决应用于分歧记录。

    人工审查者判定争议字段上哪份标注正确时，
    本函数记录该裁决。已解决的分歧可供
    build_gold_from_consensus() 生成最终金标准三元组。

    参数：
        disagreement: field_level_consensus() 返回的分歧 dict，
                      至少含 "field"、"anno_a_value"、"anno_b_value"。
        human_choice: 人工选定值，须为以下之一：
                      - anno_a_value（同意 Qwen）
                      - anno_b_value（同意 Gemma）
                      - 自定义字符串（人工给出不同值）

    返回：
        已填充分解决字段的更新后分歧 dict：
          - "resolved": True
          - "resolved_by": "human"
          - "resolution": 选定值
    """
    if not isinstance(disagreement, dict):
        logger.error("resolve_disagreement called with non-dict argument")
        return {"error": "Invalid disagreement record"}

    field = disagreement.get("field", "unknown")
    a_val = str(disagreement.get("anno_a_value", ""))
    b_val = str(disagreement.get("anno_b_value", ""))

    if human_choice == a_val:
        logger.info(
            "Resolution for '%s': human chose ANNO_A value '%s'", field, human_choice
        )
    elif human_choice == b_val:
        logger.info(
            "Resolution for '%s': human chose ANNO_B value '%s'", field, human_choice
        )
    else:
        logger.info(
            "Resolution for '%s': human provided custom value '%s' "
            "(anno_a='%s', anno_b='%s')",
            field, human_choice, a_val, b_val,
        )

    disagreement["resolved"] = True
    disagreement["resolved_by"] = "human"
    disagreement["resolution"] = human_choice

    return disagreement


def build_gold_from_consensus(
    clause_id: str,
    text: str,
    consensus: LegalTriplet,
    disagreements: List[Dict[str, Any]],
) -> LegalTriplet:
    """从共识构建最终金标准三元组，并应用人工解决结果。

    对每个争议字段：
      - 若分歧已解决（人工审查），用解决值
        替换暂定（anno_a）值。
      - 若仍未解决，保留 anno_a 作为暂定值。

    参数：
        clause_id: 条款标识（如 "LEXSPEC-001"）。仅用于
                   日志，不写入三元组。
        text: 原始条款文本。仅用于日志。
        consensus: field_level_consensus() 的共识三元组，
                   争议字段为 anno_a 取值。
        disagreements: 分歧记录列表，部分可能已通过
                       resolve_disagreement() 解决。

    返回：
        应用全部已解决取值后的最终金标准 LegalTriplet。
    """
    # 从共识取值开始（争议字段为 anno_a）。
    gold_subject_text = consensus.subject.text
    gold_subject_role = consensus.subject.role
    gold_action_predicate = consensus.action.predicate
    gold_action_object = consensus.action.object
    gold_condition_text = consensus.condition.text
    gold_condition_type = consensus.condition.type

    resolved_count = 0
    unresolved_count = 0

    for disagreement in disagreements:
        field = disagreement.get("field", "")
        is_resolved = disagreement.get("resolved", False)
        resolution = disagreement.get("resolution", "")

        if is_resolved and resolution:
            if field == "subject.text":
                gold_subject_text = resolution
            elif field == "subject.role":
                gold_subject_role = _parse_role(resolution)
            elif field == "action.predicate":
                gold_action_predicate = resolution
            elif field == "action.object":
                gold_action_object = resolution
            elif field == "condition.text":
                gold_condition_text = resolution
            elif field == "condition.type":
                gold_condition_type = _parse_condition_type(resolution)
            else:
                logger.warning(
                    "Unknown field '%s' in disagreement -- skipping resolution", field
                )
                continue

            resolved_count += 1
            logger.debug(
                "Applied resolution for '%s': '%s'", field, resolution
            )
        else:
            unresolved_count += 1
            logger.debug(
                "Field '%s' remains unresolved -- keeping anno_a value", field
            )

    logger.info(
        "Gold construction for clause '%s': %d resolved, %d unresolved (tentative)",
        clause_id, resolved_count, unresolved_count,
    )

    gold = LegalTriplet(
        subject=Subject(text=gold_subject_text, role=gold_subject_role),
        action=Action(
            predicate=gold_action_predicate,
            object=gold_action_object,
        ),
        condition=Condition(
            text=gold_condition_text,
            type=gold_condition_type,
        ),
    )

    return gold
