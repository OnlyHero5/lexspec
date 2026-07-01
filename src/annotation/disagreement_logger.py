"""
标注质量跟踪用的分歧日志
==========================
在金标准测试集构建流水线中，记录两个标注模型
（Qwen3.6 27B 与 Gemma4 31B）之间的标注分歧。

每个分歧事件记为 AnnotationDisagreement
Pydantic 模型（定义于 src/extraction/schema.py），并序列化为
JSONL，供下游分析、报告与质量监控。

分歧日志用途：
  1. **审计轨迹**：记录每条分歧及其解决方式，
     实现从原始标注到最终金标标签的完整可追溯性。
  2. **质量监控**：分歧模式揭示标注模型的系统性弱点
     （例如某模型持续误识被动语态主语）。
  3. **人工审查队列**：未解决分歧构成人工标注者裁决的工作清单。
  4. **标注者间一致性报告**：该日志是计算 Cohen's kappa、
     Krippendorff's alpha 等指标的主要数据源
     （见 src/evaluation/）。

用法：
    from src.annotation.disagreement_logger import log_disagreement
    from src.annotation.disagreement_io import save_disagreement_log

    record = log_disagreement(
        clause_id="LEXSPEC-001",
        text="Seller shall deliver the Goods.",
        anno_a=qwen_triplet,
        anno_b=gemma_triplet,
        disagreements=disagreement_list,
        resolution="Human chose anno_a for subject.role",
    )

    save_disagreement_log([record], "data/processed/annotation_log.jsonl")
"""

from __future__ import annotations

from typing import List, Optional

from src.extraction.schema import (
    LegalTriplet,
    Subject,
    Action,
    Condition,
    LegalRole,
    ConditionType,
    AnnotationDisagreement,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def log_disagreement(
    clause_id: str,
    text: str,
    anno_a: LegalTriplet,
    anno_b: LegalTriplet,
    disagreements: List[dict],
    resolution: Optional[str] = None,
) -> AnnotationDisagreement:
    """从分歧事件创建 AnnotationDisagreement 记录。

    捕获两模型标注分歧的完整上下文：原始条款文本、
    两模型标注、具体不一致字段，以及已应用的
    解决方式（人工或自动）。

    返回的 AnnotationDisagreement 为 Pydantic 模型 — 带完整
    模式校验，可通过 model_dump(mode="json") 直接序列化
    以持久化到 JSONL。

    参数：
        clause_id: 唯一条款标识（如 "LEXSPEC-001"）。
                   将分歧关联回源文档。
        text: 被标注的完整条款文本。审查分歧时保留备查。
        anno_a: 第一模型标注（通常为 Qwen3.6 27B）。
        anno_b: 第二模型标注（通常为 Gemma4 31B）。
        disagreements: 来自 field_level_consensus() 的字段级分歧记录列表。
                       每项至少含：field、anno_a_value、anno_b_value。
                       也可含：resolved、resolved_by、resolution。
        resolution: 可选的整体解决说明（如「全部由人工审查解决」、
                    「条件字段自动解决」）。

    返回：
        经完整校验的 AnnotationDisagreement Pydantic 实例。
    """
    # 最小输入校验 — 构造 AnnotationDisagreement 时
    # Pydantic 会捕获模式问题。
    if not clause_id:
        logger.warning("log_disagreement called with empty clause_id")
    if not text:
        logger.warning("log_disagreement called with empty clause text")
    if not disagreements:
        logger.debug(
            "log_disagreement called with empty disagreements list for clause '%s'",
            clause_id,
        )

    # 规范化分歧记录，确保包含预期键。
    # 部分调用方可能只提供部分字段。
    normalized_disagreements: List[dict] = []
    for i, d in enumerate(disagreements):
        if not isinstance(d, dict):
            logger.warning(
                "Disagreement item %d is not a dict (type=%s) -- skipping",
                i, type(d).__name__,
            )
            continue
        normalized = {
            "field": d.get("field", f"unknown_{i}"),
            "anno_a_value": str(d.get("anno_a_value", "")),
            "anno_b_value": str(d.get("anno_b_value", "")),
            "resolved": bool(d.get("resolved", False)),
            "resolved_by": str(d.get("resolved_by", "")),
            "resolution_text": str(d.get("resolution", d.get("resolution_text", ""))),
        }
        normalized_disagreements.append(normalized)

    # 为本分歧记录构建最终金标准三元组。
    final_gold = _build_tentative_gold(anno_a, normalized_disagreements)

    # 构造 Pydantic 模型。
    try:
        record = AnnotationDisagreement(
            clause_id=clause_id,
            text=text,
            qwen_annotation=anno_a,
            gemma_annotation=anno_b,
            disagreement_fields=normalized_disagreements,
            final_gold=final_gold,
        )
    except Exception as exc:
        logger.error(
            "Failed to construct AnnotationDisagreement for clause '%s': %s",
            clause_id, exc,
        )
        raise ValueError(
            f"Invalid AnnotationDisagreement data for clause '{clause_id}'"
        ) from exc

    # 按适当级别记录分歧事件。
    unresolved_count = sum(
        1 for d in normalized_disagreements if not d["resolved"]
    )
    resolved_count = len(normalized_disagreements) - unresolved_count

    if unresolved_count > 0:
        logger.info(
            "Disagreement logged for '%s': %d fields, %d resolved, %d unresolved%s",
            clause_id,
            len(normalized_disagreements),
            resolved_count,
            unresolved_count,
            f" -- resolution note: {resolution}" if resolution else "",
        )
    else:
        logger.debug(
            "Disagreement logged for '%s': all %d fields resolved",
            clause_id, len(normalized_disagreements),
        )

    return record


def _build_tentative_gold(
    anno_a: LegalTriplet,
    disagreements: List[dict],
) -> LegalTriplet:
    """从 anno_a 构建暂定金标准三元组，并应用已有解决结果。

    默认以 anno_a 取值（Qwen 为主模型）。对已解决且
    解决值与 anno_b 一致的分歧，用 anno_b 覆盖对应字段。

    据此得到当前解决状态下尽力而为的金标准三元组。

    参数：
        anno_a: 主模型标注（作为基底）。
        disagreements: 规范化后的分歧记录列表。

    返回：
        反映当前解决状态的最佳努力金标准 LegalTriplet。
        未解决字段保留 anno_a 取值。
    """
    # 默认以 anno_a 取值。
    gold_subject_text = anno_a.subject.text
    gold_subject_role = anno_a.subject.role
    gold_action_predicate = anno_a.action.predicate
    gold_action_object = anno_a.action.object
    gold_condition_text = anno_a.condition.text
    gold_condition_type = anno_a.condition.type

    for d in disagreements:
        if not d.get("resolved", False):
            # 未解决 — 保留 anno_a（已是默认）。
            continue

        field = d.get("field", "")
        resolution = d.get("resolution_text", d.get("resolution", ""))

        if not resolution:
            # 标记已解决但值为空 — 跳过。
            logger.debug(
                "Disagreement for '%s' marked resolved but has empty resolution",
                field,
            )
            continue

        # 将解决值写入对应字段。
        if field == "subject.text":
            gold_subject_text = resolution
        elif field == "subject.role":
            # 将角色字符串解析为 LegalRole 枚举。
            try:
                gold_subject_role = LegalRole(resolution)
            except ValueError:
                logger.warning(
                    "Invalid role resolution '%s' for subject.role -- keeping anno_a value",
                    resolution,
                )
        elif field == "action.predicate":
            gold_action_predicate = resolution
        elif field == "action.object":
            gold_action_object = resolution
        elif field == "condition.text":
            gold_condition_text = resolution
        elif field == "condition.type":
            try:
                gold_condition_type = ConditionType(resolution)
            except ValueError:
                logger.warning(
                    "Invalid condition type resolution '%s' -- keeping anno_a value",
                    resolution,
                )
        else:
            logger.warning(
                "Unknown field '%s' in disagreement -- cannot apply resolution",
                field,
            )

    return LegalTriplet(
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
