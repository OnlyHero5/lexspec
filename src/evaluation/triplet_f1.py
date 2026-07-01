"""
加权三元组 F1 计算 — 主任务评估指标。

在 5 个三元组字段上计算加权 F1，权重可配置：
  subject.text:    0.35 — 最高；主语归属是核心错误来源
  subject.role:    0.10 — 分类准确率（枚举精确匹配）
  action.predicate: 0.20 — 词元级 F1（词形基形式）
  action.object:    0.20 — 词元级片段 F1
  condition.text:   0.15 — 词元级片段 F1

理论:
  法律信息抽取是结构化预测任务。单一 F1 数值不足，因为不同错误的代价不同
  （误认义务方后果严重；条件边界模糊相对次要）。加权 F1 提供有原则的可分解度量。

  词元级匹配对部分正确的抽取给予部分得分。
  "the Goods sold" 与 "the Goods" 的精确率为 2/3、召回率为 2/2，
  得到 F1 ≈ 0.80，而非精确不匹配时的 0。

设计:
  - 全部文本比较经 text_normalizer.py 的 normalize() 处理。
  - 权重和为 1.0；由 compute_triplet_f1() 中的归一化保证。
  - 可计算逐样本 F1，供统计显著性检验使用。
  - 默认权重与 constraints.yaml 的 f1_weights 节一致。
"""

from __future__ import annotations

from typing import Optional, Dict, List

from src.extraction.schema import LegalTriplet
from src.evaluation.text_normalizer import normalize
from src.evaluation.field_f1 import load_f1_weights, compute_field_f1
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_triplet_f1(
    predictions: List[LegalTriplet],
    gold: List[LegalTriplet],
    weights: Optional[Dict[str, float]] = None,
    party_aliases: Optional[Dict[str, List[str]]] = None,
    constraints_path: str = "configs/constraints.yaml",
) -> Dict[str, float]:
    """在全部 5 个字段上计算加权三元组 F1。

    对 5 个字段分别：
    - subject.text:     归一化后的词元级 F1。
                         将主语文本视为词元集合，计算词元重叠的精确率/召回率/F1，
                         给予部分得分。
    - subject.role:     分类准确率 — 枚举精确匹配。
                         角色为离散值，无部分得分。
    - action.predicate: 词形基形式上的词元级 F1。对 "deliver" 与 "shall deliver"
                         等近似匹配给予部分得分。
    - action.object:    词元级片段 F1，同样采用词元重叠。
    - condition.text:   词元级片段 F1。边界错误可获部分得分。

    总体 F1 = 各字段 F1 的加权平均，权重归一化后和为 1.0。

    输入列表须等长（1:1 对齐）。若长度不同，仅评估重叠前缀（并发出警告）。

    参数:
        predictions: 系统预测的 LegalTriplet 列表。
        gold: 金标准 LegalTriplet 列表（与 predictions 等长）。
        weights: 可选权重覆盖字典。为 None 时从 constraints.yaml 加载。
        party_aliases: 比较时 normalize() 使用的当事方别名映射。
        constraints_path: weights 为 None 时的 constraints YAML 路径。

    返回:
        字典，键包括：
        - subject_text_f1, subject_text_precision, subject_text_recall
        - subject_role_acc
        - predicate_f1, predicate_precision, predicate_recall
        - object_f1, object_precision, object_recall
        - condition_f1, condition_precision, condition_recall
        - overall_f1（5 个字段 F1 的加权平均）

    异常:
        ValueError: predictions 与 gold 长度不一致（截断警告后）或两者均为空。
    """
    n_pred = len(predictions)
    n_gold = len(gold)

    if n_pred == 0 and n_gold == 0:
        raise ValueError("Cannot compute F1 on empty prediction and gold lists.")

    if n_pred != n_gold:
        min_len = min(n_pred, n_gold)
        logger.warning(
            "Length mismatch: predictions=%d, gold=%d. Evaluating on first %d pairs.",
            n_pred, n_gold, min_len,
        )
        predictions = predictions[:min_len]
        gold = gold[:min_len]

    n = len(predictions)

    # 归一化权重：确保和为 1.0。
    w = dict(weights) if weights is not None else load_f1_weights(constraints_path)
    weight_sum = sum(w.values())
    if abs(weight_sum - 1.0) > 1e-9:
        logger.debug("Normalizing weights from sum=%.4f to 1.0", weight_sum)
        w = {k: v / weight_sum for k, v in w.items()}

    # -------------------------------------------------------------------
    # 提取并归一化全部样本的字段值
    # -------------------------------------------------------------------

    # subject.text：两侧归一化。
    pred_subject_texts = [
        normalize(p.subject.text, party_aliases=party_aliases)
        for p in predictions
    ]
    gold_subject_texts = [
        normalize(g.subject.text, party_aliases=party_aliases)
        for g in gold
    ]

    # subject.role：提取枚举值为字符串以便比较。
    pred_subject_roles = [p.subject.role.value for p in predictions]
    gold_subject_roles = [g.subject.role.value for g in gold]

    # action.predicate：归一化供词元级比较。
    pred_predicates = [
        normalize(p.action.predicate, party_aliases=party_aliases)
        for p in predictions
    ]
    gold_predicates = [
        normalize(g.action.predicate, party_aliases=party_aliases)
        for g in gold
    ]

    # action.object：归一化。
    pred_objects = [
        normalize(p.action.object, party_aliases=party_aliases)
        for p in predictions
    ]
    gold_objects = [
        normalize(g.action.object, party_aliases=party_aliases)
        for g in gold
    ]

    # condition.text：归一化。NONE 条件为空字符串。
    pred_conditions = [
        normalize(p.condition.text, party_aliases=party_aliases)
        for p in predictions
    ]
    gold_conditions = [
        normalize(g.condition.text, party_aliases=party_aliases)
        for g in gold
    ]

    # -------------------------------------------------------------------
    # 计算各字段指标
    # -------------------------------------------------------------------

    # 主语文本：词元级 F1。
    st_precision, st_recall, st_f1 = compute_field_f1(
        pred_subject_texts, gold_subject_texts, match_type="token", show_progress=False,
    )

    # 主语角色：分类准确率。
    role_correct = sum(1 for p, g in zip(pred_subject_roles, gold_subject_roles) if p == g)
    role_acc = role_correct / n if n > 0 else 0.0

    # 谓词：词元级 F1。
    pr_precision, pr_recall, pr_f1 = compute_field_f1(
        pred_predicates, gold_predicates, match_type="token", show_progress=False,
    )

    # 宾语：词元级 F1。
    ob_precision, ob_recall, ob_f1 = compute_field_f1(
        pred_objects, gold_objects, match_type="token", show_progress=False,
    )

    # 条件：词元级 F1。
    co_precision, co_recall, co_f1 = compute_field_f1(
        pred_conditions, gold_conditions, match_type="token", show_progress=False,
    )

    # -------------------------------------------------------------------
    # 计算加权总体 F1
    # -------------------------------------------------------------------
    # 保留 10 位小数以抑制浮点表示伪影（例如 IEEE 754 下各分量为 1.0 时
    # 0.35 + 0.10 + 0.20 + 0.20 + 0.15 = 0.9999999999999999）。语义上为 1.0。
    overall = round(
        w["subject_text"] * st_f1
        + w["subject_role"] * role_acc
        + w["predicate"] * pr_f1
        + w["object"] * ob_f1
        + w["condition"] * co_f1,
        10,
    )

    logger.info(
        "Weighted triplet F1: overall=%.4f (n=%d, st=%.3f, role=%.3f, pred=%.3f, obj=%.3f, cond=%.3f)",
        overall, n, st_f1, role_acc, pr_f1, ob_f1, co_f1,
    )

    return {
        "subject_text_f1": st_f1,
        "subject_text_precision": st_precision,
        "subject_text_recall": st_recall,
        "subject_role_acc": role_acc,
        "predicate_f1": pr_f1,
        "predicate_precision": pr_precision,
        "predicate_recall": pr_recall,
        "object_f1": ob_f1,
        "object_precision": ob_precision,
        "object_recall": ob_recall,
        "condition_f1": co_f1,
        "condition_precision": co_precision,
        "condition_recall": co_recall,
        "overall_f1": overall,
    }
