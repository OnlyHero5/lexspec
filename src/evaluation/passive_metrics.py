"""
被动语态恢复准确率指标。

评估系统是否在被动结构中正确识别逻辑主语（obl:agent）。
LLM 抽取系统的已知弱点：常将句法主语（nsubj:pass，受事）
当作法律主语。
"""

from __future__ import annotations

from typing import Optional, List, Dict

from src.extraction.schema import LegalTriplet, DependencyTree
from src.evaluation.text_normalizer import normalize
from src.utils.progress import progress_bar
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_passive_recovery_accuracy(
    predictions: List[LegalTriplet],
    trees: List[DependencyTree],
    gold: List[LegalTriplet],
) -> Dict[str, float]:
    """计算被动语态论元恢复准确率。

    法律合同中的被动语态是抽取难点：
    在 "The Price shall be paid by Buyer" 中，句法主语为 "Price"
    （nsubj:pass），法律施事（义务方）为 "Buyer"（obl:agent）。
    LLM 常误将受事当作主体方。

    本指标：
      1. 识别 UD 树检测到被动语的子句
         （存在 nsubj:pass 与 aux:pass）。
      2. 检查预测是否将施事（obl:agent）正确识别为主语，
         而非受事。
      3. 报告恢复准确率与假施事率。

    仅在检测到被动语的子句上评估 — 主动句排除。

    参数:
        predictions: 预测 LegalTriplet 列表。
        trees: DependencyTree 列表（等长）。
        gold: 金标准 LegalTriplet 列表（等长）。

    返回:
        字典，键包括：
        - passive_count: int — 被动子句数。
        - recovery_accuracy: float — 主语被正确识别为施事的比例
          （与金标主语文本匹配且角色正确）。
        - false_agent_rate: float — 预测将受事（nsubj:pass）当作主语、
          本应为施事（obl:agent）的比例。被动语态的关键错误率。
        - passive_f1_impact: float — 被动子集 F1 相对总体 F1 的影响，
          反映被动对性能的拖累。

    异常:
        ValueError: 输入列表长度不一致。
    """
    n = len(predictions)
    if n != len(trees) or n != len(gold):
        raise ValueError(
            f"All input lists must have the same length. "
            f"Got predictions={len(predictions)}, trees={len(trees)}, gold={len(gold)}."
        )

    passive_indices: List[int] = []
    passive_pred_subjects: List[str] = []
    passive_gold_subjects: List[str] = []
    passive_pred_roles: List[str] = []
    passive_gold_roles: List[str] = []
    passive_agent_tokens: List[Optional[str]] = []  # 树中 obl:agent 文本

    for i, (pred, tree, g) in enumerate(
        progress_bar(
            zip(predictions, trees, gold),
            desc="Detecting passive clauses",
            unit="sample",
            total=n,
        )
    ):
        # 检测被动：须同时有 nsubj:pass 与 aux:pass。
        has_nsubj_pass = tree.has_deprel("nsubj:pass")
        has_aux_pass = tree.has_deprel("aux:pass")

        if has_nsubj_pass and has_aux_pass:
            passive_indices.append(i)

            # 收集预测与金标主语信息。
            passive_pred_subjects.append(normalize(pred.subject.text))
            passive_gold_subjects.append(normalize(g.subject.text))
            passive_pred_roles.append(pred.subject.role.value)
            passive_gold_roles.append(g.subject.role.value)

            # 从树提取 obl:agent 词元文本（真施事）。
            agent_tokens = tree.find_tokens_by_deprel("obl:agent")
            agent_text = normalize(agent_tokens[0].text) if agent_tokens else None
            passive_agent_tokens.append(agent_text)

    passive_count = len(passive_indices)
    if passive_count == 0:
        logger.info("No passive voice clauses detected in this dataset.")
        return {
            "passive_count": 0,
            "recovery_accuracy": 0.0,
            "false_agent_rate": 0.0,
            "passive_f1_impact": 0.0,
        }

    # 计算恢复准确率：是否识别到正确施事？
    correct_recoveries = 0
    false_agent_count = 0

    # 获取 nsubj:pass 词元供假施事检测。
    for idx in progress_bar(
        passive_indices, desc="Passive recovery scoring", unit="clause",
    ):
        pred = predictions[idx]
        tree = trees[idx]
        g = gold[idx]

        # 检查主语文本是否与金标匹配（归一化词元级）。
        pred_text = normalize(pred.subject.text)
        gold_text = normalize(g.subject.text)

        # 主语文本恢复的词元重叠 F1。
        pred_tokens = set(pred_text.split())
        gold_tokens = set(gold_text.split())
        if pred_tokens and gold_tokens:
            overlap = pred_tokens & gold_tokens
            f1 = 2 * len(overlap) / (len(pred_tokens) + len(gold_tokens)) if (len(pred_tokens) + len(gold_tokens)) > 0 else 0.0
        else:
            f1 = 1.0 if not pred_tokens and not gold_tokens else 0.0

        # 成功恢复：与金标主语高词元重叠且角色匹配。
        if f1 >= 0.8 and pred.subject.role == g.subject.role:
            correct_recoveries += 1

        # 假施事检测：预测是否使用 nsubj:pass（受事）作为主语？
        # 检查预测主语是否匹配 nsubj:pass 文本。
        nsubj_pass_tokens = tree.find_tokens_by_deprel("nsubj:pass")
        if nsubj_pass_tokens and pred_text:
            nsubj_pass_text = normalize(nsubj_pass_tokens[0].text)
            nsubj_pass_set = set(nsubj_pass_text.split())
            overlap = pred_tokens & nsubj_pass_set
            if overlap and len(overlap) / max(len(pred_tokens), 1) >= 0.5:
                # 预测主要匹配受事 — 可能为假施事错误。
                false_agent_count += 1

    recovery_accuracy = correct_recoveries / passive_count
    false_agent_rate = false_agent_count / passive_count

    logger.info(
        "Passive recovery: %d passive clauses, accuracy=%.4f, false_agent=%.4f",
        passive_count, recovery_accuracy, false_agent_rate,
    )

    # 粗略 F1 影响：被动子集 F1 vs 总体。
    # 简化估计；完整计算需逐样本 F1。
    passive_f1_impact = 0.0  # 预留与逐样本 F1 集成。

    return {
        "passive_count": passive_count,
        "recovery_accuracy": recovery_accuracy,
        "false_agent_rate": false_agent_rate,
        "passive_f1_impact": passive_f1_impact,
    }
