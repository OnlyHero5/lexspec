"""
字段级 F1 计算（词元级匹配）。

权重通过 load_f1_weights() 从 configs/constraints.yaml 加载。
"""

from __future__ import annotations

from typing import Optional, Dict, List, Tuple

from src.extraction.schema import LegalTriplet
from src.evaluation.text_normalizer import normalize
from src.utils.constraints import get_f1_weights, load_constraints_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_f1_weights(constraints_path: str = "configs/constraints.yaml") -> Dict[str, float]:
    """从 constraints YAML 加载加权 F1 各分量权重。"""
    config = load_constraints_config(constraints_path)
    return get_f1_weights(config, constraints_path)


def compute_field_f1(
    preds: List[str],
    golds: List[str],
    match_type: str = "token",
    weights: Optional[Dict[str, float]] = None,
    constraints_path: str = "configs/constraints.yaml",
    party_aliases: Optional[Dict[str, List[str]]] = None,
) -> Tuple[float, float, float]:
    """计算单个文本字段的精确率、召回率与 F1。"""
    if len(preds) != len(golds):
        raise ValueError(
            f"Prediction/gold length mismatch: {len(preds)} vs {len(golds)}"
        )

    if match_type == "exact":
        tp = sum(
            1 for p, g in zip(preds, golds)
            if normalize(p, party_aliases=party_aliases) == normalize(g, party_aliases=party_aliases)
        )
        precision = tp / len(preds) if preds else 0.0
        recall = tp / len(golds) if golds else 0.0
    elif match_type == "token":
        total_prec = total_rec = 0.0
        for pred, gold in zip(preds, golds):
            p, r, _ = token_f1(pred, gold, party_aliases=party_aliases)
            total_prec += p
            total_rec += r
        precision = total_prec / len(preds) if preds else 0.0
        recall = total_rec / len(golds) if golds else 0.0
    else:
        raise ValueError(f"Unknown match_type: {match_type}")

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def token_f1(
    pred: str,
    gold: str,
    party_aliases: Optional[Dict[str, List[str]]] = None,
) -> Tuple[float, float, float]:
    """归一化后两个文本片段之间的词元级 F1。"""
    pred_tokens = set(normalize(pred, party_aliases=party_aliases).split())
    gold_tokens = set(normalize(gold, party_aliases=party_aliases).split())

    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    overlap = pred_tokens & gold_tokens
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def compute_per_sample_f1(
    predictions: List[LegalTriplet],
    gold: List[LegalTriplet],
    weights: Optional[Dict[str, float]] = None,
    constraints_path: str = "configs/constraints.yaml",
    party_aliases: Optional[Dict[str, List[str]]] = None,
) -> List[float]:
    """计算逐样本加权 F1 得分，用于显著性检验。"""
    w = weights if weights is not None else load_f1_weights(constraints_path)
    total_weight = sum(w.values())
    w_norm = {k: v / total_weight for k, v in w.items()}

    scores: List[float] = []
    for pred, g in zip(predictions, gold):
        field_scores: Dict[str, float] = {}

        _, _, field_scores["subject_text"] = token_f1(
            pred.subject.text, g.subject.text, party_aliases=party_aliases
        )
        field_scores["subject_role"] = (
            1.0 if pred.subject.role == g.subject.role else 0.0
        )
        _, _, field_scores["predicate"] = token_f1(
            pred.action.predicate, g.action.predicate, party_aliases=party_aliases
        )
        _, _, field_scores["object"] = token_f1(
            pred.action.object, g.action.object, party_aliases=party_aliases
        )
        _, _, field_scores["condition"] = token_f1(
            pred.condition.text, g.condition.text, party_aliases=party_aliases
        )

        sample_f1 = sum(w_norm[k] * field_scores[k] for k in w_norm)
        scores.append(sample_f1)

    return scores
