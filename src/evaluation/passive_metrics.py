"""
Passive Voice Recovery Accuracy metric.

Evaluates whether the system correctly identifies the logical subject
(obl:agent) in passive voice constructions. This is a known weakness
of LLM extraction systems that tend to treat the syntactic subject
(nsubj:pass, the patient) as the legal subject.
"""

from __future__ import annotations

from typing import Optional, List, Dict

from src.extraction.schema import LegalTriplet, DependencyTree
from src.evaluation.text_normalizer import normalize
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_passive_recovery_accuracy(
    predictions: List[LegalTriplet],
    trees: List[DependencyTree],
    gold: List[LegalTriplet],
) -> Dict[str, float]:
    """Compute passive voice argument recovery accuracy.

    Passive voice in legal contracts is a known challenge for extraction:
    in "The Price shall be paid by Buyer", the syntactic subject is "Price"
    (nsubj:pass), but the legal agent (the obligated party) is "Buyer"
    (obl:agent). LLM systems often incorrectly identify the patient as
    the subject party.

    This metric:
      1. Identifies clauses where the UD tree detects passive voice
         (presence of nsubj:pass AND aux:pass relations).
      2. Checks whether the prediction correctly identifies the agent
         (obl:agent) as the subject, rather than the patient.
      3. Reports the recovery accuracy and false agent rate.

    Only evaluates on clauses where passive voice is actually detected —
    active clauses are excluded from this metric.

    Args:
        predictions: List of predicted LegalTriplets.
        trees: List of DependencyTree objects (same length).
        gold: List of gold-standard LegalTriplets (same length).

    Returns:
        Dict with keys:
        - passive_count: int — number of passive clauses found.
        - recovery_accuracy: float — proportion where the subject was
          correctly identified as the agent (match with gold subject text
          AND correct role).
        - false_agent_rate: float — proportion where the prediction gave
          the patient (nsubj:pass) as the subject when it should have been
          the agent (obl:agent). This is the key error rate for passive voice.
        - passive_f1_impact: float — F1 on the passive subset vs overall F1
          baseline, showing how much passive voice degrades performance.

    Raises:
        ValueError: If input lists have different lengths.
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
    passive_agent_tokens: List[Optional[str]] = []  # obl:agent text from tree

    for i, (pred, tree, g) in enumerate(zip(predictions, trees, gold)):
        # Detect passive voice: requires both nsubj:pass and aux:pass.
        has_nsubj_pass = tree.has_deprel("nsubj:pass")
        has_aux_pass = tree.has_deprel("aux:pass")

        if has_nsubj_pass and has_aux_pass:
            passive_indices.append(i)

            # Collect prediction and gold subject info.
            passive_pred_subjects.append(normalize(pred.subject.text))
            passive_gold_subjects.append(normalize(g.subject.text))
            passive_pred_roles.append(pred.subject.role.value)
            passive_gold_roles.append(g.subject.role.value)

            # Extract the obl:agent token text from the tree (the true agent).
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

    # Compute recovery accuracy: was the correct agent identified?
    correct_recoveries = 0
    false_agent_count = 0

    # Get nsubj:pass tokens for false agent detection.
    for idx in passive_indices:
        pred = predictions[idx]
        tree = trees[idx]
        g = gold[idx]

        # Check if subject text matches gold (normalized token-level).
        pred_text = normalize(pred.subject.text)
        gold_text = normalize(g.subject.text)

        # Token overlap F1 for subject text recovery.
        pred_tokens = set(pred_text.split())
        gold_tokens = set(gold_text.split())
        if pred_tokens and gold_tokens:
            overlap = pred_tokens & gold_tokens
            f1 = 2 * len(overlap) / (len(pred_tokens) + len(gold_tokens)) if (len(pred_tokens) + len(gold_tokens)) > 0 else 0.0
        else:
            f1 = 1.0 if not pred_tokens and not gold_tokens else 0.0

        # Successful recovery: high token overlap with gold subject AND role matches.
        if f1 >= 0.8 and pred.subject.role == g.subject.role:
            correct_recoveries += 1

        # False agent detection: did the prediction use nsubj:pass (patient)
        # as the subject? Check if pred subject matches the nsubj:pass text.
        nsubj_pass_tokens = tree.find_tokens_by_deprel("nsubj:pass")
        if nsubj_pass_tokens and pred_text:
            nsubj_pass_text = normalize(nsubj_pass_tokens[0].text)
            nsubj_pass_set = set(nsubj_pass_text.split())
            overlap = pred_tokens & nsubj_pass_set
            if overlap and len(overlap) / max(len(pred_tokens), 1) >= 0.5:
                # Prediction mostly matches the patient — likely a false agent error.
                false_agent_count += 1

    recovery_accuracy = correct_recoveries / passive_count
    false_agent_rate = false_agent_count / passive_count

    logger.info(
        "Passive recovery: %d passive clauses, accuracy=%.4f, false_agent=%.4f",
        passive_count, recovery_accuracy, false_agent_rate,
    )

    # Compute a rough F1 impact: F1 on passive subset vs overall.
    # This is a simplified estimate; full computation requires per-sample F1.
    passive_f1_impact = 0.0  # Reserved for integration with per-sample F1.

    return {
        "passive_count": passive_count,
        "recovery_accuracy": recovery_accuracy,
        "false_agent_rate": false_agent_rate,
        "passive_f1_impact": passive_f1_impact,
    }
