"""
Linguistic error classification — primary category determination.

Analyzes the UD tree to determine which linguistic phenomenon caused the
extraction error. Implements the 5 priority-ordered rules from the
two-level classification system.
"""

from __future__ import annotations

from typing import List, Dict, Any

from src.extraction.schema import (
    LegalTriplet, DependencyTree, ErrorCategory,
)
from src.evaluation.text_normalizer import normalize


def determine_primary_category(
    prediction: LegalTriplet,
    gold: LegalTriplet,
    tree: DependencyTree,
    field_errors: List[str],
) -> tuple[ErrorCategory, Dict[str, Any]]:
    """Analyze UD tree to determine the linguistic phenomenon causing the error.

    Checks each phenomenon rule in priority order and returns the first match.
    Collects UD evidence for generating the linguistic explanation.

    Args:
        prediction: System prediction.
        gold: Gold standard.
        tree: UD dependency tree.
        field_errors: Pre-detected field errors.

    Returns:
        Tuple of (primary ErrorCategory, dict of UD evidence for explanation).
    """
    evidence: Dict[str, Any] = {}

    # -------------------------------------------------------------------
    # Rule 1: Passive Voice Error
    # -------------------------------------------------------------------
    # Check for passive voice: nsubj:pass identifies the patient (syntactic
    # subject), which in passive voice is the logical object. The LLM may
    # have extracted the patient instead of the real agent.
    has_nsubj_pass = tree.has_deprel("nsubj:pass")
    has_aux_pass = tree.has_deprel("aux:pass")
    has_obl_agent = tree.has_deprel("obl:agent")

    if has_nsubj_pass and has_aux_pass:
        # Passive voice detected. Check if the error involves the subject.
        if any(e.startswith("subject") for e in field_errors):
            # Collect evidence.
            nsubj_pass_tokens = tree.find_tokens_by_deprel("nsubj:pass")
            aux_pass_tokens = tree.find_tokens_by_deprel("aux:pass")
            obl_agent_tokens = tree.find_tokens_by_deprel("obl:agent")

            evidence["passive_detected"] = True
            evidence["nsubj_pass_tokens"] = [
                {"text": t.text, "index": t.index, "lemma": t.lemma}
                for t in nsubj_pass_tokens
            ]
            evidence["aux_pass_tokens"] = [
                {"text": t.text, "index": t.index}
                for t in aux_pass_tokens
            ]
            if obl_agent_tokens:
                evidence["obl_agent_tokens"] = [
                    {"text": t.text, "index": t.index, "lemma": t.lemma}
                    for t in obl_agent_tokens
                ]
            evidence["pred_subject"] = prediction.subject.text
            evidence["gold_subject"] = gold.subject.text

            return ErrorCategory.PASSIVE_VOICE, evidence

    # -------------------------------------------------------------------
    # Rule 2: Conditional Boundary Error
    # -------------------------------------------------------------------
    has_advcl = tree.has_deprel("advcl")
    has_mark = tree.has_deprel("mark")

    if has_advcl and has_mark:
        # Condition clause exists. Check if error involves condition.
        if any(e.startswith("condition") for e in field_errors):
            advcl_tokens = tree.find_tokens_by_deprel("advcl")
            mark_tokens = tree.find_tokens_by_deprel("mark")

            evidence["advcl_detected"] = True
            evidence["advcl_tokens"] = [
                {"text": t.text, "index": t.index, "lemma": t.lemma}
                for t in advcl_tokens
            ]
            evidence["mark_tokens"] = [
                {"text": t.text, "index": t.index, "lemma": t.lemma}
                for t in mark_tokens
            ]

            # Compute predicted vs UD condition span tokens for comparison.
            for advcl_token in advcl_tokens:
                span = tree.get_subtree_span(advcl_token.index)
                evidence.setdefault("ud_condition_spans", []).append({
                    "text": span.text,
                    "deprel": span.deprel,
                    "token_count": len(span.tokens),
                })

            evidence["pred_condition"] = prediction.condition.text
            evidence["gold_condition"] = gold.condition.text

            return ErrorCategory.CONDITIONAL_BOUNDARY, evidence

    # -------------------------------------------------------------------
    # Rule 3: Relative Clause Confusion
    # -------------------------------------------------------------------
    has_relcl = tree.has_deprel("acl:relcl")

    if has_relcl:
        # Check if the error might be due to relative clause confusion.
        # This is harder to detect deterministically; we check if the
        # predicate or object text tokens appear inside a relative clause.
        relcl_tokens = tree.find_tokens_by_deprel("acl:relcl")

        for relcl_token in relcl_tokens:
            relcl_subtree = tree.get_subtree_tokens(relcl_token.index)
            relcl_lemma_set = {t.lemma.lower() for t in relcl_subtree}

            # Check if the predicted predicate or object lemmas appear in
            # a relative clause subtree. If so, the LLM may have extracted
            # from the relative clause instead of the main clause.
            pred_lemma_set = set(normalize(prediction.action.predicate).lower().split())
            obj_lemma_set = set(normalize(prediction.action.object).lower().split())

            pred_in_relcl = bool(pred_lemma_set & relcl_lemma_set)
            obj_in_relcl = bool(obj_lemma_set & relcl_lemma_set)

            if pred_in_relcl or obj_in_relcl:
                evidence["relcl_detected"] = True
                evidence["relcl_head"] = {
                    "text": relcl_token.text,
                    "index": relcl_token.index,
                    "lemma": relcl_token.lemma,
                }
                evidence["relcl_subtree_text"] = " ".join(
                    t.text for t in relcl_subtree
                )
                evidence["predicate_in_relcl"] = pred_in_relcl
                evidence["object_in_relcl"] = obj_in_relcl
                evidence["pred_predicate"] = prediction.action.predicate
                evidence["pred_object"] = prediction.action.object

                return ErrorCategory.RELATIVE_CLAUSE, evidence

    # -------------------------------------------------------------------
    # Rule 4: Long-distance Dependency Error
    # -------------------------------------------------------------------
    # Measure the dependency distance from the root predicate to its subject
    # and object. If distance > 3 and there are errors on those arguments,
    # classify as long-distance dependency error.
    root_idx = tree.root_index
    if root_idx is not None:
        root_token = tree.get_token(root_idx)
        if root_token and root_token.upos == "VERB":
            # Find subject and object dependents of the root.
            subjects = tree.get_children(root_idx, deprel="nsubj") + tree.get_children(root_idx, deprel="nsubj:pass")
            objects = tree.get_children(root_idx, deprel="obj")

            max_distance = 0
            # Check subject distance.
            for subj in subjects:
                dist = tree.get_dependency_distance(subj.index, root_idx)
                max_distance = max(max_distance, dist)
                if dist > 3 and any(e.startswith("subject") for e in field_errors):
                    evidence["long_distance_detected"] = True
                    evidence["distant_relation"] = f"nsubj(predicate, subject)"
                    evidence["distance"] = dist
                    evidence["subject_token"] = {
                        "text": subj.text, "index": subj.index, "lemma": subj.lemma,
                    }
                    evidence["predicate_token"] = {
                        "text": root_token.text, "index": root_token.index,
                        "lemma": root_token.lemma,
                    }
                    return ErrorCategory.LONG_DISTANCE_DEPENDENCY, evidence

            # Check object distance.
            for obj in objects:
                dist = tree.get_dependency_distance(obj.index, root_idx)
                max_distance = max(max_distance, dist)
                if dist > 3 and any(
                    e in field_errors for e in ["action.object", "action.predicate"]
                ):
                    evidence["long_distance_detected"] = True
                    evidence["distant_relation"] = "obj(predicate, object)"
                    evidence["distance"] = dist
                    evidence["object_token"] = {
                        "text": obj.text, "index": obj.index, "lemma": obj.lemma,
                    }
                    evidence["predicate_token"] = {
                        "text": root_token.text, "index": root_token.index,
                        "lemma": root_token.lemma,
                    }
                    return ErrorCategory.LONG_DISTANCE_DEPENDENCY, evidence

    # -------------------------------------------------------------------
    # Rule 5: Negation/Exception Error
    # -------------------------------------------------------------------
    has_neg = tree.has_deprel("neg")

    if has_neg:
        # Negation is present. Check if subject role is incorrect.
        # Negation typically changes obligor → prohibited_party.
        if any(e.startswith("subject.role") for e in field_errors) or \
           any(e.startswith("subject") for e in field_errors):
            neg_tokens = tree.find_tokens_by_deprel("neg")

            evidence["negation_detected"] = True
            evidence["neg_tokens"] = [
                {"text": t.text, "index": t.index, "lemma": t.lemma}
                for t in neg_tokens
            ]
            evidence["pred_role"] = prediction.subject.role.value
            evidence["gold_role"] = gold.subject.role.value

            return ErrorCategory.NEGATION_EXCEPTION, evidence

    # Also check for exception conditions (condition type EXCEPTION that may
    # have confused the role assignment).
    if (
        gold.condition.type.value == "exception"
        and any(e.startswith("subject") for e in field_errors)
    ):
        evidence["exception_condition"] = True
        evidence["gold_condition_type"] = "exception"
        evidence["pred_role"] = prediction.subject.role.value
        evidence["gold_role"] = gold.subject.role.value

        return ErrorCategory.NEGATION_EXCEPTION, evidence

    # -------------------------------------------------------------------
    # Fallback: Other Error
    # -------------------------------------------------------------------
    evidence["no_specific_pattern"] = True
    evidence["field_errors"] = field_errors

    return ErrorCategory.OTHER_ERROR, evidence
