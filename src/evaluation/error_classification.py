"""
语言学错误分类 — 主类别判定。

分析 UD 树以确定导致抽取错误的语言学现象。
实现两级分类体系中按优先级排序的 5 条规则。
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
    long_distance_token_threshold: int,
) -> tuple[ErrorCategory, Dict[str, Any]]:
    """分析 UD 树以确定导致错误的语言学现象。

    按优先级依次检查各现象规则，返回首个匹配项。
    收集 UD 证据供生成语言学解释。

    参数:
        prediction: 系统预测。
        gold: 金标准。
        tree: UD 依存树。
        field_errors: 已检测的字段错误。

    返回:
        (主 ErrorCategory, 解释用 UD 证据字典) 元组。
    """
    evidence: Dict[str, Any] = {}

    # -------------------------------------------------------------------
    # 规则 1：被动语态错误
    # -------------------------------------------------------------------
    # 检测被动：nsubj:pass 标识受事（句法主语），
    # 被动句中为逻辑宾语。LLM 可能抽取受事而非真施事。
    has_nsubj_pass = tree.has_deprel("nsubj:pass")
    has_aux_pass = tree.has_deprel("aux:pass")
    has_obl_agent = tree.has_deprel("obl:agent")

    if has_nsubj_pass and has_aux_pass:
        # 检测到被动。检查错误是否涉及主语。
        if any(e.startswith("subject") for e in field_errors):
            # 收集证据。
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
    # 规则 2：条件边界错误
    # -------------------------------------------------------------------
    has_advcl = tree.has_deprel("advcl")
    has_mark = tree.has_deprel("mark")

    if has_advcl and has_mark:
        # 存在条件从句。检查错误是否涉及条件。
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

            # 计算预测与 UD 条件片段词元以便比较。
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
    # 规则 3：关系从句混淆
    # -------------------------------------------------------------------
    has_relcl = tree.has_deprel("acl:relcl")

    if has_relcl:
        # 检查是否可能由关系从句混淆导致。
        # 确定性检测较难；检查谓词或宾语词元是否出现在关系从句内。
        relcl_tokens = tree.find_tokens_by_deprel("acl:relcl")

        for relcl_token in relcl_tokens:
            relcl_subtree = tree.get_subtree_tokens(relcl_token.index)
            relcl_lemma_set = {t.lemma.lower() for t in relcl_subtree}

            # 检查预测谓词或宾语词形是否出现在关系从句子树中。
            # 若是，LLM 可能从关系从句而非主句抽取。
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
    # 规则 4：长距离依存错误
    # -------------------------------------------------------------------
    # 测量根谓词到主语、宾语的依存距离。
    # 若超过 long_distance_token_threshold 且这些论元上有误，
    # 归类为长距离依存错误。
    root_idx = tree.root_index
    if root_idx is not None:
        root_token = tree.get_token(root_idx)
        if root_token and root_token.upos == "VERB":
            # 查找根节点的主语、宾语依存子节点。
            subjects = tree.get_children(root_idx, deprel="nsubj") + tree.get_children(root_idx, deprel="nsubj:pass")
            objects = tree.get_children(root_idx, deprel="obj")

            max_distance = 0
            # 检查主语距离。
            for subj in subjects:
                dist = tree.get_dependency_distance(subj.index, root_idx)
                max_distance = max(max_distance, dist)
                if dist > long_distance_token_threshold and any(e.startswith("subject") for e in field_errors):
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

            # 检查宾语距离。
            for obj in objects:
                dist = tree.get_dependency_distance(obj.index, root_idx)
                max_distance = max(max_distance, dist)
                if dist > long_distance_token_threshold and any(
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
    # 规则 5：否定/例外错误
    # -------------------------------------------------------------------
    has_neg = tree.has_deprel("neg")

    if has_neg:
        # 存在否定。检查主语角色是否错误。
        # 否定通常将 obligor 变为 prohibited_party。
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

    # 另检查例外条件（EXCEPTION 类型可能混淆角色赋值）。
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
    # 兜底：其他错误
    # -------------------------------------------------------------------
    evidence["no_specific_pattern"] = True
    evidence["field_errors"] = field_errors

    return ErrorCategory.OTHER_ERROR, evidence
