"""
错误案例的语言学解释模板函数。

各函数向 parts 列表追加中英双语解释，
引用导致抽取错误的具体 UD 关系与句法结构。

全部函数签名相同：
    (parts, prediction, gold, tree, evidence) -> None
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

from src.extraction.schema import (
    LegalTriplet, DependencyTree,
)


def add_passive_explanation(
    parts: List[str],
    prediction: LegalTriplet,
    gold: LegalTriplet,
    tree: Optional[DependencyTree],
    evidence: Dict[str, Any],
) -> None:
    """向 parts 列表追加被动语态专用解释。"""
    parts.append(
        "**中文**: 该句为被动语态结构。在被动句中，句法主语(nsubj:pass)是受事者(patient)，"
        "而非施事者(agent)。系统错误地将受事者识别为法律主体，而忽略了由obl:agent标记的"
        "真正施事者。正确的法律主体应是通过'by'引入的施事者(obl:agent)。\n"
    )
    parts.append(
        "**English**: This clause uses passive voice construction. In passive voice, "
        "the syntactic subject (nsubj:pass) is the PATIENT (the entity undergoing the action), "
        "not the AGENT (the entity performing it). The system incorrectly extracted the "
        "patient as the legal subject, failing to identify the true agent marked by obl:agent. "
        "The correct legal subject should be the agent introduced by 'by' (obl:agent "
        "dependency relation).\n"
    )

    if tree is not None:
        nsubj_pass = tree.find_tokens_by_deprel("nsubj:pass")
        obl_agent = tree.find_tokens_by_deprel("obl:agent")
        if nsubj_pass:
            parts.append(
                f"- nsubj:pass (受事/Patient): token {nsubj_pass[0].index} "
                f"`{nsubj_pass[0].text}` (系统误提取/incorrectly extracted by system)"
            )
        if obl_agent:
            parts.append(
                f"- obl:agent (施事/Agent): token {obl_agent[0].index} "
                f"`{obl_agent[0].text}` (应为法律主体/should be the legal subject)"
            )
        parts.append(f"- 系统输出 (System output): `{prediction.subject.text}` "
                     f"(应为/Should be: `{gold.subject.text}`)")


def add_condition_explanation(
    parts: List[str],
    prediction: LegalTriplet,
    gold: LegalTriplet,
    tree: Optional[DependencyTree],
    evidence: Dict[str, Any],
) -> None:
    """向 parts 列表追加条件边界专用解释。"""
    parts.append(
        "**中文**: 系统未能正确识别条件从句的边界。条件从句由advcl依存关系标记，其范围"
        "包括从属连词(mark)引导的整个从句。系统的提取要么遗漏了条件从句的部分内容，"
        "要么将非条件内容错误地纳入了条件范围。\n"
    )
    parts.append(
        "**English**: The system failed to correctly identify the boundary of the "
        "conditional clause. Condition clauses are marked by the advcl dependency "
        "relation, with their scope covering the entire clause introduced by the "
        "subordinating conjunction (mark). The system's extraction either omitted "
        "part of the conditional content or incorrectly included non-conditional "
        "material within the condition span.\n"
    )

    if evidence.get("ud_condition_spans"):
        for span_info in evidence["ud_condition_spans"]:
            parts.append(
                f"- UD条件从句 (UD Condition Span): `{span_info['text']}` "
                f"(deprel: {span_info['deprel']}, tokens: {span_info['token_count']})"
            )
    parts.append(f"- 系统提取 (System extraction): `{prediction.condition.text or '(none)'}`")
    parts.append(f"- 金标条件 (Gold condition): `{gold.condition.text or '(none)'}`")


def add_relcl_explanation(
    parts: List[str],
    prediction: LegalTriplet,
    gold: LegalTriplet,
    tree: Optional[DependencyTree],
    evidence: Dict[str, Any],
) -> None:
    """向 parts 列表追加关系从句专用解释。"""
    parts.append(
        "**中文**: 句中包含关系从句(acl:relcl)，该从句内嵌了另一个谓词-论元结构。"
        "系统可能从关系从句内部提取了谓词或宾语，而非从主句中提取正确的法律行为要素。"
        "关系从句通常修饰名词短语，其内部的谓词不应被视为法律主体动作。\n"
    )
    parts.append(
        "**English**: This clause contains a relative clause (acl:relcl) that embeds "
        "an additional predicate-argument structure. The system likely extracted the "
        "predicate or object from within the relative clause rather than from the "
        "main clause. Relative clauses typically modify noun phrases, and their "
        "internal predicates should not be treated as legal subject actions.\n"
    )

    if evidence.get("relcl_head"):
        parts.append(
            f"- 关系从句中心词 (RelCl head): token {evidence['relcl_head']['index']} "
            f"`{evidence['relcl_head']['text']}`"
        )
    if evidence.get("relcl_subtree_text"):
        parts.append(
            f"- 关系从句内容 (RelCl content): `{evidence['relcl_subtree_text'][:200]}`"
        )


def add_long_distance_explanation(
    parts: List[str],
    prediction: LegalTriplet,
    gold: LegalTriplet,
    tree: Optional[DependencyTree],
    evidence: Dict[str, Any],
) -> None:
    """向 parts 列表追加长距离依存专用解释。"""
    dist = evidence.get("distance", "?")
    parts.append(
        f"**中文**: 谓词与其论元之间的依存距离为 {dist} 个词(token)，超过正常范围(≤3)。"
        "长距离依存关系对LLM提取构成挑战，因为模型在处理线性距离较大的句法关系时，"
        "注意力分布会衰减，导致论元识别不准确。\n"
    )
    parts.append(
        f"**English**: The dependency distance between the predicate and its argument "
        f"is {dist} tokens, exceeding the normal range (≤3). Long-distance dependencies "
        "pose a challenge for LLM extraction because attention distributions decay "
        "over long linear distances, leading to inaccurate argument identification.\n"
    )

    if evidence.get("distant_relation"):
        parts.append(f"- 远距离依存关系 (Distant relation): {evidence['distant_relation']}")
    if evidence.get("predicate_token"):
        parts.append(
            f"- 谓词 (Predicate): token {evidence['predicate_token']['index']} "
            f"`{evidence['predicate_token']['text']}`"
        )
    if evidence.get("subject_token"):
        parts.append(
            f"- 主语 (Subject): token {evidence['subject_token']['index']} "
            f"`{evidence['subject_token']['text']}`"
        )
    if evidence.get("object_token"):
        parts.append(
            f"- 宾语 (Object): token {evidence['object_token']['index']} "
            f"`{evidence['object_token']['text']}`"
        )


def add_negation_explanation(
    parts: List[str],
    prediction: LegalTriplet,
    gold: LegalTriplet,
    tree: Optional[DependencyTree],
    evidence: Dict[str, Any],
) -> None:
    """向 parts 列表追加否定/例外专用解释。"""
    parts.append(
        "**中文**: 句中存在否定标记(neg)或例外条件(exception condition)，这些成分"
        "改变了法律主体的角色分类。否定/例外通常将义务方(obligor)转换为被禁止方"
        "(prohibited_party)，或将权利持有方(right_holder)的角色反转。系统未能"
        "正确解释否定或例外对法律角色的影响。\n"
    )
    parts.append(
        "**English**: The clause contains a negation marker (neg dependency) or "
        "an exception condition that alters the legal role classification of the "
        "subject party. Negation/exceptions typically convert an obligor into a "
        "prohibited party, or invert the role of a right holder. The system failed "
        "to correctly interpret the effect of negation/exception on legal role "
        "assignment.\n"
    )

    if tree is not None:
        neg_tokens = tree.find_tokens_by_deprel("neg")
        if neg_tokens:
            parts.append(
                f"- 否定词 (Negation): token {neg_tokens[0].index} "
                f"`{neg_tokens[0].text}`"
            )
    parts.append(f"- 系统预测角色 (Pred role): `{prediction.subject.role.value}`")
    parts.append(f"- 金标角色 (Gold role): `{gold.subject.role.value}`")


def add_generic_explanation(
    parts: List[str],
    prediction: LegalTriplet,
    gold: LegalTriplet,
    tree: Optional[DependencyTree],
    evidence: Dict[str, Any],
) -> None:
    """向 parts 列表追加通用（其他）错误解释。"""
    parts.append(
        "**中文**: 系统输出与金标存在差异，但未识别到特定的语言学现象。可能是由于"
        "多个因素共同作用导致的提取错误。请人工审查以确定根本原因。\n"
    )
    parts.append(
        "**English**: The system output differs from the gold standard, but no "
        "specific linguistic phenomenon was identified as the primary cause. "
        "The error may result from a combination of multiple factors. "
        "Manual review is recommended to determine the root cause.\n"
    )
