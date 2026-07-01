"""
错误案例的语言学解释分发器。

自 error_analyzer.py 拆出以减小模块体积。分发器 generate_explanation()
按主错误类别路由到模板函数。模板函数定义于 explanation_templates.py。
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any

from src.extraction.schema import (
    LegalTriplet, DependencyTree, ErrorCategory, FieldErrorType, ValidationResult,
)
from src.evaluation.explanation_templates import (
    add_passive_explanation,
    add_condition_explanation,
    add_relcl_explanation,
    add_long_distance_explanation,
    add_negation_explanation,
    add_generic_explanation,
)


def generate_explanation(
    prediction: LegalTriplet,
    gold: LegalTriplet,
    tree: Optional[DependencyTree],
    primary: ErrorCategory,
    secondary: FieldErrorType,
    field_errors: List[str],
    ud_evidence: Dict[str, Any],
    validation_result: Optional[ValidationResult],
) -> str:
    """生成中英双语语言学解释。

    解释引用具体 UD 关系、词元索引及导致抽取错误的句法结构。
    双语格式兼顾中文研究者与国际（英文）发表场合。

    格式:
        [中文标题] (英文标题)
        中文详细解释，引用具体 UD 依存关系与句法结构。
        英文详细解释，引用具体 UD 关系与句法结构。

    参数:
        prediction: 系统预测。
        gold: 金标准。
        tree: UD 树（可为 None）。
        primary: 主错误类别。
        secondary: 次错误类别。
        field_errors: 错误字段路径列表。
        ud_evidence: 分类阶段收集的 UD 证据字典。
        validation_result: 可选验证结果，供补充上下文。

    返回:
        双语自然语言解释字符串。
    """
    parts: List[str] = []

    # ---- 含错误类型标签的标题 ----
    # 主类别双语标签。
    primary_labels = {
        ErrorCategory.PASSIVE_VOICE: "被动语态错误 (Passive Voice Error)",
        ErrorCategory.CONDITIONAL_BOUNDARY: "条件边界错误 (Conditional Boundary Error)",
        ErrorCategory.RELATIVE_CLAUSE: "关系从句混淆 (Relative Clause Confusion)",
        ErrorCategory.LONG_DISTANCE_DEPENDENCY: "长距离依存错误 (Long-distance Dependency Error)",
        ErrorCategory.NEGATION_EXCEPTION: "否定/例外错误 (Negation/Exception Error)",
        ErrorCategory.OTHER_ERROR: "其他错误 (Other Error)",
    }

    # 次类别标签。
    secondary_labels = {
        FieldErrorType.SUBJECT: "主语错误 (Subject Error)",
        FieldErrorType.ROLE: "角色错误 (Role Error)",
        FieldErrorType.PREDICATE: "谓词错误 (Predicate Error)",
        FieldErrorType.OBJECT: "宾语错误 (Object Error)",
        FieldErrorType.CONDITION_OMISSION: "条件遗漏 (Condition Omission)",
        FieldErrorType.CONDITION_OVEREXTENSION: "条件过度扩展 (Condition Over-extension)",
    }

    header_cn = primary_labels.get(primary, "未知错误")
    header_en = secondary_labels.get(secondary, "Unknown Error")
    parts.append(f"## {header_cn} — {header_en}\n")

    # ---- 预测 vs 金标对比 ----
    parts.append("### 预测 vs 金标 (Prediction vs Gold)\n")
    parts.append(f"- 预测主语 (Pred Subject): `{prediction.subject.text}` [{prediction.subject.role.value}]")
    parts.append(f"- 金标主语 (Gold Subject):  `{gold.subject.text}` [{gold.subject.role.value}]")
    parts.append(f"- 预测谓词 (Pred Predicate): `{prediction.action.predicate}`")
    parts.append(f"- 金标谓词 (Gold Predicate):  `{gold.action.predicate}`")
    parts.append(f"- 预测宾语 (Pred Object): `{prediction.action.object}`")
    parts.append(f"- 金标宾语 (Gold Object):  `{gold.action.object}`")
    parts.append(f"- 预测条件 (Pred Condition): `{prediction.condition.text or '(none)'}`")
    parts.append(f"- 金标条件 (Gold Condition):  `{gold.condition.text or '(none)'}`")
    parts.append(f"- 错误字段 (Error Fields): {', '.join(field_errors)}\n")

    # ---- 按主类别的语言学分析 ----
    parts.append("### 语言学分析 (Linguistic Analysis)\n")

    if primary == ErrorCategory.PASSIVE_VOICE:
        add_passive_explanation(parts, prediction, gold, tree, ud_evidence)

    elif primary == ErrorCategory.CONDITIONAL_BOUNDARY:
        add_condition_explanation(parts, prediction, gold, tree, ud_evidence)

    elif primary == ErrorCategory.RELATIVE_CLAUSE:
        add_relcl_explanation(parts, prediction, gold, tree, ud_evidence)

    elif primary == ErrorCategory.LONG_DISTANCE_DEPENDENCY:
        add_long_distance_explanation(parts, prediction, gold, tree, ud_evidence)

    elif primary == ErrorCategory.NEGATION_EXCEPTION:
        add_negation_explanation(parts, prediction, gold, tree, ud_evidence)

    else:
        add_generic_explanation(parts, prediction, gold, tree, ud_evidence)

    # ---- UD 证据节（若可用） ----
    if tree is not None and tree.token_count > 0 and ud_evidence:
        parts.append("\n### UD依存证据 (UD Dependency Evidence)\n")
        if tree.root_index is not None:
            root = tree.get_token(tree.root_index)
            if root:
                parts.append(f"- 根谓词 (Root): token {root.index} `{root.text}` "
                             f"(lemma: {root.lemma}, deprel: {root.deprel})")
        parts.append(f"- 句子 (Sentence): `{tree.text[:200]}`")

        # 列出关键 UD 关系。
        rels_found = []
        for deprel in ["nsubj", "nsubj:pass", "obj", "obl:agent", "advcl", "mark",
                        "acl:relcl", "neg", "aux", "aux:pass"]:
            if tree.has_deprel(deprel):
                tokens = tree.find_tokens_by_deprel(deprel)
                for t in tokens:
                    rels_found.append(f"  - {deprel}: token {t.index} `{t.text}` "
                                      f"(lemma: {t.lemma}, head: {t.head})")
        if rels_found:
            parts.append("- 相关依存关系 (Relevant dependency relations):")
            parts.extend(rels_found)

    # ---- 验证上下文（若可用） ----
    if validation_result is not None and validation_result.corrections:
        parts.append("\n### 验证修正记录 (Validation Corrections)\n")
        for corr in validation_result.corrections:
            parts.append(f"- 字段 (Field) `{corr.field}`: "
                         f"`{corr.original}` → `{corr.corrected}`")
            if corr.reason:
                parts.append(f"  理由 (Reason): {corr.reason}")

    return "\n".join(parts)
