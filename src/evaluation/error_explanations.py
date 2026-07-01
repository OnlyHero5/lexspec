"""
Linguistic explanation dispatcher for error cases.

Extracted from error_analyzer.py to reduce module size. The dispatcher
generate_explanation() routes to template functions based on the primary
error category. Template functions are defined in explanation_templates.py.
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
    """Generate a bilingual (Chinese + English) linguistic explanation.

    The explanation cites specific UD relations, token indices, and
    syntactic structures that caused the extraction error. The bilingual
    format serves both Chinese-speaking researchers and international
    (English-language) publication venues.

    Format:
        [中文标题] (English Title)
        中文详细解释，引用具体的UD依存关系和句法结构。
        English detailed explanation, citing specific UD relations
        and syntactic structures.

    Args:
        prediction: System prediction.
        gold: Gold standard.
        tree: UD tree (may be None).
        primary: Primary error category.
        secondary: Secondary error category.
        field_errors: List of error field paths.
        ud_evidence: Dict of UD evidence collected during classification.
        validation_result: Optional validation result for additional context.

    Returns:
        Bilingual natural-language explanation string.
    """
    parts: List[str] = []

    # ---- Header with error type labels ----
    # Primary category labels in bilingual format.
    primary_labels = {
        ErrorCategory.PASSIVE_VOICE: "被动语态错误 (Passive Voice Error)",
        ErrorCategory.CONDITIONAL_BOUNDARY: "条件边界错误 (Conditional Boundary Error)",
        ErrorCategory.RELATIVE_CLAUSE: "关系从句混淆 (Relative Clause Confusion)",
        ErrorCategory.LONG_DISTANCE_DEPENDENCY: "长距离依存错误 (Long-distance Dependency Error)",
        ErrorCategory.NEGATION_EXCEPTION: "否定/例外错误 (Negation/Exception Error)",
        ErrorCategory.OTHER_ERROR: "其他错误 (Other Error)",
    }

    # Secondary category labels.
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

    # ---- Prediction vs Gold comparison ----
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

    # ---- Linguistic analysis based on primary category ----
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

    # ---- UD Evidence section (if available) ----
    if tree is not None and tree.token_count > 0 and ud_evidence:
        parts.append("\n### UD依存证据 (UD Dependency Evidence)\n")
        if tree.root_index is not None:
            root = tree.get_token(tree.root_index)
            if root:
                parts.append(f"- 根谓词 (Root): token {root.index} `{root.text}` "
                             f"(lemma: {root.lemma}, deprel: {root.deprel})")
        parts.append(f"- 句子 (Sentence): `{tree.text[:200]}`")

        # List key UD relations found.
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

    # ---- Validation context (if available) ----
    if validation_result is not None and validation_result.corrections:
        parts.append("\n### 验证修正记录 (Validation Corrections)\n")
        for corr in validation_result.corrections:
            parts.append(f"- 字段 (Field) `{corr.field}`: "
                         f"`{corr.original}` → `{corr.corrected}`")
            if corr.reason:
                parts.append(f"  理由 (Reason): {corr.reason}")

    return "\n".join(parts)
