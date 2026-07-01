"""
七步校验算法 —— 独立实现
========================================================
从 ConstraintValidator.validate() 提取以保持文件体积可控。

本模块包含七步算法的核心编排逻辑。
ConstraintValidator 类委托给此函数，类本身专注于
组件生命周期与配置。
"""

from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING

from src.extraction.schema import (
    DependencyTree,
    Token,
    LegalTriplet,
    ValidationStatus,
    ValidationResult,
    LinguisticEvidence,
    FieldCorrection,
    LegalRole,
    ConditionSpan,
)
from src.utils.logging import get_logger

from src.linguistic.validator._steps_1_2 import step1_find_predicate, step2_detect_voice
from src.linguistic.validator._steps_3_4 import step3_validate_subject, step4_validate_object
from src.linguistic.validator._step_5 import step5_validate_condition
from src.linguistic.validator._step_6 import step6_validate_role
from src.linguistic.validator._step_7 import step7_determine_status
from src.linguistic.validator._results import (
    build_linguistic_evidence,
    build_feedback,
    apply_corrections,
)

if TYPE_CHECKING:
    from src.linguistic.validator.validator import ConstraintValidator

logger = get_logger(__name__)


def run_validation(
    validator: "ConstraintValidator",
    triplet: LegalTriplet,
    text: str,
    tree: Optional[DependencyTree] = None,
) -> ValidationResult:
    """运行完整的七步约束校验算法。

    所有校验的主入口。编排七步流水线并产出含状态、
    证据、修正与反馈的 ValidationResult。

    若 `tree` 为 None，通过 StanzaParser 内部解析文本。
    批处理时预先解析并传入 tree，以避免重复 Stanza 调用。

    参数：
        validator: 提供解析器访问与组件依赖的 ConstraintValidator 实例。
        triplet: 待校验的大语言模型抽取法律三元组。
        text: 原始合同从句文本（发送给大语言模型的精确输入）。
        tree: 预解析依存树。为 None 时从 text 解析。

    返回：
        含完整校验详情的 ValidationResult。

    抛出：
        ValueError: 解析失败（空文本、未产生句子）时。
        RuntimeError: Stanza 未初始化且 parser 为 None 时。
    """
    # 确保有解析树。
    if tree is None:
        logger.debug("No pre-parsed tree provided — parsing text")
        tree = validator.parser.parse(text)

    if tree.token_count == 0:
        raise ValueError(
            "Cannot validate: dependency tree has zero tokens. "
            "The input text may be empty or unparseable."
        )

    # 七步算法的累加器。
    corrections: List[FieldCorrection] = []
    feedback_parts: List[str] = []
    ud_subject: Optional[Token] = None
    ud_object: Optional[Token] = None
    condition_span: Optional[ConditionSpan] = None
    is_passive_detected: bool = False
    modality_aux: str = ""
    polarity: str = "positive"
    legal_role: LegalRole = LegalRole.OTHER
    predicate_token: Optional[Token] = None

    logger.info(
        "Validating triplet for sentence (%d tokens): '%s'",
        tree.token_count,
        text[:100] + "..." if len(text) > 100 else text,
    )

    # ---- 步骤 1：定位根谓词 ----
    predicate_token = step1_find_predicate(tree)
    if predicate_token is None:
        # 无根谓词则无法校验。
        # 句子可能残缺或解析失败。
        logger.warning("No root predicate found in tree — cannot validate")
        return ValidationResult(
            status=ValidationStatus.REFLEXION_REQUIRED,
            original_prediction=triplet,
            corrected_prediction=None,
            linguistic_evidence=LinguisticEvidence(),
            corrections=[],
            feedback=(
                "Could not locate the main clause predicate in the dependency "
                "parse. The sentence may be fragmentary, ungrammatical, or "
                "the parser may have produced a malformed tree. Please verify "
                "the input text and re-parse."
            ),
        )

    predicate_idx = predicate_token.index
    predicate_lemma = predicate_token.lemma
    logger.info(
        "Step 1: Root predicate='%s' (lemma='%s') at index %d",
        predicate_token.text, predicate_lemma, predicate_idx,
    )

    # ---- 步骤 2：检测语态；恢复语义论元 ----
    is_passive_detected, raw_ud_subject, raw_ud_object = step2_detect_voice(
        tree, predicate_idx
    )

    # 被动语态：raw_ud_subject = obl:agent（语义施事），
    # raw_ud_object = nsubj:pass（语义受事）。
    # 主动语态：raw_ud_subject = nsubj，raw_ud_object = obj。
    ud_subject = raw_ud_subject
    ud_object = raw_ud_object

    logger.info(
        "Step 2: Passive=%s, UD subject='%s', UD object='%s'",
        is_passive_detected,
        ud_subject.text if ud_subject else "None",
        ud_object.text if ud_object else "None",
    )

    # ---- 步骤 3：校验主语 ----
    subject_valid = step3_validate_subject(
        triplet.subject.text, ud_subject, corrections, feedback_parts
    )
    if subject_valid:
        logger.info("Step 3: Subject VALID (%s)", triplet.subject.text)
    else:
        logger.info("Step 3: Subject needs correction")

    # ---- 步骤 4：校验宾语 ----
    object_valid = step4_validate_object(
        triplet.action.object, ud_object, corrections, feedback_parts
    )
    if object_valid:
        logger.info("Step 4: Object VALID (%s)", triplet.action.object)
    else:
        logger.info("Step 4: Object needs correction")

    # ---- 步骤 5：校验条件 ----
    condition_span = step5_validate_condition(
        triplet, tree, predicate_idx,
        validator.condition_extractor, validator._condition_overlap,
        corrections, feedback_parts,
    )
    if condition_span is not None:
        logger.info(
            "Step 5: UD condition='%s' (type=%s)",
            condition_span.text[:80],
            condition_span.condition_type.value,
        )
    else:
        logger.info("Step 5: No UD condition detected")

    # ---- 步骤 6：校验情态/角色 ----
    role_valid = step6_validate_role(
        triplet.subject.role, tree, predicate_idx,
        predicate_lemma,
        validator.polarity_detector, corrections, feedback_parts,
    )

    # 提取情态供证据使用（即使角色有效）。
    modality_aux, polarity = validator.polarity_detector.detect_modality(
        tree, predicate_idx
    )
    legal_role, _ = validator.polarity_detector.detect(
        tree, predicate_idx, predicate_lemma
    )

    if role_valid:
        logger.info("Step 6: Role VALID (%s)", triplet.subject.role.value)
    else:
        logger.info("Step 6: Role needs correction")

    # ---- 步骤 7：确定输出状态 ----
    status = step7_determine_status(corrections)
    logger.info("Step 7: Final status = %s", status.value)

    # ---- 构建 ValidationResult ----
    evidence = build_linguistic_evidence(
        tree=tree,
        predicate_idx=predicate_idx,
        ud_subject=ud_subject,
        ud_object=ud_object,
        condition_span=condition_span,
        is_passive_detected=is_passive_detected,
        modality_aux=modality_aux,
        polarity=polarity,
        legal_role=legal_role,
    )

    feedback = build_feedback(feedback_parts)

    corrected_triplet = None
    if status == ValidationStatus.CORRECTED and corrections:
        corrected_triplet = apply_corrections(triplet, corrections)

    return ValidationResult(
        status=status,
        original_prediction=triplet,
        corrected_prediction=corrected_triplet,
        linguistic_evidence=evidence,
        corrections=corrections,
        feedback=feedback,
    )
