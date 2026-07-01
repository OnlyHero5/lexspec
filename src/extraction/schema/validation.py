"""
LexSpec 数据模型 —— 校验模型
===========================

UD 约束校验器的输出模型：LinguisticEvidence, FieldCorrection, ValidationResult。
"""

from typing import Optional, List

from pydantic import BaseModel, Field

from .enums import ValidationStatus, ConditionType, LegalRole
from .triplet import LegalTriplet


class LinguisticEvidence(BaseModel):
    """从单条条款 UD 依存解析中提取的语言学证据（校验参照）。

    字段在校验过程中逐步填充，主要字段含义：
        predicate / predicate_index: 根谓词词元及其 1 基索引。
        ud_subject: UD 识别的主语（主动 ``nsubj``，被动 ``obl:agent``）。
        ud_object: UD 识别的宾语（主动 ``obj``，被动 ``nsubj:pass``）。
        condition_span / condition_type: 条件从句文本与语义类型。
        passive_detected: 是否检测到被动构造（``nsubj:pass`` / ``aux:pass``）。
        modality_aux: 情态助动词词元（``shall`` / ``may`` 等）。
        polarity: 子句极性（``positive`` / ``negative``）。
        legal_role: 由情态+语态推导的法律角色。
        max_argument_distance: 谓词到论元的最大依存距离（长距离分析用）。
    """
    predicate: str = Field(
        default="",
        description="识别的根谓词词元形式，如 'deliver'、'indemnify'"
    )
    predicate_index: int = Field(
        default=0,
        description="根谓词的 1 基词元索引；未找到时为 0"
    )
    ud_subject: str = Field(
        default="",
        description="通过 UD 识别的主语——主动时为 nsubj，被动时为 obl:agent"
    )
    ud_object: str = Field(
        default="",
        description="通过 UD 识别的宾语——主动时为 obj，被动时为 nsubj:pass"
    )
    condition_span: str = Field(
        default="",
        description="检测到的条件子句跨度的完整文本（若有）"
    )
    condition_type: ConditionType = Field(
        default=ConditionType.NONE,
        description="检测到的条件子句的语义类型"
    )
    passive_detected: bool = Field(
        default=False,
        description="若检测到被动语态（存在 nsubj:pass 或 aux:pass）则为 True"
    )
    modality_aux: str = Field(
        default="",
        description="情态助动词词元，如 'shall'、'may'、'must'；无情态时为空"
    )
    polarity: str = Field(
        default="positive",
        description="子句极性：'positive' 或 'negative'（基于否定词）"
    )
    legal_role: LegalRole = Field(
        default=LegalRole.OTHER,
        description="根据情态 + 语态 + 论元结构推导的法律角色"
    )
    max_argument_distance: int = Field(
        default=0,
        description=(
            "从根谓词到任意 nsubj/nsubj:pass/obj 论元的最大词元级依存距离；"
            "用于长距离错误提示"
        ),
    )


class FieldCorrection(BaseModel):
    """校验器对 LLM 预测应用的单字段修正记录。

    字段:
        field: 点号路径，如 ``subject.text``、``action.predicate``。
        original: LLM 原始输出值。
        corrected: 由 UD 句法分析推导的修正值。
        reason: 引用 UD 关系与自然语言说明的修正理由。
    """
    field: str = Field(
        description="点号表示的字段路径，如 'subject.text'、'condition.type'、'action.predicate'"
    )
    original: str = Field(
        default="",
        description="大语言模型产生的原始值"
    )
    corrected: str = Field(
        default="",
        description="从 UD 句法分析推导的修正值"
    )
    reason: str = Field(
        default="",
        description="引用 UD 关系与句法证据的自然语言理由"
    )


class ValidationResult(BaseModel):
    """UD 约束校验器对单条条款的完整输出。

    字段:
        status: 校验结论 — ``VALID``（一致）、``CORRECTED``（已自动修正）、
            ``REFLEXION_REQUIRED``（需 LLM 重生成）。
        original_prediction: LLM 原始三元组，保留供错误分析。
        corrected_prediction: 自动修正后的三元组；仅 ``CORRECTED`` 时有值。
        linguistic_evidence: 用于比对的 UD 句法证据。
        corrections: 已应用的字段修正列表；``VALID`` 时为空。
        feedback: 人类可读的错误说明；供 Reflexion 与日志使用。
    """
    status: ValidationStatus = Field(
        description="校验结果：VALID、CORRECTED 或 REFLEXION_REQUIRED"
    )
    original_prediction: LegalTriplet = Field(
        description="大语言模型原始预测，保留供错误分析"
    )
    corrected_prediction: Optional[LegalTriplet] = Field(
        default=None,
        description="修正后的三元组；status 为 CORRECTED 时设置，REFLEXION_REQUIRED 时为 None"
    )
    linguistic_evidence: LinguisticEvidence = Field(
        description="用于校验的 UD 推导句法证据"
    )
    corrections: List[FieldCorrection] = Field(
        default_factory=list,
        description="已应用的字段修正列表；VALID 时为空"
    )
    feedback: str = Field(
        default="",
        description=(
            "说明出错原因的可读反馈。"
            "用于 Reflexion 提示词与错误日志。status 为 VALID 时为空。"
        )
    )
