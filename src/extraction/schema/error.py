"""
LexSpec 数据模型 —— 错误分析与标注模型
=======================================

用于评估诊断与多标注者金标准构建的 ErrorCase 与 AnnotationDisagreement。
"""

from typing import List, Dict, Any

from pydantic import BaseModel, Field

from .enums import ErrorCategory, FieldErrorType
from .triplet import LegalTriplet


class ErrorCase(BaseModel):
    """单条预测与金标不一致时的错误分析记录。

    字段:
        error_id: 唯一标识（如 ``E-001``），便于报告追溯。
        text: 被分析的原始条款文本摘要。
        prediction: 系统预测三元组（可能经 UD 修正）。
        gold: 金标准三元组。
        primary_category: 主类 —— 导致错误的语言学现象（被动、条件边界等）。
        secondary_category: 次类 —— 受影响的字段类型（主语、谓词、条件等）。
        linguistic_explanation: 引用 UD 关系的双语/详细语言学解释。
    """
    error_id: str = Field(
        description="唯一错误标识，如 'E-001'，便于在报告中追溯"
    )
    text: str = Field(
        description="被分析的原始条款文本"
    )
    prediction: LegalTriplet = Field(
        description="系统预测的三元组（修正后或原始）"
    )
    gold: LegalTriplet = Field(
        description="金标准三元组（人工标注或裁决后）"
    )
    primary_category: ErrorCategory = Field(
        description="主要错误类别——导致错误的语言学现象"
    )
    secondary_category: FieldErrorType = Field(
        description="次要错误类别——哪些字段不正确"
    )
    linguistic_explanation: str = Field(
        description=(
            "错误的详细自然语言解释，引用具体 UD 依存关系"
            "（如 'nsubj:pass at token 3'、'advcl from token 2 to 8'），"
            "并说明句法结构为何导致抽取失败。"
        )
    )


class AnnotationDisagreement(BaseModel):
    """
    多标注者金标准构建流水线中两个标注模型（Qwen 与 Gemma）分歧的记录。

    分歧通过以下方式解决：
      1. 基于规则的自动裁决（UD 证据明确的案例）
      2. 人工审核（模糊或复杂案例）

    解决过程记录在 ``disagreement_fields`` 中，
    最终裁决结果保存在 ``final_gold`` 中。
    """
    clause_id: str = Field(
        description="唯一条款标识，如 'C-0042'，链接到源文档"
    )
    text: str = Field(
        description="被标注的完整条款文本"
    )
    qwen_annotation: LegalTriplet = Field(
        description="Qwen 模型产生的标注"
    )
    gemma_annotation: LegalTriplet = Field(
        description="Gemma 模型产生的标注"
    )
    disagreement_fields: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "分歧记录列表，每条为含以下键的字典："
            "field (str)、qwen_value (str)、gemma_value (str)、"
            "resolution (str)、resolved_by (str —— 'auto' 或 'human')"
        )
    )
    final_gold: LegalTriplet = Field(
        description="分歧解决后的最终金标准三元组"
    )
