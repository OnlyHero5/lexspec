"""
LexSpec 数据模型包
=================

项目中所有模块（抽取、语言学、修正、标注、评估）共享的唯一数据契约。

模型层次:
  LegalTriplet -> {Subject, Action, Condition}          核心抽取输出
  DependencyTree -> {Token}                              UD 依存树封装
  ValidationResult -> {status, evidence, corrections}   校验器输出
  ErrorCase -> {error_id, categories, explanation}      错误分析
  AnnotationDisagreement -> {qwen, gemma, resolution}   标注分歧

设计原则:
  - 所有模型使用 Pydantic v2 语法（model_validate、model_dump 等）
  - 每个字段附带 description 实现自文档化序列化
  - 枚举类型使用字符串基类，确保 JSON 序列化兼容
  - DependencyTree 提供辅助方法遍历 UD 依存树，将所有语言学模块
    与原始 Stanza 对象隔离开来
"""

from .enums import (
    LegalRole,
    ConditionType,
    ValidationStatus,
    ErrorCategory,
    FieldErrorType,
)
from .triplet import (
    Subject,
    Action,
    Condition,
    LegalTriplet,
)
from .dependency import (
    Token,
    ClauseSpan,
    ConditionSpan,
)
from ._dependency_tree import DependencyTree
from .validation import (
    LinguisticEvidence,
    FieldCorrection,
    ValidationResult,
)
from .error import (
    ErrorCase,
    AnnotationDisagreement,
)

__all__ = [
    # 枚举
    "LegalRole",
    "ConditionType",
    "ValidationStatus",
    "ErrorCategory",
    "FieldErrorType",
    # 核心三元组
    "Subject",
    "Action",
    "Condition",
    "LegalTriplet",
    # 依存
    "Token",
    "ClauseSpan",
    "ConditionSpan",
    "DependencyTree",
    # 校验
    "LinguisticEvidence",
    "FieldCorrection",
    "ValidationResult",
    # 错误 / 标注
    "ErrorCase",
    "AnnotationDisagreement",
]
