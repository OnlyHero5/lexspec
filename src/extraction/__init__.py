"""
LexSpec 抽取包
============================

本包负责使用基于大语言模型的抽取方法，从合同条款中抽取法律三元组，
并通过通用依存句法约束进行事后校验。

公开 API:
  - 大语言模型客户端:  ClientConfig、LLMClient —— 兼容 OpenAI 的 llama.cpp
                      远程推理封装，支持重试与结构化输出。
  - 抽取器:            LegalTripletExtractor —— 调用大语言模型并解析结构化
                      LegalTriplet 响应，采用多策略 JSON 解析。
  - 模式模型:          Subject、Action、Condition、LegalTriplet、DependencyTree、
                      ValidationResult、ErrorCase、AnnotationDisagreement
  - 枚举:              LegalRole、ConditionType、ValidationStatus、ErrorCategory、
                      FieldErrorType
"""

# --- 模式模型（核心数据契约）---
from src.extraction.schema import (
    # 核心三元组模型
    LegalTriplet,
    Subject,
    Action,
    Condition,
    # 枚举
    LegalRole,
    ConditionType,
    ValidationStatus,
    ErrorCategory,
    FieldErrorType,
    # 语言学模型
    Token,
    ClauseSpan,
    ConditionSpan,
    DependencyTree,
    # 校验模型
    LinguisticEvidence,
    FieldCorrection,
    ValidationResult,
    # 标注 / 错误分析
    ErrorCase,
    AnnotationDisagreement,
)

# --- 大语言模型客户端 ---
from src.extraction.client import (
    ClientConfig,
    LLMClient,
)

# --- 抽取器 ---
from src.extraction.extractor import (
    LegalTripletExtractor,
)

__all__ = [
    # 核心三元组
    "LegalTriplet",
    "Subject",
    "Action",
    "Condition",
    # 枚举
    "LegalRole",
    "ConditionType",
    "ValidationStatus",
    "ErrorCategory",
    "FieldErrorType",
    # 语言学
    "Token",
    "ClauseSpan",
    "ConditionSpan",
    "DependencyTree",
    # 校验
    "LinguisticEvidence",
    "FieldCorrection",
    "ValidationResult",
    # 标注 / 错误
    "ErrorCase",
    "AnnotationDisagreement",
    # 客户端
    "ClientConfig",
    "LLMClient",
    # 抽取器
    "LegalTripletExtractor",
]
