"""
LexSpec 标注包
==============
用于构建金标准测试集的双模型标注流水线。

本包实现仅在第一阶段（LexSpec-100 测试集构建）中使用的多标注者工作流。
两个标注模型（Qwen3.6 27B、Gemma4 31B）独立标注每条合同条款，
输出通过字段级投票共识进行调和。

重要：这些模型与第二、三阶段使用的实验模型（Qwen3.5 9B）完全隔离。
标注模型的预测不会泄漏到训练、提示或评估中。

包结构：
  - llm_annotator.py:       单模型标注的 LLMAnnotator 类
  - consensus.py:           字段级投票、分歧解决、
                            金标准构建
  - statistics.py:          标注者间一致性统计计算
  - normalization.py:       模糊比较用的文本规范化
  - field_helpers.py:       字段提取、解析与比较辅助函数
  - triplet_coercion.py:    JSON 转 LegalTriplet 的强制转换工具
  - prompts.py:             从 YAML 加载提示词（无回退）
  - response_parser.py:     标注用的 LLM 响应解析
  - disagreement_logger.py: AnnotationDisagreement 记录
  - disagreement_io.py:     分歧记录持久化（JSONL 读写）
  - reviewer.py:            跨模型标注审查器

公开 API：
  - LLMAnnotator:              使用单个 LLM 标注条款
  - CrossModelReviewer:        让一个 LLM 审查另一模型的标注
  - field_level_consensus:     逐字段比较两份标注
  - resolve_disagreement:      将人工裁决应用于分歧
  - build_gold_from_consensus: 从共识数据构建最终金标准三元组
  - generate_annotation_stats: 计算标注者间一致性统计
  - normalize_text:            模糊比较用的文本规范化
  - coerce_to_triplet:         将原始 JSON 强制转换为 LegalTriplet
  - infer_condition_type:      从文本推断条件类型（规范版本）
  - log_disagreement:          创建 AnnotationDisagreement 记录
  - save_disagreement_log:     将分歧记录持久化到 JSONL
"""

from src.annotation.llm_annotator import LLMAnnotator
from src.annotation.reviewer import CrossModelReviewer
from src.annotation.consensus import (
    field_level_consensus,
    resolve_disagreement,
    build_gold_from_consensus,
)
from src.annotation.statistics import generate_annotation_stats
from src.annotation.normalization import normalize_text
from src.annotation.triplet_coercion import coerce_to_triplet, infer_condition_type
from src.annotation.disagreement_logger import log_disagreement
from src.annotation.disagreement_io import save_disagreement_log, append_disagreement_log

__all__ = [
    "CrossModelReviewer",
    "LLMAnnotator",
    "field_level_consensus",
    "resolve_disagreement",
    "build_gold_from_consensus",
    "generate_annotation_stats",
    "normalize_text",
    "coerce_to_triplet",
    "infer_condition_type",
    "log_disagreement",
    "save_disagreement_log",
    "append_disagreement_log",
]
