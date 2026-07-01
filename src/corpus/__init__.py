"""
LexSpec 语料库包
================
从 CUAD v1 合同数据构建 LexSpec 评估语料库。

本包负责：
  - 在 UD parse 树上检测语言现象（被动、条件等）
  - 以多种格式加载 CUAD v1（完整合同、专家片段、QA 片段）
  - 通过 Stanza 分句提取条款
  - 测试集选择策略（分层抽样、全量选择）

用法::

    from src.corpus import (
        detect_phenomena, is_boilerplate_clause,
        load_cuad_data, load_cuad_spans, load_cuad_qa_spans,
        split_into_clauses, build_clause_records,
        select_all_clauses, select_balanced_testset,
    )
"""

from src.corpus.phenomena_detector import (
    detect_phenomena,
    is_boilerplate_clause,
)

from src.corpus.cuad_loader import (
    load_cuad_data,
    load_cuad_spans,
    load_cuad_qa_spans,
)

from src.corpus.clause_processor import (
    split_into_clauses,
    build_clause_records,
)

from src.corpus.selection import (
    select_all_clauses,
    select_balanced_testset,
)

__all__ = [
    "detect_phenomena",
    "is_boilerplate_clause",
    "load_cuad_data",
    "load_cuad_spans",
    "load_cuad_qa_spans",
    "split_into_clauses",
    "build_clause_records",
    "select_all_clauses",
    "select_balanced_testset",
]
