"""
CUAD v1 数据加载工具。

支持三种数据源格式：
  - CUAD_v1.json 中的完整合同上下文（句子模式）
  - master_clauses.csv 中的专家标注条款片段（片段模式）
  - CUAD_v1.json 中的 QA 答案片段（qa_spans 模式）
"""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import List, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

# 加载 CUAD 片段时跳过的列。
_SKIP_COLUMNS = frozenset({
    "Filename", "Document Name", "Document Name-Answer",
    "Parties", "Parties-Answer",
})


def _normalize_cuad_span_text(raw: str) -> Optional[str]:
    """将 master_clauses.csv 单元格规范为单条条款文本。

    CUAD CSV 中条款常以 Python 列表字符串存储，例如
    ``\"['May 8, 2014', '8th day of May 2014']\"``。
    本函数解析列表并取最长片段作为代表性条款文本。
    """
    text = (raw or "").strip()
    if len(text) < 20:
        return None

    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list):
            parts = [str(item).strip() for item in parsed if str(item).strip()]
            if not parts:
                return None
            text = max(parts, key=len)
        elif parsed is not None:
            text = str(parsed).strip()
        else:
            # 列表字面量解析失败时，尽量去掉外层括号。
            text = text.strip("[]").strip().strip("'\"")

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 20:
        return None
    return text


def load_cuad_data(cuad_path: str) -> List[str]:
    """加载 CUAD v1 JSON 并提取全部段落上下文。

    CUAD v1 JSON 结构::

        {
          "version": "aok_v1.0",
          "data": [
            {"title": "...", "paragraphs": [
              {"context": "<完整合同文本>", "qas": [...]},
              ...
            ]},
            ...
          ]
        }

    提取每个段落的 ``context`` 字段。各上下文可能
    含整份合同；分句由下游 Stanza 分句器完成。

    参数：
        cuad_path: CUAD_v1.json 文件路径。

    返回：
        段落上下文字符串列表（每段一条）。
    """
    logger.info("Loading CUAD data: %s", cuad_path)
    with open(cuad_path, "r", encoding="utf-8") as fh:
        cuad = json.load(fh)

    contexts: List[str] = []
    for doc in cuad.get("data", []):
        for para in doc.get("paragraphs", []):
            ctx = para.get("context", "")
            if ctx and ctx.strip():
                contexts.append(ctx.strip())

    logger.info(
        "Loaded %d paragraph contexts from %d CUAD documents",
        len(contexts), len(cuad.get("data", [])),
    )
    return contexts


def load_cuad_spans(master_clauses_path: str) -> List[str]:
    """从 CUAD master_clauses.csv 加载专家标注条款片段。

    每个非 Answer 列可能含 41 类之一的一条条款文本。
    返回去重后的条款文本列表（通常约 5k 个唯一片段）。
    """
    path = Path(master_clauses_path)
    if not path.exists():
        raise FileNotFoundError(f"master_clauses.csv not found: {master_clauses_path}")

    seen: set = set()
    clauses: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for col, val in row.items():
                if col in _SKIP_COLUMNS or col.endswith("-Answer"):
                    continue
                text = _normalize_cuad_span_text(val or "")
                if text is None:
                    continue
                if text in seen:
                    continue
                seen.add(text)
                clauses.append(text)
    logger.info("Loaded %d unique clause fragments from %s", len(clauses), path)
    return clauses


def load_cuad_qa_spans(cuad_path: str) -> List[str]:
    """从 CUAD SQuAD 格式 JSON 加载答案片段（约 13k 标注片段）。"""
    with open(cuad_path, "r", encoding="utf-8") as fh:
        cuad = json.load(fh)

    seen: set = set()
    clauses: List[str] = []
    for doc in cuad.get("data", []):
        for para in doc.get("paragraphs", []):
            for qa in para.get("qas", []):
                if qa.get("is_impossible"):
                    continue
                for ans in qa.get("answers", []):
                    text = (ans.get("text") or "").strip()
                    if len(text) < 20:
                        continue
                    norm = re.sub(r"\s+", " ", text)
                    if norm in seen:
                        continue
                    seen.add(norm)
                    clauses.append(text)
    logger.info("Loaded %d unique QA answer spans from %s", len(clauses), cuad_path)
    return clauses
