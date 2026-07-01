"""
基于 UD 依存 parse 树的语言现象检测。

各检测规则基于 UD v2 依存关系，见
LexSpec 设计文档 4.2–4.6 节。
"""

from __future__ import annotations

import re
from typing import Dict

from src.linguistic.stanza_parser import StanzaParser
from src.linguistic.ud_features import (
    UDFeatureExtractor,
    compute_mean_dependency_distance,
)
from src.linguistic.passive_detector import PassiveDetector
from src.extraction.schema import DependencyTree

# ---------------------------------------------------------------------------
# 非实质性文本过滤
# ---------------------------------------------------------------------------

# 分句结果中需排除的模板文本
# （页眉、签名、目录等）。
_BOILERPLATE_RE = re.compile(
    r"^(page\s+\d+|exhibit\s+[a-z0-9]+|schedule\s+[a-z0-9]+|"
    r"table of contents|signature page|in witness whereof|"
    r"cc:\s|very truly yours)",
    re.IGNORECASE,
)

# 加载 CUAD 片段时跳过的列。
_SKIP_COLUMNS = frozenset({
    "Filename", "Document Name", "Document Name-Answer",
    "Parties", "Parties-Answer",
})


# ======================================================================
# 语言现象检测
# ======================================================================


def detect_phenomena(
    tree: DependencyTree,
    long_distance_mdd: float,
) -> Dict[str, bool]:
    """在依存 parse 树中检测语言现象。

    参数：
        tree: 单条合同条款解析得到的 DependencyTree。
        long_distance_mdd: 来自 constraints.yaml
            validation.long_distance_mdd 的平均依存距离阈值。
    """
    phenomena: Dict[str, bool] = {}

    # --- 被动语态 ---
    passive_detected = False
    for token in tree.find_tokens_by_upos("VERB"):
        if PassiveDetector.is_passive_loose(tree, token.index):
            passive_detected = True
            break
    phenomena["passive"] = passive_detected

    # --- 条件从句：依存检测（advcl + mark）---
    has_advcl = tree.has_deprel("advcl")
    has_mark = tree.has_deprel("mark")
    phenomena["conditional"] = has_advcl and has_mark

    # --- 条件从句：词汇回退检测 ---
    # 部分条件从句（尤其由 "if"/"unless"/"subject to" 引导的）
    # 可能未被 Stanza 解析为 advcl+mark。
    # 用文本中的词汇标记作回退。
    if not phenomena["conditional"]:
        text_lower = tree.text.lower()
        lexical_conditionals = [
            " if ", " unless ", " provided that ", " so long as ",
            " subject to ", " in the event that ", " in the event of ",
            " on condition that ", " conditioned upon ",
            " except ", " except as ", " other than ",
            " upon ", " within ", " after ", " before ",
        ]
        for marker in lexical_conditionals:
            if marker in text_lower:
                phenomena["conditional"] = True
                break

    # --- 关系从句 ---
    phenomena["relative_clause"] = tree.has_deprel("acl:relcl")

    # --- 长距离依存（平均依存距离 > 阈值）---
    mdd = compute_mean_dependency_distance(tree)
    phenomena["long_distance"] = mdd > long_distance_mdd

    # --- 否定 ---
    phenomena["negation"] = tree.has_deprel("neg")

    # --- 定义性条款检测 ---
    phenomena["is_definition"] = _is_definition_clause(tree)

    return phenomena


def _is_definition_clause(tree: DependencyTree) -> bool:
    """判断条款是否为术语定义而非操作性条款。

    识别模式：
      - "X means Y" / "X shall mean Y"
      - '"Term" means ...'（引号术语 + means）
      - '1.4 "Term" means ...'（编号 + 引号术语 + 定义）

    参数：
        tree: 单条合同条款解析得到的 DependencyTree。

    返回：
        若为定义性条款则为 True。
    """
    text = tree.text.strip()
    text_lower = text.lower()

    # 模式 1："X means Y" — 主语为引号术语或短名词短语。
    if re.search(r'\bmeans\b', text_lower):
        if text.startswith('"') or text[0].isdigit():
            return True
        if re.search(r'["\']?\w+["\']?\s+(?:shall\s+)?means?\b', text_lower):
            return True

    # 模式 2：编号 + 引号术语开头（如 "1.4 'Term' means..."）。
    if re.match(r'^\d+\.\d+\s+["\']', text):
        return True

    return False


def is_boilerplate_clause(text: str) -> bool:
    """启发式过滤非操作性文本片段。"""
    t = text.strip()
    if len(t) < 15:
        return True
    if _BOILERPLATE_RE.match(t):
        return True
    words = t.split()
    if len(words) <= 8 and t.isupper():
        return True
    return False
