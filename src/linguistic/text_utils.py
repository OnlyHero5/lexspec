"""
文本匹配与规范化工具
==========================================
用于将大语言模型抽取文本与 UD 词元比对的自包含工具函数。
从 ConstraintValidator 中提取，以保持该模块专注于核心校验算法。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.extraction.schema import Token

from src.utils.logging import get_logger

logger = get_logger(__name__)


def normalize_text(text: str) -> str:
    """规范化文本以便比对。

    按顺序执行以下步骤：
    1. 将整个字符串转为小写。
    2. 去除前导冠词："the Seller" -> "seller"，
       "a Party" -> "party"，"an Agreement" -> "agreement"。
    3. 去除尾部标点：句号、逗号、分号等。
    4. 将多个空白字符合并为单个空格。
    5. 去除首尾空白。

    此规范化确保大小写、冠词和标点等表层差异
    不会导致误报不匹配。目标是比对语义内容，
    而非表层形式。

    参数：
        text: 来自大语言模型或 UD 的原始文本字符串。

    返回：
        适用于比对的规范化文本。
    """
    if not text:
        return ""

    normalized = text.lower().strip()

    # 去除前导冠词（"the"、"a"、"an"）。
    # 使用词边界检查，避免从词中间去除 "the"
    #（例如 "theory"）。
    normalized = re.sub(
        r'^\s*(the|a|an)\s+', '', normalized, count=1
    )

    # 去除尾部标点。
    normalized = normalized.rstrip(".,;:!?\"'()[]{}")

    # 合并空白。
    normalized = re.sub(r'\s+', ' ', normalized)

    # 最终去除首尾空白。
    normalized = normalized.strip()

    return normalized


def match_text(llm_text: str, ud_token: "Token") -> bool:
    """检查大语言模型抽取文本是否与 UD 词元匹配。

    在规范化（小写、去除冠词、去除标点）后进行匹配。检查：

    1. 双方规范化后的精确匹配。
    2. 子串匹配：UD 词元文本是大语言模型文本的子串
       （处理并列扩展，如 "Buyer and Seller" 匹配 UD "Buyer"），
       或反之。
    3. 词元重叠：两字符串至少共享一个实词
       （名词、动词、形容词）。适用于大语言模型抽取了
       更长的名词短语但中心名词与 UD 词元匹配的情况。

    参数：
        llm_text: 大语言模型抽取的文本。
        ud_token: UD 解析中的词元。

    返回：
        规范化后文本匹配则返回 True。
    """
    llm_norm = normalize_text(llm_text)
    ud_norm = normalize_text(ud_token.text)

    # 规范化后精确匹配。
    if llm_norm == ud_norm:
        return True

    # 子串匹配（处理并列与名词短语扩展）。
    if ud_norm in llm_norm or llm_norm in ud_norm:
        logger.debug(
            "Substring match: LLM='%s' contains UD='%s'",
            llm_norm, ud_norm,
        )
        return True

    # 词元重叠：将双方拆分为词集，检查是否
    # 至少有一个共同的实词。
    llm_words = set(llm_norm.split())
    ud_words = set(ud_norm.split())

    # 从考虑范围中排除功能词。
    function_words = {
        "the", "a", "an", "of", "in", "to", "for", "on", "at",
        "by", "with", "from", "and", "or", "not", "no", "any",
        "all", "each", "every", "such", "its", "his", "her",
    }
    llm_content = llm_words - function_words
    ud_content = ud_words - function_words

    common = llm_content & ud_content
    if common:
        logger.debug(
            "Content-word overlap match: %s", common,
        )
        return True

    return False
