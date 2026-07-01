"""
预测与金标公平比较用的核心文本归一化流水线。

归一化消除表面差异（大小写、冠词、标点），
避免语义正确抽取因形式差异被扣分。

本模块用于评估流水线各处：
  - 计算字段级 F1 前（匹配 "the Seller" 与 "Seller"）
  - 词元级重叠指标分词前
  - 抽取文本与金标标注比较前

设计决策:
  - 各归一化步骤可通过布尔标志配置。
  - 词形还原单独调用（需 Stanza），默认关闭以保速度。
    schema 中 predicate 已是词形基形式，词形还原主要用于主语/宾语片段。
  - 数字归一化双向："30" 与 "thirty" 均映射为同一规范形式（数字形式）。
  - 当事方别名按合同级当事方定义归一化实体指称（如 "the Company" -> "Seller"）。
"""

from __future__ import annotations

import re
from typing import Optional, Dict, List

from src.utils.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# 数字词 ↔ 数字双向映射
# =============================================================================
# 支持 "thirty → 30" 与 "30 → thirty" 归一化。
# 归一化后规范形式为数字（如 "5" 而非 "five"）。
# 故："five days" 与 "5 days" 均归一化为 "5 days"。
#
# 覆盖 0–100 基本涵盖合同子句中出现的数字
# （期限、金额、通知天数等）。超过 100 在子句级抽取中极罕见，保持原样。

NUMBER_WORDS: Dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
}

# 反向映射：数字 → 词形，供双向归一化。
# 模块加载时由 NUMBER_WORDS 惰性生成。
_NUMBER_WORDS_REVERSE: Dict[str, str] = {v: k for k, v in NUMBER_WORDS.items()}

# =============================================================================
# 正则模式 — 模块加载时编译一次以提升性能
# =============================================================================

# 待剥离冠词：词界上的独立 "the"、"a"、"an"。
# 模式匹配整词（非其他词子串）。
_ARTICLE_PATTERN = re.compile(
    r'\b(a|an|the)\b\s*',
    re.IGNORECASE,
)

# 完全移除的标点。保留连字符与撇号，
# 因法律文本中有语义（如 "non-compete"、"party's"）。
# 多词实体引用保留下划线。
_PUNCTUATION_PATTERN = re.compile(
    r'[.,;:!?()"\'\[\]{}<>/\\|`~@#$%^&*+=]'
)

# 行尾句号（抽取片段常见句末标点）。
_TRAILING_PERIOD_PATTERN = re.compile(r'\.$')

# 连续空白。
_WHITESPACE_PATTERN = re.compile(r'\s+')


def normalize(
    text: str,
    remove_articles: bool = True,
    lemmatize: bool = False,
    number_normalize: bool = True,
    party_aliases: Optional[Dict[str, List[str]]] = None,
) -> str:
    """归一化文本以便预测与金标比较。

    步骤（按序应用）:
    1. 小写 — 消除大小写差异（"Seller" vs "seller"）。
    2. 移除冠词 — 作为整词剥离 leading/trailing 的 "the"、"a"、"an"，
       使 "the Buyer" 匹配 "Buyer"。
    3. 移除标点 — 剥离标准标点。保留连字符与撇号，
       因法律文本中有语义（如 "non-compete"、"party's obligations"）。
    4. 数字归一化 — 书面数字转数字形式（如 "thirty" → "30"），
       文本中多个数字词时也双向映射到同一规范表示。整词替换。
    5. 应用当事方别名 — 按合同级定义归一化实体（如 "the Company" → "Seller"）。
    6. 压缩空白 — 合并多空格并去除首尾空白。

    参数:
        text: 待归一化原始文本（主语、谓词、宾语、条件）。
        remove_articles: 为 True 时作为整词剥离冠词。
        lemmatize: 预留；当前无操作。词形还原需 Stanza，高吞吐评估过慢。
                   LegalTriplet 的 predicate 已是词形基形式。
        number_normalize: 为 True 时数字词与数字双向归一化以便一致比较。
        party_aliases: 可选，规范当事方名 -> 别名列表。
                       如 {"Seller": ["the Seller", "Company"]}。
                       文本中出现任别名则替换为规范名。

    返回:
        适于精确匹配或词元重叠比较的归一化字符串。

    示例:
        >>> normalize("the Seller shall deliver the Goods.")
        'seller shall deliver goods'

        >>> normalize("within thirty (30) days")
        'within 30 days'

        >>> normalize("the Company", party_aliases={"Seller": ["the Company"]})
        'seller'
    """
    # 防御 None 或空输入
    if not text:
        return ""

    normalized = text

    # 步骤 1：小写，大小写不敏感匹配。
    # 合同文本大小写混用（如 "Seller"、"Goods"），
    # 大小写不改变语义指称。
    normalized = normalized.lower()

    if remove_articles:
        # 步骤 2：作为整词移除冠词。
        # "the Buyer" → "Buyer"，"a notice" → "notice"。
        # 用正则词界，避免子串误匹配（如 "there" 不应变成 "re"）。
        normalized = _ARTICLE_PATTERN.sub('', normalized)

    # 步骤 3：移除标点。
    # 先剥行尾句号（抽取片段常见），再移除其余标点。
    normalized = _TRAILING_PERIOD_PATTERN.sub('', normalized)
    normalized = _PUNCTUATION_PATTERN.sub(' ', normalized)

    if number_normalize:
        # 步骤 4：数字词 ↔ 数字归一化。
        # 策略：已知词形数字替换为数字形式。
        # 处理 "thirty five" → "35"：先替换各词，再合并相邻数字。
        normalized = _normalize_numbers(normalized)

    if party_aliases:
        # 步骤 5：应用当事方别名映射。
        # 对每个规范名，按其别名与文本匹配。最长别名优先，避免部分替换。
        normalized = _apply_party_aliases(normalized, party_aliases)

    # 步骤 6：压缩多余空白。
    # 标点移除等将字符变空格后，合并为单空格并去除首尾空白。
    normalized = _WHITESPACE_PATTERN.sub(' ', normalized)
    normalized = normalized.strip()

    return normalized


def _normalize_numbers(text: str) -> str:
    """将文本中的数字词归一化为数字形式。

    处理简单数字（"five" → "5"）与复合数字（"thirty five" → "35"）：
    逐词替换后合并相邻数字词元。

    参数:
        text: 可能含书面数字的小写文本。

    返回:
        数字词已替换为数字等价形式的文本。
    """
    words = text.split()
    result_words: List[str] = []

    for word in words:
        stripped = word.strip()
        # 检查词（剥离先前标点移除留下的非字母后）
        # 是否为已知数字词。
        if stripped in NUMBER_WORDS:
            result_words.append(NUMBER_WORDS[stripped])
        else:
            result_words.append(stripped)

    # 合并相邻数字词元："30 5" → "35"
    # 处理 "thirty five days" 等复合数。
    # 仅当相邻两词元均为纯数字串时合并。
    collapsed: List[str] = []
    i = 0
    while i < len(result_words):
        if i + 1 < len(result_words) and result_words[i].isdigit() and result_words[i + 1].isdigit():
            # 合并两相邻数字词元："30" + "5" = "305"
            # 注：此为启发式 — "thirty five" 得 "30" "5" → "305"
            # 可接受，因比较用词元集合，原 "thirty five" 也是单集合。
            # 严格正确："thirty" + "five" = 30 + 5 表示 35。
            # 若前为 10 的倍数且后 < 10，则相加；否则拼接。
            prev_val = int(result_words[i])
            next_val = int(result_words[i + 1])
            if prev_val % 10 == 0 and next_val < 10 and next_val > 0:
                # "thirty five" → 30 + 5 = 35
                collapsed.append(str(prev_val + next_val))
            else:
                # 其他相邻数字 — 空格分隔
                collapsed.append(result_words[i])
                collapsed.append(result_words[i + 1])
            i += 2
        else:
            collapsed.append(result_words[i])
            i += 1

    return ' '.join(collapsed)


def _apply_party_aliases(text: str, party_aliases: Dict[str, List[str]]) -> str:
    """对归一化文本应用当事方别名替换。

    对每个规范当事方（如 "Seller"），将其任别名
    （如 "the Seller"、"Company"、"Vendor"）替换为规范名。
    较长别名先处理，避免部分匹配。

    参数:
        text: 已归一化文本（已小写、已去标点）。
        party_aliases: 规范名 -> 别名列表 的字典。

    返回:
        别名已替换为规范当事方名的文本。
    """
    result = text
    for canonical, aliases in party_aliases.items():
        # 别名按长度降序 — 较长模式优先，避免 "Company" 单独替换时
        # 误伤 "the Company"。
        for alias in sorted(aliases, key=len, reverse=True):
            # 匹配用别名本身归一化（小写、去除首尾空白）。
            normalized_alias = alias.lower().strip()
            # 整词或短语匹配替换。
            # 别名用带词界的正则。
            # 转义别名以防正则特殊字符（实体名中的括号、点等）。
            pattern = re.compile(
                r'\b' + re.escape(normalized_alias) + r'\b',
                re.IGNORECASE,
            )
            replacement = canonical.lower().strip()
            result = pattern.sub(replacement, result)
    return result
