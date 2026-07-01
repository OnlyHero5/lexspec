"""
PolarityDetector 的主检测接口。
"""

from __future__ import annotations

from typing import Tuple

from src.extraction.schema import DependencyTree, LegalRole
from src.linguistic.ud_features import (
    find_aux_verb,
    find_obl_agent,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def detect(
    self,
    tree: DependencyTree,
    predicate_idx: int,
    predicate_lemma: str = "",
) -> Tuple[LegalRole, str]:
    """检测谓词主语的法律角色与极性。

    决策逻辑（有序，先匹配者优先）：

    1. 词项覆盖：若 predicate_lemma 为 "indemnify"，主语为
       INDEMNIFYING_PARTY，与情态无关。此为法律惯例 ——
       赔偿义务属于特定义务类别。

    2. 检测否定：检查谓词是否有 neg 依存，
       或附近是否出现否定小品词。

    3. 检测情态助动词：查找 aux 动词（aux 关系）。
       提取词元形式供规则匹配。

    4. 在 modality_rules 中查找：映射 (aux_lemma, is_negated) -> LegalRole。

    5. 回退：未找到情态时返回 OTHER。
       见于无情态的从句：
       - 一般现在时："Seller delivers the Goods."
       - 过去时："Seller delivered the Goods."
       - 不定式补语："Seller agrees to deliver..."

    参数：
        tree: 依存树。
        predicate_idx: 谓词的 1 基索引。
        predicate_lemma: 谓词动词词元（用于 "indemnify" 等词项覆盖）。

    返回：
        (LegalRole, polarity) 元组。
        - role: 分类后的法律角色枚举。
        - polarity: "positive" 或 "negative" 字符串。
    """
    # 步骤 1：赔偿义务的词项覆盖。
    # "Indemnify" 及其形态变体始终将主语标为赔偿方。
    # 覆盖任何基于情态的分类，因赔偿的法律语义
    # 区别于简单义务。
    if predicate_lemma.lower() in ("indemnify", "indemnified"):
        logger.debug(
            "Lexical override: predicate '%s' -> INDEMNIFYING_PARTY",
            predicate_lemma,
        )
        return (LegalRole.INDEMNIFYING_PARTY, "positive")

    # 步骤 2：检测否定。
    is_negated = self._is_negated(tree, predicate_idx)
    polarity = "negative" if is_negated else "positive"

    # 步骤 3：检测情态助动词。
    modal_word, _ = detect_modality(self, tree, predicate_idx)

    if not modal_word:
        # 未找到情态助动词。
        # 谓词无道义标记 —— 仅凭句法无法确定角色。
        # 常见于定义条款、鉴于条款与裸事实陈述。
        logger.debug(
            "No modal auxiliary found for predicate at index %d — "
            "role is OTHER", predicate_idx,
        )
        return (LegalRole.OTHER, polarity)

    # 步骤 4：查找 (modal, negated) -> role。
    role = self._lookup.get((modal_word.lower(), is_negated))

    if role is not None:
        logger.debug(
            "Role classified: aux='%s', negated=%s -> %s",
            modal_word, is_negated, role.value,
        )
        return (role, polarity)

    # 步骤 5：找到情态但不匹配任何规则。
    # 可能是非道义情态（如表能力的 "can"、表将来的 "will"），
    # 或不在规则集中的情态。
    # 默认为 OTHER —— 校验器将记为不确定。
    logger.debug(
        "Modal '%s' (negated=%s) does not match any rule — "
        "role is OTHER", modal_word, is_negated,
    )
    return (LegalRole.OTHER, polarity)


def detect_modality(
    self,
    tree: DependencyTree,
    predicate_idx: int,
) -> Tuple[str, str]:
    """检测具体情态助动词与极性。

    比 detect() 更细粒度 —— 返回实际情态词，
    供语言学证据与错误说明使用。

    用于：
      - 校验器 _build_linguistic_evidence() 填充 modality_aux 字段。
      - 错误分析器解释角色不匹配。

    参数：
        tree: 依存树。
        predicate_idx: 谓词的 1 基索引。

    返回：
        (modal_word, polarity) 元组。
        - modal_word: 情态助动词词元（如 "shall"、"may"、"must"），
          无辅助动词时为空字符串。
        - polarity: "positive" 或 "negative" 字符串。
    """
    # 查找助动词（若有）。
    aux_token = find_aux_verb(tree, predicate_idx)

    # 根据否定确定极性。
    is_negated = self._is_negated(tree, predicate_idx)
    polarity = "negative" if is_negated else "positive"

    if aux_token is None:
        return ("", polarity)

    # 使用词元形式以一致匹配规则。
    # Stanza 将情态还原为规范形式：
    # "must" -> "must"，"shall" -> "shall"，"may" -> "may"。
    modal_word = aux_token.lemma.lower() if aux_token.lemma else aux_token.text.lower()

    logger.debug(
        "Detected modality: aux='%s' (lemma='%s'), polarity='%s'",
        aux_token.text, modal_word, polarity,
    )

    return (modal_word, polarity)


def detect_role_with_voice(
    self,
    tree: DependencyTree,
    predicate_idx: int,
    predicate_lemma: str = "",
    is_passive: bool = False,
) -> LegalRole:
    """检测法律角色，考虑被动语态。

    被动构造中，表层主语为受事而非施事。
    角色（义务方等）应赋给语义施事（obl:agent），
    而非表层主语。

    本方法正确处理：
      1. 主动语态：角色适用于 nsubj（表层主语 = 施事）。
      2. 被动语态：角色适用于 obl:agent（语义施事）。
      3. 无施事被动：无法确定角色 —— 返回 OTHER。

    参数：
        tree: 依存树。
        predicate_idx: 谓词的 1 基索引。
        predicate_lemma: 谓词词元。
        is_passive: 谓词是否为被动语态。

    返回：
        LegalRole 枚举值。
    """
    role, _ = detect(self, tree, predicate_idx, predicate_lemma)

    if not is_passive:
        return role

    # 被动语态下检查 obl:agent 是否存在。
    # 若通过 obl:agent 表达施事，角色适用于该施事。
    # 若无施事，角色「悬空」—— 知有情态但不知适用对象。
    agent = find_obl_agent(tree, predicate_idx)
    if agent is None:
        # 带情态的无施事被动，例如
        # "All notices shall be delivered in writing."
        # 知有义务（shall），但句法施事未表达。
        # 大语言模型需从文档上下文推断
        # 哪方承担义务。
        logger.debug(
            "Agentless passive with modality — role cannot be "
            "assigned to a syntactic agent."
        )
        return LegalRole.OTHER

    return role


def get_modality_evidence(
    self,
    tree: DependencyTree,
    predicate_idx: int,
) -> dict:
    """提取情态相关证据供错误分析使用。

    返回情态特征的完整字典，供错误分析器
    解释角色分类决策。

    参数：
        tree: 依存树。
        predicate_idx: 谓词的 1 基索引。

    返回：
        含以下键的字典：
          - has_aux (bool): 存在 aux 关系
          - aux_lemma (str): 助动词词元（若有）
          - aux_text (str): 助动词表层文本
          - is_negated (bool): 检测到否定
          - polarity (str): "positive" 或 "negative"
          - found_roles (list): 匹配 (aux, neg) 对的全部角色
    """
    aux_token = find_aux_verb(tree, predicate_idx)
    is_negated = self._is_negated(tree, predicate_idx)

    evidence = {
        "has_aux": aux_token is not None,
        "aux_lemma": aux_token.lemma.lower() if aux_token and aux_token.lemma else "",
        "aux_text": aux_token.text.lower() if aux_token else "",
        "is_negated": is_negated,
        "polarity": "negative" if is_negated else "positive",
        "matched_roles": [],
    }

    # 检查哪些角色匹配此 (aux, negated) 组合。
    if aux_token and aux_token.lemma:
        aux_lower = aux_token.lemma.lower()
        for (role_aux, role_neg), role in self._lookup.items():
            if role_aux == aux_lower and role_neg == is_negated:
                evidence["matched_roles"].append(role.value)

    return evidence
