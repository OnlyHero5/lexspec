"""
论元提取：find_nsubj、find_obj、find_nsubj_pass、find_obl_agent。
"""

from __future__ import annotations

from typing import Optional

from src.extraction.schema import DependencyTree, Token
from src.utils.logging import get_logger

logger = get_logger(__name__)


def find_nsubj(tree: DependencyTree, predicate_idx: int) -> Optional[Token]:
    """查找名词性主语（主动语态施事/行为者）。

    UD: nsubj(predicate, subject) —— 从句的句法主语。
    主动语态中为施事/行为者 —— 执行谓词所述动作的实体。

    法律文本中标识承担/执行动作的当事方：
      "SELLER shall deliver"  -> nsubj(deliver, Seller)
      "BUYER must pay"        -> nsubj(pay, Buyer)

    参数：
        tree: 依存树。
        predicate_idx: 谓词词元的 1 基索引。

    返回：
        nsubj Token，无名词性主语时返回 None
        （如祈使从句、非人称构造）。

    注意：
        UD 中动词至多一个 nsubj。并列
        （"Buyer and Seller shall deliver"）使用 conj 关系
        —— 第一并列项为 nsubj，其余为 conj 依存。
    """
    children = tree.get_children(predicate_idx, deprel="nsubj")
    if children:
        if len(children) > 1:
            logger.debug(
                "Multiple nsubj candidates for predicate %d — "
                "returning first. This may indicate a parse error.",
                predicate_idx,
            )
        return children[0]
    return None


def find_obj(tree: DependencyTree, predicate_idx: int) -> Optional[Token]:
    """查找直接宾语（主动语态受事/主题）。

    UD: obj(predicate, object) —— 及物动词的直接宾语。
    主动语态中为受事/主题 —— 承受动作的实体。

    法律文本中标识动作作用对象：
      "deliver THE GOODS"       -> obj(deliver, goods)
      "pay ALL AMOUNTS DUE"     -> obj(pay, amounts)
      "indemnify THE COMPANY"   -> obj(indemnify, company)

    参数：
        tree: 依存树。
        predicate_idx: 谓词词元的 1 基索引。

    返回：
        obj Token，谓词不及物时返回 None
        （无直接宾语，如 "The Agreement terminates"）。

    注意：
        法律动词绝大多数为及物（作用于某物）。
        不及物谓词存在（"The Agreement shall terminate"）
        但不产生 obj —— 动作无直接受事。
    """
    children = tree.get_children(predicate_idx, deprel="obj")
    if children:
        return children[0]
    return None


def find_nsubj_pass(tree: DependencyTree,
                    predicate_idx: int) -> Optional[Token]:
    """查找被动名词性主语（表层主语，语义受事）。

    UD: nsubj:pass(predicate, patient) —— 被动从句的句法主语。
    表层主语语义上为受事/主题，非行为者。行为者以 obl:agent（by 短语）出现。

    这是大语言模型抽取错误的最常见来源：大语言模型常将
    表层主语（nsubj:pass）当作施事，导致三元组主宾颠倒。

    示例：
      "THE GOODS shall be delivered by Seller"
        -> nsubj:pass(delivered, goods)  <-- 受事（表层主语）
        -> obl:agent(delivered, Seller)  <-- 施事（真实行为者）

    校验器使用本函数检测大语言模型是否将被动主语误作施事并相应修正。

    参数：
        tree: 依存树。
        predicate_idx: 谓词词元的 1 基索引。

    返回：
        nsubj:pass Token（语义受事），谓词非被动时返回 None。
    """
    children = tree.get_children(predicate_idx, deprel="nsubj:pass")
    if children:
        return children[0]
    return None


def find_obl_agent(tree: DependencyTree,
                   predicate_idx: int) -> Optional[Token]:
    """查找斜格施事（by 短语，被动中的语义行为者）。

    UD: obl:agent(predicate, agent) —— 被动构造中的施事，
    通常由介词 "by" 引入。此为真实语义行为者 —— 执行动作的实体。

    UD 指南指出 obl:agent 专用于由 "by" 标记的被动施事
    （非英语中等价表达）。为一般 obl（斜格）关系的语言特定子类型。

    法律文本中，obl:agent 为尽管表层主语为其他实体、
    实际执行动作的当事方：
      "delivered BY SELLER"       -> obl:agent(delivered, Seller)
      "indemnified BY THE COMPANY" -> obl:agent(indemnified, company)

    边界情况：无施事被动。
      "The goods were delivered."（无 "by" 短语）
      -> obl:agent 为 None。施事语义存在但
         句法未表达。校验器标记为 REFLEXION_REQUIRED，
         因大语言模型须从话语上下文推断施事。

    参数：
        tree: 依存树。
        predicate_idx: 谓词词元的 1 基索引。

    返回：
        obl:agent Token（语义施事/行为者），或 None。
    """
    children = tree.get_children(predicate_idx, deprel="obl:agent")
    if children:
        return children[0]
    return None
