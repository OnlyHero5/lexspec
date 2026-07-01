"""
被动检测：is_passive、is_passive_loose、get_passive_features。
"""

from __future__ import annotations

from src.extraction.schema import DependencyTree
from src.linguistic.ud_features import (
    find_nsubj_pass,
    find_obl_agent,
    find_aux_pass,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def is_passive(tree: DependencyTree, predicate_idx: int) -> bool:
    """检测谓词是否为被动语态。

    检测规则（置信检测需同时满足）：

    1. 存在 nsubj:pass 关系：存在表层主语，语义上为受事。
       这是被动语态的定义特征 —— 主动语态中作宾语的论元
       被提升为主语位置。

    2. 存在 aux:pass 关系：存在被动助动词 "be"/"get"。
       确认构造为形态被动（动词形式：be + 过去分词），
       区别于形容词分词构造（"The door remained
       closed" —— 可能有 nsubj:pass 但无 aux:pass，
       故为形容词性而非被动）。

    要求两者同时存在的语言学理由：
      - 单独 nsubj:pass 可出现在形容词过去分词
        （"the documents attached" —— "attached" 为形容词性）
        及某些非作格构造中。
      - 单独 aux:pass（无 nsubj:pass）暗示虚词或非人称构造
        （"It was decided that..." —— 虚词 "it" 为 nsubj，
        非 nsubj:pass）。
      - 两者兼有 = 明确的形态被动。

    参数：
        tree: 依存树。
        predicate_idx: 谓词词元的 1 基索引。

    返回：
        置信检测到被动语态时返回 True。

    注意：
        对无施事被动（"The Goods were delivered"），
        因 nsubj:pass + aux:pass 均存在仍返回 True。
        obl:agent 的缺失在 restore_args() 中单独处理。
    """
    # 准则 1：是否存在被动主语（受事处于主语位置）？
    has_passive_subject = find_nsubj_pass(tree, predicate_idx) is not None

    # 准则 2：是否存在被动助动词？
    has_passive_aux = find_aux_pass(tree, predicate_idx) is not None

    if has_passive_subject and has_passive_aux:
        logger.debug(
            "Passive detected at predicate index %d: nsubj:pass + aux:pass",
            predicate_idx,
        )
        return True

    if has_passive_subject and not has_passive_aux:
        # 可能为形容词分词或非作格。
        # 例："The documents attached hereto" —
        # "attached" 可能有 nsubj:pass 但为形容词性而非被动。
        # 法律文本中少见但值得记录。
        logger.debug(
            "nsubj:pass found at predicate %d but no aux:pass — "
            "may be adjectival, not passive. Treating as non-passive.",
            predicate_idx,
        )
        return False

    if not has_passive_subject and has_passive_aux:
        # 少见：有 aux:pass 无 nsubj:pass。
        # 可能为带虚词 "it" 的非人称被动。
        # 例："It is agreed that..." —— "it" 为 nsubj（虚词），
        # 非 nsubj:pass。因存在 aux:pass 仍视为被动。
        logger.debug(
            "aux:pass found at predicate %d but no nsubj:pass — "
            "possible impersonal passive. Treating as passive.",
            predicate_idx,
        )
        return True

    return False


def is_passive_loose(tree: DependencyTree, predicate_idx: int) -> bool:
    """宽松被动检测 —— 仅需 nsubj:pass 或 aux:pass 之一。

    更宽松的检查，用于测试集采样
    （现象分类），以捕获边界情况。
    严格 is_passive() 用于校验/修正，
    因假阳性代价高于假阴性。

    参数：
        tree: 依存树。
        predicate_idx: 谓词词元的 1 基索引。

    返回：
        存在 nsubj:pass 或 aux:pass 之一时返回 True。
    """
    return (
        find_nsubj_pass(tree, predicate_idx) is not None
        or find_aux_pass(tree, predicate_idx) is not None
    )


def get_passive_features(
    tree: DependencyTree,
    predicate_idx: int,
) -> dict:
    """提取全部被动相关特征供错误分析使用。

    返回刻画被动构造的特征字典。供错误分析器
    解释特定被动构造导致大语言模型抽取错误的原因。

    参数：
        tree: 依存树。
        predicate_idx: 谓词词元的 1 基索引。

    返回：
        含以下键的字典：
          - is_passive (bool): 严格被动检测结果
          - is_passive_loose (bool): 宽松检测结果
          - has_nsubj_pass (bool): 存在 nsubj:pass 关系
          - has_aux_pass (bool): 存在 aux:pass 关系
          - has_obl_agent (bool): 存在 obl:agent 关系
          - is_agentless (bool): 被动且无表达的施事
          - subject_text (str): nsubj:pass 文本（表层主语/受事）
          - agent_text (str): obl:agent 文本（若存在）
    """
    nsubj_pass = find_nsubj_pass(tree, predicate_idx)
    aux_pass = find_aux_pass(tree, predicate_idx)
    obl_agent = find_obl_agent(tree, predicate_idx)

    return {
        "is_passive": is_passive(tree, predicate_idx),
        "is_passive_loose": is_passive_loose(tree, predicate_idx),
        "has_nsubj_pass": nsubj_pass is not None,
        "has_aux_pass": aux_pass is not None,
        "has_obl_agent": obl_agent is not None,
        "is_agentless": (
            is_passive(tree, predicate_idx)
            and obl_agent is None
        ),
        "subject_text": nsubj_pass.text if nsubj_pass else "",
        "agent_text": obl_agent.text if obl_agent else "",
    }
