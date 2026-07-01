"""
被动论元恢复：restore_passive_args、get_active_args。
"""

from __future__ import annotations

from typing import Optional, Tuple

from src.extraction.schema import DependencyTree, Token
from src.linguistic.ud_features import (
    find_nsubj_pass,
    find_obl_agent,
    find_nsubj,
    find_obj,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def restore_passive_args(
    tree: DependencyTree,
    predicate_idx: int,
) -> Tuple[Optional[Token], Optional[Token]]:
    """从被动构造恢复语义施事与受事。

    通过「撤销」被动变换，将表层句法映射到语义角色：

    表层句法（被动）              语义角色（主动等价）
    ----------------------------     ----------------------------------
    nsubj:pass 词元（表层主语） -> 语义受事（-> 三元组宾语）
    obl:agent 词元（by 短语）   -> 语义施事（-> 三元组主语）

    此映射是大语言模型错误的关键修正：大语言模型
    常将 nsubj:pass 当作施事（主语），实为受事（宾语）。

    变换示例：
      输入:  "The Goods shall be delivered by Seller within 30 days."
      输出: agent=Token("Seller")  (-> 三元组主语)
            patient=Token("Goods") (-> 三元组宾语)

    无施事被动处理：
      "The Goods were delivered."
      -> agent = None  (隐含 —— REFLEXION_REQUIRED)
      -> patient = Token("Goods")  (正确宾语)

    参数：
        tree: 依存树。
        predicate_idx: 谓词词元的 1 基索引。

    返回：
        (agent, patient) 元组。任一方可为 None。
        - agent: obl:agent 词元（真实行为者 -> 三元组主语）。
                 无施事被动时为 None。
        - patient: nsubj:pass 词元（真实受事 -> 三元组宾语）。
                   被动检测失败时为 None。

    注意：
        调用方应先调用 is_passive() 再调用本方法。
        对非被动谓词调用时，patient 将为 None。
    """
    # 被动中的语义施事为 obl:agent（by 短语行为者）。
    # 此为执行动作的主体，应成为修正后三元组的主语。
    agent = find_obl_agent(tree, predicate_idx)

    # 被动中的语义受事为 nsubj:pass（表层主语）。
    # 此为承受动作的主体，应成为修正后三元组的宾语。
    patient = find_nsubj_pass(tree, predicate_idx)

    if agent is None and patient is not None:
        # 无施事被动：施事语义隐含但
        # 句法未表达。法律文本中常见，
        # 责任方由上下文明确时：
        #   "All notices shall be delivered in writing."
        # 校验器标记为 REFLEXION_REQUIRED，
        # 因大语言模型须推断谁送达通知。
        logger.debug(
            "Agentless passive at predicate %d: agent is implied "
            "but not expressed syntactically. Marking for Reflexion.",
            predicate_idx,
        )

    if agent is not None and patient is not None:
        logger.debug(
            "Restored passive args: agent='%s' (index %d), "
            "patient='%s' (index %d)",
            agent.text, agent.index,
            patient.text, patient.index,
        )
    elif agent is not None:
        logger.debug(
            "Partial passive restoration: agent='%s', no patient found",
            agent.text,
        )
    elif patient is not None:
        logger.debug(
            "Partial passive restoration: patient='%s', no agent found "
            "(agentless passive)",
            patient.text,
        )

    return (agent, patient)


def get_active_args(
    tree: DependencyTree,
    predicate_idx: int,
) -> Tuple[Optional[Token], Optional[Token]]:
    """获取主动语态谓词的语义施事与受事。

    为 restore_passive_args() 的主动语态对应方法。
    主动语态下映射直接：
      - 施事 = nsubj（主语 = 行为者）
      - 受事 = obj（直接宾语 = 承受者）

    当 is_passive() 返回 False 时，校验器使用本方法
    获取主动论元以与大语言模型输出比对。

    参数：
        tree: 依存树。
        predicate_idx: 谓词词元的 1 基索引。

    返回：
        (agent, patient) 元组。
        - agent: nsubj 词元（行为者 -> 三元组主语）
        - patient: obj 词元（承受者 -> 三元组宾语）
    """
    agent = find_nsubj(tree, predicate_idx)
    patient = find_obj(tree, predicate_idx)
    return (agent, patient)
