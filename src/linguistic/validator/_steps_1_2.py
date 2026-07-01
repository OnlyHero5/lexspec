"""
校验步骤 1 与 2：谓词定位与语态检测。

步骤 1：通过 UD 解析定位根谓词。
步骤 2：检测被动语态并恢复语义论元。
"""

from __future__ import annotations

from typing import Optional, Tuple

from src.extraction.schema import DependencyTree, Token
from src.linguistic.ud_features import find_root_predicate
from src.linguistic.passive_detector import PassiveDetector
from src.utils.logging import get_logger

logger = get_logger(__name__)


def step1_find_predicate(tree: DependencyTree) -> Optional[Token]:
    """步骤 1：通过 UD 解析定位根谓词。

    根词元（head=0）通常为主句动词。
    这是后续全部分析的起点，因 UD 标注中
    全部论元（主语、宾语、条件）均为根谓词的依存。

    UD 依据：依存树的根为 head 字段为 0 的词元。
    格式良好的 UD 树中恰有一个根。若有多个（畸形解析），
    选择第一个 VERB 根。

    法律从句中根几乎总是 VERB：
      "Seller shall deliver the Goods." -> root = "deliver"
      "The Agreement shall be governed by..." -> root = "governed"

    边界情况：根可能为 AUX（系词），语义谓词在 xcomp 补语中。
    尝试解析：
      "The Agreement IS binding." -> root = AUX "is"
      再检查 xcomp(IS, binding) -> predicate = "binding"

    参数：
        tree: 依存树。

    返回：
        根谓词 Token，未找到根时返回 None。
    """
    return find_root_predicate(tree)


def step2_detect_voice(
    tree: DependencyTree,
    predicate_idx: int,
) -> Tuple[bool, Optional[Token], Optional[Token]]:
    """步骤 2：检测被动语态并恢复语义论元。

    关键认识：被动语态下，UD 表层关系
    （nsubj:pass、obl:agent）不直接对应三元组字段
    （subject=施事、object=受事）。需映射：

    被动映射：
      - UD nsubj:pass（表层主语） -> 语义受事 -> 三元组宾语
      - UD obl:agent（by 短语）   -> 语义施事 -> 三元组主语

    主动映射：
      - UD nsubj（主语）  -> 语义施事 -> 三元组主语
      - UD obj（宾语）    -> 语义受事 -> 三元组宾语

    本步骤产出用于校验大语言模型三元组的
    UD 推导主语与宾语词元（作为真值）。

    参数：
        tree: 依存树。
        predicate_idx: 谓词的 1 基索引。

    返回：
        (is_passive, ud_subject, ud_object) 元组。
        - is_passive: 是否检测到被动语态。
        - ud_subject: 语义施事（行为者）。
          - 主动：nsubj 词元。
          - 被动：obl:agent 词元（无施事被动时可为 None）。
        - ud_object: 语义受事（被作用者）。
          - 主动：obj 词元。
          - 被动：nsubj:pass 词元。
    """
    is_passive = PassiveDetector.is_passive(tree, predicate_idx)

    if is_passive:
        logger.debug("Passive voice detected at predicate index %d", predicate_idx)
        # 被动：语义施事为 obl:agent，受事为 nsubj:pass。
        agent, patient = PassiveDetector.restore_passive_args(tree, predicate_idx)
        # 返回：ud_subject = 施事，ud_object = 受事。
        return (True, agent, patient)
    else:
        logger.debug("Active voice at predicate index %d", predicate_idx)
        # 主动：语义施事 = nsubj，受事 = obj。
        agent, patient = PassiveDetector.get_active_args(tree, predicate_idx)
        return (False, agent, patient)
