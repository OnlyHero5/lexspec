"""
校验步骤 7：状态判定。

判定三元组为 VALID、CORRECTED 或 REFLEXION_REQUIRED。
"""

from __future__ import annotations

from typing import List

from src.extraction.schema import FieldCorrection, ValidationStatus
from src.utils.logging import get_logger

logger = get_logger(__name__)


def step7_determine_status(
    corrections: List[FieldCorrection],
) -> ValidationStatus:
    """步骤 7：确定输出状态。

    决策逻辑：

    1. 无修正 -> VALID。
       大语言模型输出与 UD 解析句法一致。
       无需更改。此为理想结果。

    2. 有修正且 UD 对全部修正字段有候选 -> CORRECTED。
       可用 UD 证据自动修正大语言模型输出。
       ValidationResult 中的 corrected_prediction 含
       自动修正后的三元组。

       CORRECTED 条件：
         - 全部主语修正均有非 None 的 ud_subject。
         - 全部宾语修正均有非 None 的 ud_object。
         （条件修正按构造总有 UD 证据，
          因仅在 ud_spans 非空时添加。）

    3. 有修正但 UD 对部分修正字段缺证据 -> REFLEXION_REQUIRED。
       大语言模型需凭语言学提示重新分析，
       因无法置信地自动修正。

       REFLEXION_REQUIRED 触发条件：
         - 无施事被动：UD 有受事无施事。
           大语言模型可能从上下文正确推断施事，
           但句法上无法验证。
         - 缺失 UD 主语：nsubj 与 obl:agent 均为 None。
         - 缺失 UD 宾语：obj 与 nsubj:pass 均为 None。
           （不及物动词或解析错误。）

    参数：
        corrections: 已识别的修正列表。

    返回：
        ValidationStatus 枚举值。
    """
    if not corrections:
        return ValidationStatus.VALID

    # 检查是否有修正针对 UD 缺证据的字段。
    # 仅当 UD 对每个修正字段都有候选时才能自动修正。
    needs_reflexion = False

    for correction in corrections:
        field = correction.field

        if field == "subject.text" and correction.corrected == "":
            # UD 无主语可作为修正。
            needs_reflexion = True
            logger.debug(
                "Reflexion needed: subject correction but UD has no subject"
            )

        if field == "action.object" and correction.corrected == "":
            # UD 无宾语可作为修正。
            needs_reflexion = True
            logger.debug(
                "Reflexion needed: object correction but UD has no object"
            )

        if field == "condition.text" and correction.corrected == "":
            # UD 未找到条件 —— 但大语言模型可能抽取了
            # Stanza 遗漏的有效条件。
            needs_reflexion = True
            logger.debug(
                "Reflexion needed: condition removal but possible parse gap"
            )

    # 附加检查：若 ud_subject 或 ud_object 为 None
    # 且有修正针对对应字段，需要 Reflexion。
    # （上文 correction.corrected 检查已捕获显式空修正，
    #  此处也捕获修正文本非空但 UD 词元本身缺失的情况 ——
    #  正常运行不应发生，属安全检查。）
    # 实际上已覆盖：若 ud_subject 为 None 且有主语修正，
    # correction 的 corrected=""，因无 UD 词元可用。

    if needs_reflexion:
        logger.info("Status: REFLEXION_REQUIRED (%d corrections with gaps)", len(corrections))
        return ValidationStatus.REFLEXION_REQUIRED

    logger.info("Status: CORRECTED (%d auto-correctable corrections)", len(corrections))
    return ValidationStatus.CORRECTED
