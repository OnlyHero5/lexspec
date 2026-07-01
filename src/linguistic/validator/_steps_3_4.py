"""
校验步骤 3 与 4：主语与宾语校验。

步骤 3：根据 UD 推导的语义施事校验主语字段。
步骤 4：根据 UD 推导的语义受事校验宾语字段。
"""

from __future__ import annotations

from typing import Optional, List

from src.extraction.schema import Token, FieldCorrection
from src.linguistic.ud_features import UDFeatureExtractor
from src.linguistic.text_utils import match_text, normalize_text
from src.utils.logging import get_logger

logger = get_logger(__name__)


def step3_validate_subject(
    llm_subject: str,
    ud_subject: Optional[Token],
    corrections: List[FieldCorrection],
    feedback_parts: List[str],
) -> bool:
    """步骤 3：校验主语字段。

    将大语言模型抽取的 subject.text 与 UD 推导的
    语义施事词元比对。

    匹配准则（按优先级）：
      1. 规范化后精确匹配："Seller" == "Seller"。
      2. 子串匹配："the Seller" 包含 "Seller"。
      3. 词元重叠：两字符串共享关键实词。
      4. 并列匹配：UD 主语为并列项，大语言模型
         抽取了完整并列或单一并列项。

    若大语言模型主语与 UD 主语不匹配：
      - UD 有明确主语 -> 添加 FieldCorrection（CORRECTED 路径）。
      - UD 无主语（无施事被动、缺失 nsubj）->
        添加反馈并触发 REFLEXION。

    参数：
        llm_subject: 大语言模型抽取的主语文本。
        ud_subject: UD 推导的语义施事词元（或 None）。
        corrections: 追加 FieldCorrection 的列表。
        feedback_parts: 追加反馈字符串的列表。

    返回：
        主语有效（无需修正）时返回 True。
    """
    # 情况 A：大语言模型抽取了主语但 UD 未找到。
    # 可能原因：
    #   - 无施事被动：大语言模型从上下文推断施事。
    #     这实际是大语言模型的良好行为，但我们标记
    #     供人工复核，因 UD 解析无法确认。
    #   - 大语言模型幻觉出主语。
    if ud_subject is None:
        if llm_subject and llm_subject.strip():
            # 大语言模型在 UD 无主语处抽取了主语。
            feedback_parts.append(
                f"Subject '{llm_subject}' was extracted by the LLM, but "
                f"no syntactic subject was found in the UD parse. This may "
                f"be an agentless passive where the LLM correctly inferred "
                f"the agent from discourse context. Manual review recommended."
            )
            # 不应用修正（无法确定正误）。
            return True  # 视为「有效」，因无法证伪。
        else:
            # 大语言模型与 UD 一致：无主语。
            return True

    # 情况 B：UD 找到主语但大语言模型未抽取。
    if not llm_subject or not llm_subject.strip():
        feedback_parts.append(
            f"Subject is missing from the LLM extraction but UD parse "
            f"identifies '{ud_subject.text}' as the semantic agent "
            f"(deprel={ud_subject.deprel})."
        )
        corrections.append(FieldCorrection(
            field="subject.text",
            original="",
            corrected=ud_subject.text,
            reason=(
                f"UD parse identifies '{ud_subject.text}' as the "
                f"semantic agent via {ud_subject.deprel} relation. "
                f"The LLM omitted the subject entirely."
            ),
        ))
        return False

    # 情况 C：双方均有主语 —— 比对。
    if match_text(llm_subject, ud_subject):
        return True  # 匹配。

    # 主语不匹配。
    # 若大语言模型主语包含 UD 主语文本作为子串，
    # 可能为并列扩展 —— 可接受。
    if normalize_text(ud_subject.text) in normalize_text(llm_subject):
        logger.debug(
            "LLM subject '%s' contains UD subject '%s' — accepting as "
            "coordination expansion.", llm_subject, ud_subject.text,
        )
        return True

    # 不匹配：添加修正。
    corrections.append(FieldCorrection(
        field="subject.text",
        original=llm_subject,
        corrected=ud_subject.text,
        reason=(
            f"UD parse identifies '{ud_subject.text}' as the semantic "
            f"agent via {ud_subject.deprel} relation (head: predicate "
            f"at index {ud_subject.head}). The LLM extracted "
            f"'{llm_subject}' which does not match the UD evidence. "
            f"This may indicate subject-object reversal in passive voice "
            f"or the LLM extracting a modifier instead of the head noun."
        ),
    ))
    feedback_parts.append(
        f"Subject mismatch: LLM extracted '{llm_subject}' but UD parse "
        f"identifies '{ud_subject.text}' as the semantic agent "
        f"(deprel={ud_subject.deprel})."
    )
    return False


def step4_validate_object(
    llm_object: str,
    ud_object: Optional[Token],
    corrections: List[FieldCorrection],
    feedback_parts: List[str],
) -> bool:
    """步骤 4：校验宾语字段。

    将大语言模型 action.object 与 UD 推导的语义受事比对。

    使用与主语校验（步骤 3）相同的匹配逻辑：
      - 规范化后精确匹配
      - 子串/词元重叠匹配
      - 并列扩展匹配

    边界情况：
      - 不及物动词：UD 无 obj。大语言模型也无宾语时有效。
        大语言模型为不及物动词抽取宾语时，可能为幻觉
        或从介词补语抽取。
      - 被动无 nsubj:pass：受事可能未表达。
        法律文本中少见，但非人称构造中可能。

    参数：
        llm_object: 大语言模型抽取的宾语文本。
        ud_object: UD 推导的语义受客词元（或 None）。
        corrections: 追加 FieldCorrection 的列表。
        feedback_parts: 追加反馈字符串的列表。

    返回：
        宾语有效（无需修正）时返回 True。
    """
    # 情况 A：大语言模型与 UD 均无宾语。
    if ud_object is None and (not llm_object or not llm_object.strip()):
        return True

    # 情况 B：UD 有宾语但大语言模型无。
    if ud_object is not None and (not llm_object or not llm_object.strip()):
        feedback_parts.append(
            f"Object is missing from the LLM extraction but UD parse "
            f"identifies '{ud_object.text}' as the semantic patient "
            f"(deprel={ud_object.deprel})."
        )
        corrections.append(FieldCorrection(
            field="action.object",
            original="",
            corrected=ud_object.text,
            reason=(
                f"UD parse identifies '{ud_object.text}' as the direct "
                f"object via {ud_object.deprel} relation. The LLM omitted "
                f"the object entirely."
            ),
        ))
        return False

    # 情况 C：UD 无宾语但大语言模型抽取了。
    # 动词可能不及物，或大语言模型将介词补语当作宾语。
    if ud_object is None and llm_object and llm_object.strip():
        feedback_parts.append(
            f"Object '{llm_object}' was extracted by the LLM but no "
            f"direct object was found in the UD parse. The predicate "
            f"may be intransitive. If the LLM object is a prepositional "
            f"complement or indirect object, this may be semantically "
            f"valid but syntactically unconfirmed."
        )
        # 无法证伪大语言模型抽取，但予以标记。
        # 不自动修正 —— 大语言模型可能在语义上正确
        # 即使非句法 obj。
        return True

    # 情况 D：双方均有宾语 —— 比对。
    if ud_object is not None and match_text(llm_object, ud_object):
        return True

    # 不匹配：大语言模型与 UD 对宾语意见不一。
    if ud_object is not None:
        corrections.append(FieldCorrection(
            field="action.object",
            original=llm_object,
            corrected=ud_object.text,
            reason=(
                f"UD parse identifies '{ud_object.text}' as the semantic "
                f"patient via {ud_object.deprel} relation (head: predicate "
                f"at index {ud_object.head}). The LLM extracted "
                f"'{llm_object}' which does not match. This may indicate "
                f"subject-object reversal in passive voice or the LLM "
                f"extracting an oblique modifier instead of the direct object."
            ),
        ))
        feedback_parts.append(
            f"Object mismatch: LLM extracted '{llm_object}' but UD parse "
            f"identifies '{ud_object.text}' as the semantic patient "
            f"(deprel={ud_object.deprel})."
        )
        return False

    return True
