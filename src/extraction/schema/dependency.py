"""
LexSpec 数据模型 —— 语言学模型
=============================

Universal Dependencies 依存句法树的结构化表示。
提供 Token, ClauseSpan, ConditionSpan 三个核心数据类。
DependencyTree 已移至 ``._dependency_tree`` 模块。
"""

from typing import Optional, List, Dict

from pydantic import BaseModel, Field

from .enums import ConditionType


class Token(BaseModel):
    """UD 解析中的单个词元（对应 CoNLL-U 表的一行）。

    字段:
        index: 句子内 1 基词元索引。
        text: 表层词形（如 ``delivered``）。
        lemma: 词典词元（如 ``deliver``）。
        upos: 通用词性（``VERB``、``NOUN`` 等）。
        xpos: 语言特定词性；不可用时为空。
        deprel: UD 依存关系（``nsubj``、``obj``、``advcl`` 等）。
        head: 中心词索引（1 基）；``0`` 表示该词元为句法根。
        feats: 形态特征字典（如 ``{"Tense": "Past"}``）。
    """
    index: int = Field(
        description="句子中的 1 基词元索引"
    )
    text: str = Field(
        description="词元在句子中的表层形式"
    )
    lemma: str = Field(
        description="规范词元形式，如 'was' 对应 'be'，'delivered' 对应 'deliver'"
    )
    upos: str = Field(
        description="通用词性标签，如 'VERB'、'NOUN'、'ADP'、'CCONJ'"
    )
    xpos: str = Field(
        default="",
        description="语言特定词性标签（树库特定）；不可用时为空"
    )
    deprel: str = Field(
        description="通用依存关系标签，如 'nsubj'、'obj'、'advcl'、'mark'"
    )
    head: int = Field(
        description="中心词词元索引（1 基）；0 表示该词元为句法根"
    )
    feats: Dict[str, str] = Field(
        default_factory=dict,
        description="形态特征键值对，如 {'Tense': 'Past', 'Voice': 'Pass'}"
    )


class ClauseSpan(BaseModel):
    """句法子句的词元跨度（条件从句、关系从句等）。

    字段:
        tokens: 属于该跨度的有序 1 基词元索引列表。
        text: 跨度表层文本（词元以空格连接）。
        deprel: 跨度中心词对其支配词的依存关系。
        mark_token: 引入子句的标记词（``if``、``unless`` 等）；无则为 None。
    """
    tokens: List[int] = Field(
        description="属于该跨度的有序 1 基词元索引列表"
    )
    text: str = Field(
        description="跨度的表层文本——词元以空格连接"
    )
    deprel: str = Field(
        default="",
        description="跨度中心词对其支配词的依存关系"
    )
    mark_token: Optional[Token] = Field(
        default=None,
        description="引入该子句的标记词（如 'if'、'unless'、'that'），若存在"
    )


class ConditionSpan(ClauseSpan):
    """
    带语义类型分类的条件子句跨度。

    扩展 ClauseSpan，根据标记词与句法上下文添加 condition_type
    （如 advcl 且 mark='if' -> TRIGGER 等）。
    """
    condition_type: ConditionType = Field(
        default=ConditionType.NONE,
        description="根据标记词与句法结构分类的条件类型"
    )
    mark_text: str = Field(
        default="",
        description="标记词的表层文本，如 'if'、'unless'、'after'"
    )
