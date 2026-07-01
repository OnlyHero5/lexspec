"""
LexSpec 数据模型 —— 核心三元组模型
===================================

法律动作框架的核心数据结构：Subject, Action, Condition, LegalTriplet。
"""

from pydantic import BaseModel, Field

from .enums import LegalRole, ConditionType


class Subject(BaseModel):
    """法律动作主体 —— 履行、承担、享有义务、权利、限制或赔偿的当事方。

    字段:
        text: 当事方名称或标识字符串，如 ``"Seller"``、``"the Buyer"``。
        role: 法律角色枚举（``OBLIGOR`` 义务方、``RIGHT_HOLDER`` 权利方、
            ``PROHIBITED_PARTY`` 禁止方等），由情态与句法位置推导。

    示例:
      - ``"Seller"`` + ``OBLIGOR``:  ``"Seller shall deliver the Goods"``
      - ``"Buyer"`` + ``RIGHT_HOLDER``: ``"Buyer may terminate this Agreement"``
    """
    text: str = Field(
        description="当事方名称/标识，如 'Seller'、'the Buyer'、'each Indemnifying Party'"
    )
    role: LegalRole = Field(
        description="根据情态与句法位置推导的法律角色分类"
    )


class Action(BaseModel):
    """核心法律动作 —— 谓词（主要动词词元）及其直接宾语。

    字段:
        predicate: 主要动词的词元形式（如 ``deliver`` 而非 ``delivered``），
            便于跨时态比较。
        object: 动作所作用的对象文本（如 ``"the Goods"``、``"all Losses"``）。
    """
    predicate: str = Field(
        description="主要动词词元形式，如 'deliver' 而非 'delivered'，'indemnify' 而非 'indemnifies'"
    )
    object: str = Field(
        description="动作对象，如 'the Goods'、'the Agreement'、'all Losses'"
    )


class Condition(BaseModel):
    """条件子句 —— 限定法律动作的前提、触发事件、时间限制或例外。

    字段:
        text: 完整条件子句文本；无条件时为空字符串 ``""``。
        type: 条件语义类型（``TRIGGER`` / ``TEMPORAL`` / ``EXCEPTION`` /
            ``NONE``），由引导词与句法结构判定。
    """
    text: str = Field(
        default="",
        description="完整条件子句文本；无条件时为空字符串"
    )
    type: ConditionType = Field(
        default=ConditionType.NONE,
        description="条件子句的语义分类"
    )


class LegalTriplet(BaseModel):
    """从单条合同条款抽取的完整法律动作框架（基本分析单元）。

    字段:
        subject: 法律动作主体（谁）。
        action: 核心动作，含谓词与宾语（做什么、作用于何物）。
        condition: 限定动作生效的条件子句（在什么情况下）；可为空。

    示例:
      条款 ``"Seller shall deliver the Goods within 30 days of Closing"`` 对应::
        Subject("Seller", OBLIGOR),
        Action("deliver", "the Goods"),
        Condition("within 30 days...", TEMPORAL)
    """
    subject: Subject = Field(
        description="法律动作主体 —— 履行/接受动作的当事方"
    )
    action: Action = Field(
        description="核心法律动作 —— 谓词 + 宾语"
    )
    condition: Condition = Field(
        default_factory=Condition,
        description="限定动作的条件子句；可为空/NONE",
    )
