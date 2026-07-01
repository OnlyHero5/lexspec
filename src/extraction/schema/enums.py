"""
LexSpec 数据模型 —— 枚举类型定义
=================================

项目所有模块共享的受控词汇表。
"""

from enum import Enum


class LegalRole(str, Enum):
    """
    合同条款中主体当事方的法律角色。

    由情态（shall/may/must not/agrees to）与句法结构
    （主动/被动、主语位置）共同决定。
    供 UD 校验器修正大语言模型分配的角色。
    """
    OBLIGOR = "obligor"
    RIGHT_HOLDER = "right_holder"
    PROHIBITED_PARTY = "prohibited_party"
    INDEMNIFYING_PARTY = "indemnifying_party"
    OTHER = "other"


class ConditionType(str, Enum):
    """
    附加于法律动作的条件子句的语义类型。

    - TEMPORAL:  时间条件（如 "within 30 days"、"after Closing"）
    - TRIGGER:   事件前提（如 "if Buyer defaults"、"upon delivery"）
    - EXCEPTION: 义务的例外情形（如 "unless waived"、"except as provided"）
    - NONE:      子句中无条件
    """
    TEMPORAL = "temporal"
    TRIGGER = "trigger"
    EXCEPTION = "exception"
    NONE = "none"


class ValidationStatus(str, Enum):
    """
    基于 UD 的约束校验器对大语言模型预测的校验结果。

    - VALID:              预测与 UD 证据一致；无需修正。
    - CORRECTED:          预测有轻微错误且已自动修正。
    - REFLEXION_REQUIRED: 预测有结构性错误，需大语言模型重新抽取
                          （用作迭代 Reflexion 循环中的反馈）。
    """
    VALID = "VALID"
    CORRECTED = "CORRECTED"
    REFLEXION_REQUIRED = "REFLEXION_REQUIRED"


class ErrorCategory(str, Enum):
    """
    主要错误类别——导致抽取错误的语言学现象。

    对应具体 UD 句法模式：
    - PASSIVE_VOICE:            使用 nsubj:pass 而非 nsubj；宾语位于主语位置
    - CONDITIONAL_BOUNDARY:     advcl/mark 范围被大语言模型误识别
    - RELATIVE_CLAUSE:          acl:relcl 嵌套使抽取器混淆
    - LONG_DISTANCE_DEPENDENCY: 谓词与论元间依存路径超过 3 条边
    - NEGATION_EXCEPTION:       否定词或 "except/unless" 改变了角色
    - OTHER_ERROR:              不属于以上类别的兜底类别
    """
    PASSIVE_VOICE = "passive_voice"
    CONDITIONAL_BOUNDARY = "conditional_boundary"
    RELATIVE_CLAUSE = "relative_clause"
    LONG_DISTANCE_DEPENDENCY = "long_distance_dependency"
    NEGATION_EXCEPTION = "negation_exception"
    OTHER_ERROR = "other"


class FieldErrorType(str, Enum):
    """
    次要错误类型——LegalTriplet 的哪些字段受到影响。

    用于按字段与语言学现象交叉制表错误率。
    """
    SUBJECT = "subject"
    ROLE = "role"
    PREDICATE = "predicate"
    OBJECT = "object"
    CONDITION_OMISSION = "condition_omission"
    CONDITION_OVEREXTENSION = "condition_overextension"
