"""
LexSpec 语言学约束模块
=====================================
LexSpec 项目的核心 —— 以通用依存句法（UD）约束作为大语言模型
抽取法律三元组的事后校验器。

本包提供完整的语言学分析流水线：
  1. StanzaParser:      封装 Stanza，对法律文本进行 UD 依存解析。
  2. UDFeatureExtractor: 从 UD 树中提取映射到法律三元组字段
                         （主语、宾语、条件）的句法特征。
  3. PassiveDetector:   检测被动语态，从表层句法恢复语义论元
                         映射（施事/受事）。
  4. ConditionExtractor: 利用 advcl+mark 模式提取条件从句边界，
                         并按法律领域分类体系分类。
  5. PolarityDetector:  检测情态助动词与否定，分类法律角色
                         （义务方、权利方等）。
  6. ConstraintValidator: 核心算法 —— 七步校验器，将大语言模型
                         三元组与 UD 句法结构比对，产出已校验、
                         已修正或需 Reflexion 标记的输出。

设计原则：
  所有模块消费来自 src.extraction.schema 的 DependencyTree 对象。
  本包外部不暴露原始 Stanza 对象。此隔离使解析后端可替换，
  而不影响任何下游代码。

UD 理论基础：
  - de Marneffe & Manning (2014). Stanford Typed Dependencies Manual.
  - Nivre et al. (2020). Universal Dependencies v2 Guidelines.
  - Tesnière (1959). Éléments de syntaxe structurale.

使用示例::

    from src.linguistic import StanzaParser, ConstraintValidator
    parser = StanzaParser()
    validator = ConstraintValidator(parser=parser)
    tree = parser.parse("Seller shall deliver the Goods within 30 days.")
    result = validator.validate(triplet, text, tree)
"""

from src.linguistic.stanza_parser import StanzaParser
from src.linguistic.ud_features import UDFeatureExtractor
from src.linguistic.passive_detector import PassiveDetector
from src.linguistic.condition_extractor import ConditionExtractor
from src.linguistic.polarity_detector import PolarityDetector
from src.linguistic.validator import ConstraintValidator

__all__ = [
    "StanzaParser",
    "UDFeatureExtractor",
    "PassiveDetector",
    "ConditionExtractor",
    "PolarityDetector",
    "ConstraintValidator",
]
