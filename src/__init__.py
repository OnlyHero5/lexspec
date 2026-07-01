"""
LexSpec — 法律合同分析的计算语言学
================================================================

LexSpec 是一个计算语言学项目，用于分析法律合同条款并抽取
(subject, action, condition) 三元组。采用基于大语言模型的抽取方法，
并通过通用依存句法（Universal Dependencies）约束进行事后校验。

流水线概览:
  - extraction:    基于大语言模型的条款三元组抽取
  - linguistic:    基于 UD 的句法分析与校验
  - correction:    基于约束的事后修正
  - annotation:    多标注者流水线与金标准构建
  - evaluation:    错误分析与性能指标

作者:  LexSpec Team
版本: 1.0.0
许可证: MIT
"""

__version__ = "1.0.0"
__author__ = "LexSpec Team"
__description__ = (
    "基于大语言模型与通用依存句法约束，从法律合同条款中抽取"
    "（主语、动作、条件）三元组的计算语言学流水线。"
)
__project_name__ = "lexspec"
