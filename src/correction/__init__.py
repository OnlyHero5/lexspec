"""
LexSpec 纠错（Reflexion）包
===========================
当 UD 约束校验器检测到结构错误并返回 REFLEXION_REQUIRED 时，
本自纠错模块利用结构化错误反馈提示 LLM 重新抽取法律三元组。

全部提示模板与错误提示从 configs/prompts.yaml 加载。
距离阈值从 configs/constraints.yaml 加载。
"""

from src.correction.reflexion import ReflexionGenerator

__all__ = [
    "ReflexionGenerator",
]
