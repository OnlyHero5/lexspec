"""
提示词加载 —— 从 YAML 配置文件加载提示词模板（统一入口 re-export）
"""

from src.utils.prompt_loader import load_extraction_prompts as load_prompts

__all__ = ["load_prompts"]
