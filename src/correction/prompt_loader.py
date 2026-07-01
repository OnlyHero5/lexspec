"""
Reflexion 提示词加载器 — 从统一提示词加载器重新导出。
"""

from src.utils.prompt_loader import load_extraction_prompts, load_reflexion_config


def load_reflexion_prompt(prompts_path: str) -> str:
    """从 YAML 加载 Reflexion 反馈模板。"""
    feedback, _, _ = load_reflexion_config(prompts_path)
    return feedback


def load_system_prompt(prompts_path: str) -> str:
    """加载 Reflexion 重新生成用的抽取系统提示词。"""
    return load_extraction_prompts(prompts_path)["system"]
