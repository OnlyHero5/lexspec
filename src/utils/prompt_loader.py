"""
提示词加载器 —— LLMAnnotator 的统一提示词管理
============================================

从 ``configs/prompts.yaml`` 加载提示词模板。配置缺失或格式错误时直接
抛出异常，禁止静默回退到硬编码备用提示词。

使用示例::

    from src.utils.prompt_loader import 提示词加载器

    loader = 提示词加载器("configs/prompts.yaml")
    system, user = loader.加载标注提示词()
    system, user = loader.加载审查提示词()
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


class 提示词加载器:
    """从 YAML 配置文件加载提示词模板。

    所有提示词必须来自配置文件。配置缺失或字段不完整时抛出异常。

    Attributes:
        config: 解析后的 YAML 配置字典。
        source_path: 配置文件路径（用于错误消息）。
    """

    def __init__(self, prompts_path: str = "configs/prompts.yaml") -> None:
        """加载并解析提示词配置文件。

        Args:
            prompts_path: 提示词 YAML 配置文件路径。

        Raises:
            FileNotFoundError: 配置文件不存在。
            yaml.YAMLError: YAML 解析失败。
        """
        self.source_path = prompts_path
        path = Path(prompts_path)
        if not path.exists():
            raise FileNotFoundError(
                f"提示词配置文件未找到: '{prompts_path}'。请确认 configs/prompts.yaml 存在。"
            )
        with open(path, "r", encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh)
        if self.config is None:
            raise ValueError(f"提示词配置文件 '{prompts_path}' 为空。")
        logger.info("已从 %s 加载提示词配置", self.source_path)

    # ------------------------------------------------------------------
    # 标注提示词
    # ------------------------------------------------------------------

    def 加载标注提示词(self) -> Tuple[str, str]:
        """加载标注阶段的 (system_prompt, user_template)。

        Returns:
            (系统提示词, 用户提示词模板) 元组。

        Raises:
            KeyError: 缺少 annotation.system 或 annotation.user。
        """
        section = self.config.get("annotation", {})
        system = self._提取必需字符串(section, "system", "annotation.system")
        user = self._提取必需字符串(section, "user", "annotation.user")

        if "{dependency_info}" in user:
            user = user.replace("{dependency_info}", "")
            logger.debug("已从标注用户模板中移除 {dependency_info} 占位符")

        return system.strip(), user.strip()

    # ------------------------------------------------------------------
    # 审查提示词
    # ------------------------------------------------------------------

    def 加载审查提示词(self) -> Tuple[str, str]:
        """加载交叉审查阶段的 (system_prompt, user_template)。

        Returns:
            (系统提示词, 用户提示词模板) 元组。

        Raises:
            KeyError: 缺少 annotation.review.system 或 annotation.review.user。
        """
        review = self.config.get("annotation", {}).get("review", {})
        system = self._提取必需字符串(review, "system", "annotation.review.system")
        user = self._提取必需字符串(review, "user", "annotation.review.user")
        return system.strip(), user.strip()

    # ------------------------------------------------------------------
    # Reflexion 提示词
    # ------------------------------------------------------------------

    def 加载Reflexion提示词(self) -> Tuple[str, Dict[str, str]]:
        """加载 Reflexion 自我修正阶段的提示词。

        Returns:
            (feedback_template, error_hints_dict) 元组。

        Raises:
            KeyError: 缺少 reflexion.feedback_template 或 reflexion.error_hints。
        """
        reflexion = self.config.get("reflexion", {})
        feedback = self._提取必需字符串(
            reflexion, "feedback_template", "reflexion.feedback_template"
        )
        error_hints = reflexion.get("error_hints", {})
        if not isinstance(error_hints, dict) or not error_hints:
            raise KeyError(
                f"'{self.source_path}' 中缺少 'reflexion.error_hints'，"
                "或该值为空。必须为所有错误类别提供修正提示词。"
            )
        return feedback.strip(), error_hints

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _提取必需字符串(self, section: Dict, key: str, full_path: str) -> str:
        """从配置段中提取必需的字符串值，缺失或为空时抛出异常。

        Args:
            section: YAML 配置段字典。
            key: 键名。
            full_path: 完整路径（用于错误消息，如 'annotation.system'）。

        Returns:
            提取到的字符串（已去除首尾空白）。

        Raises:
            KeyError: 键缺失或值为空。
        """
        value = section.get(key)
        if isinstance(value, str) and value.strip():
            return value
        raise KeyError(
            f"'{self.source_path}' 中缺少 '{full_path}'，或该值为空。"
            "所有提示词必须来自配置文件。"
        )
