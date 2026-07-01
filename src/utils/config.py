"""
项目级配置加载器 —— 所有实验脚本和模块的统一配置入口
=====================================================

从 ``configs/model.yaml`` 加载服务器连接、模型标识、生成参数和 Stanza
流水线设置，并构建对应的客户端/解析器实例。

配置缺失或格式错误时直接抛出异常，禁止静默回退到默认值。

使用示例::

    from src.utils.config import (
        加载模型配置,
        构建实验客户端,
        构建标注客户端,
        构建Stanza解析器,
        获取Reflexion参数,
    )

    config = 加载模型配置("configs/model.yaml")
    client = 构建实验客户端(config)
    parser = 构建Stanza解析器(config)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import yaml

from src.extraction.client import LLMClient, ClientConfig
from src.linguistic.stanza_parser import StanzaParser
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# 配置加载
# ======================================================================


def 加载模型配置(config_path: str = "configs/model.yaml") -> Dict[str, Any]:
    """加载 model.yaml 配置文件。

    参数:
        config_path: 配置文件路径。

    返回:
        配置字典。

    异常:
        FileNotFoundError: 文件不存在。
        yaml.YAMLError: YAML 解析失败。
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"模型配置文件未找到: '{config_path}'。请确认 configs/model.yaml 存在。"
        )
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if config is None:
        raise ValueError(f"配置文件 '{config_path}' 为空。")
    return config


# ======================================================================
# LLM 客户端构建
# ======================================================================


def 构建实验客户端(
    config: Dict[str, Any],
    config_path: str = "configs/model.yaml",
) -> LLMClient:
    """从配置字典构建实验模型（Qwen3.5 9B）的 LLM 客户端。

    读取 ``server``、``models.experiment`` 和 ``generation`` 段。

    参数:
        config: 已加载的配置字典（必须由 加载模型配置 返回）。
        config_path: 配置文件路径（仅用于错误消息）。

    返回:
        配置就绪的 LLMClient 实例。

    异常:
        KeyError: 缺少必需的配置键。
    """
    server = config.get("server", {})
    models = config.get("models", {})
    experiment_cfg = models.get("experiment", {})
    gen = config.get("generation", {})

    base_url = server.get("base_url")
    if not base_url:
        raise KeyError(f"配置 '{config_path}' 中缺少 server.base_url。")

    model_name = experiment_cfg.get("name")
    if not model_name:
        raise KeyError(f"配置 '{config_path}' 中缺少 models.experiment.name。")

    client_config = ClientConfig(
        base_url=base_url,
        api_key=server.get("api_key", "not-needed"),
        model=model_name,
        temperature=gen.get("temperature", 0.0),
        max_tokens=gen.get("max_tokens", 8192),
        seed=gen.get("seed", 42),
        timeout=server.get("timeout", 1200),
        max_retries=server.get("max_retries", 3),
    )

    logger.info(
        "实验客户端已配置: model=%s, base_url=%s, temperature=%.1f, "
        "max_tokens=%d, seed=%d, timeout=%ds",
        client_config.model, client_config.base_url, client_config.temperature,
        client_config.max_tokens, client_config.seed, client_config.timeout,
    )
    return LLMClient(client_config)


def 构建标注客户端(
    config: Dict[str, Any],
    model_role: str = "secondary",
    config_path: str = "configs/model.yaml",
) -> LLMClient:
    """从配置字典构建标注模型的 LLM 客户端。

    参数:
        config: 已加载的配置字典。
        model_role: 标注模型角色 —— "primary" 或 "secondary"。
        config_path: 配置文件路径（仅用于错误消息）。

    返回:
        配置就绪的 LLMClient 实例。

    异常:
        ValueError: model_role 无效。
        KeyError: 缺少必需的配置键。
    """
    if model_role not in ("primary", "secondary"):
        raise ValueError(
            f"无效的标注模型角色: '{model_role}'，应为 'primary' 或 'secondary'"
        )

    server = config.get("server", {})
    gen = config.get("generation", {})
    ann = config.get("models", {}).get("annotation", {})

    model_cfg = ann.get(model_role, {})
    model_name = model_cfg.get("name")
    if not model_name:
        raise KeyError(
            f"配置 '{config_path}' 中缺少 models.annotation.{model_role}.name。"
        )

    base_url = server.get("base_url")
    if not base_url:
        raise KeyError(f"配置 '{config_path}' 中缺少 server.base_url。")

    client_config = ClientConfig(
        base_url=base_url,
        api_key=server.get("api_key", "not-needed"),
        model=model_name,
        temperature=gen.get("temperature", 0.0),
        max_tokens=gen.get("annotation_max_tokens", 8192),
        seed=gen.get("seed", 42),
        timeout=server.get("timeout", 1200),
        max_retries=server.get("max_retries", 3),
    )

    logger.info(
        "标注客户端已配置: role=%s model=%s url=%s timeout=%ds",
        model_role, model_name, client_config.base_url, client_config.timeout,
    )
    return LLMClient(client_config)


# ======================================================================
# Stanza 解析器构建
# ======================================================================


def 构建Stanza解析器(
    config: Dict[str, Any],
    config_path: str = "configs/model.yaml",
) -> StanzaParser:
    """从配置字典构建 Stanza 依存句法解析器。

    参数:
        config: 已加载的配置字典。
        config_path: 配置文件路径（仅用于错误消息）。

    返回:
        初始化完成的 StanzaParser 实例。
    """
    stanza_cfg = config.get("stanza", {})

    parser = StanzaParser(
        lang=stanza_cfg.get("lang", "en"),
        processors=stanza_cfg.get("processors", "tokenize,mwt,pos,lemma,depparse"),
        download_method=stanza_cfg.get("download_method", "REUSE_RESOURCES"),
    )

    logger.info(
        "Stanza 解析器已初始化: lang=%s, processors=%s",
        stanza_cfg.get("lang", "en"),
        stanza_cfg.get("processors", "tokenize,mwt,pos,lemma,depparse"),
    )
    return parser


# ======================================================================
# Reflexion 参数
# ======================================================================


def 获取Reflexion参数(
    config: Dict[str, Any],
    config_path: str = "configs/model.yaml",
) -> Dict[str, Any]:
    """从配置字典读取 Reflexion 自我修正参数。

    参数:
        config: 已加载的配置字典。
        config_path: 配置文件路径（仅用于错误消息）。

    返回:
        包含 ``max_iterations`` 和 ``temperature`` 的字典。
    """
    reflexion_cfg = config.get("reflexion", {})
    return {
        "max_iterations": reflexion_cfg.get("max_iterations", 1),
        "temperature": reflexion_cfg.get("temperature", 0.0),
    }


# ======================================================================
# 约束配置加载
# ======================================================================


def 加载约束配置(config_path: str = "configs/constraints.yaml") -> Dict[str, Any]:
    """加载 ``constraints.yaml`` 并返回完整配置字典。

    参数:
        config_path: 约束配置文件路径。

    返回:
        含 F1 权重、校验阈值、规范化规则、现象配额等全部配置节的字典。

    异常:
        FileNotFoundError: 配置文件不存在。
        ValueError: 文件为空。
    """
    from src.utils.constraints import load_constraints_config

    return load_constraints_config(config_path)
