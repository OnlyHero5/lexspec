"""
StanzaParser 的流水线初始化。

包含 __init__ 逻辑、nlp 属性及 is_available 方法。
"""

from __future__ import annotations

from typing import Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 模块级单例 —— Stanza 流水线初始化代价高
#（模型加载、GPU 分配）。创建一次后复用。
# ---------------------------------------------------------------------------
_nlp = None
_nlp_initialized: bool = False


def _init_pipeline(
    model_path: Optional[str] = None,
    lang: str = "en",
    processors: str = "tokenize,mwt,pos,lemma,depparse",
    download_method: str = "REUSE_RESOURCES",
    use_gpu: bool = True,
):
    """初始化 Stanza 流水线。

    首次实例化时，按需下载模型并加载流水线。后续实例化
    复用模块级单例，与构造参数无关（以首次配置为准）。

    参数：
        model_path: 可选的 Stanza 模型目录自定义路径。
                    为 None 时使用 Stanza 默认模型目录。
        lang: 语言代码（默认 'en' 表示英语）。
              法律合同主要为英文；此默认值反映该领域假设。
        processors: 逗号分隔的 Stanza 处理器列表。
                    默认包含 UD 分析所需的全部处理器：
                    分词、MWT 展开、词性标注、词形还原、依存解析。
        download_method: Stanza 下载策略。
                         REUSE_RESOURCES 在模型已存在时避免重复下载。
        use_gpu: 是否使用 GPU 加速（若可用）。
                 纯 CPU 环境请设为 False。
    """
    global _nlp, _nlp_initialized

    # --- 单例模式：仅初始化一次 ---
    # Stanza 模型体积大（英语约 500MB），加载代价高。
    # 整个进程生命周期内维持单一流水线实例。
    if _nlp_initialized:
        logger.debug("Reusing existing Stanza pipeline singleton")
        return

    logger.info(
        "Initializing Stanza pipeline: lang=%s, processors=%s, use_gpu=%s",
        lang, processors, use_gpu,
    )

    try:
        import stanza

        # 构造流水线前确保模型已下载。
        # REUSE_RESOURCES 在模型已存在时跳过下载。
        # 对 CI/CD 与离线环境至关重要。
        download_kwargs = {"logging_level": "WARNING"}
        pipeline_kwargs = {
            "lang": lang,
            "processors": processors,
            "download_method": download_method,
            "use_gpu": use_gpu,
            "logging_level": "WARNING",
        }
        if model_path:
            download_kwargs["model_dir"] = model_path
            pipeline_kwargs["model_dir"] = model_path

        stanza.download(lang, **download_kwargs)
        logger.info("Stanza models confirmed for language: %s", lang)

        # 使用全部所需处理器构造流水线。
        # depparse 使用 'depparse' 处理器，产出 UD v2
        # 标注（head、dprel、feats），格式为 CoNLL-U。
        _nlp = stanza.Pipeline(**pipeline_kwargs)
        _nlp_initialized = True
        logger.info("Stanza pipeline initialized successfully")

    except ImportError:
        raise ImportError(
            "Stanza is required for UD dependency parsing. "
            "Install it with: pip install stanza"
        )
    except Exception as e:
        logger.error("Failed to initialize Stanza pipeline: %s", e)
        raise RuntimeError(f"Stanza initialization failed: {e}")


def _get_nlp(self=None):
    """返回模块级 Stanza 流水线单例。

    供高级用例访问（如检查原始 Stanza 输出）。
    大多数用户应使用 parse() 与 parse_batch()。
    """
    global _nlp
    if _nlp is None:
        raise RuntimeError(
            "Stanza pipeline not initialized. Create a StanzaParser instance first."
        )
    return _nlp


def _is_available(self=None) -> bool:
    """检查 Stanza 流水线是否已初始化并就绪。

    返回：
        流水线单例可用于解析时返回 True。
    """
    global _nlp_initialized
    return _nlp_initialized
