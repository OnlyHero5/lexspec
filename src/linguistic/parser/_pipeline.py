"""
Pipeline initialization for StanzaParser.

Contains the __init__ logic, nlp property, and is_available method.
"""

from __future__ import annotations

from typing import Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton — Stanza pipeline is expensive to initialize
# (model loading, GPU allocation). We create it once and reuse.
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
    """Initialize the Stanza pipeline.

    On first instantiation, downloads models if needed and loads the
    pipeline. Subsequent instantiations reuse the module-level singleton
    regardless of constructor arguments (the first configuration wins).

    Args:
        model_path: Optional custom path to Stanza model directory.
                    If None, uses Stanza's default model directory.
        lang: Language code (default: 'en' for English).
              Legal contracts are primarily in English; this default
              reflects that domain assumption.
        processors: Comma-separated list of Stanza processors.
                    Default includes all processors needed for UD analysis:
                    tokenize, MWT expansion, POS tagging, lemmatization,
                    and dependency parsing.
        download_method: Stanza download strategy.
                         REUSE_RESOURCES avoids re-downloading models
                         if they already exist on disk.
        use_gpu: Whether to use GPU acceleration if available.
                 Set to False for CPU-only environments.
    """
    global _nlp, _nlp_initialized

    # --- Singleton pattern: only initialize once ---
    # Stanza models are large (~500MB for English) and loading them
    # is expensive. We maintain a single pipeline instance for the
    # entire process lifetime.
    if _nlp_initialized:
        logger.debug("Reusing existing Stanza pipeline singleton")
        return

    logger.info(
        "Initializing Stanza pipeline: lang=%s, processors=%s, use_gpu=%s",
        lang, processors, use_gpu,
    )

    try:
        import stanza

        # Ensure models are downloaded before constructing the pipeline.
        # REUSE_RESOURCES skips download if models already exist.
        # This is critical for CI/CD and offline environments.
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

        # Construct the pipeline with all required processors.
        # depparse uses the 'depparse' processor which produces UD v2
        # annotations (head, deprel, feats) in CoNLL-U format.
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
    """Return the module-level Stanza pipeline singleton.

    Provides access for advanced use cases (e.g., inspecting raw Stanza
    output). Most users should use parse() and parse_batch() instead.
    """
    global _nlp
    if _nlp is None:
        raise RuntimeError(
            "Stanza pipeline not initialized. Create a StanzaParser instance first."
        )
    return _nlp


def _is_available(self=None) -> bool:
    """Check whether the Stanza pipeline is initialized and ready.

    Returns:
        True if the pipeline singleton is available for parsing.
    """
    global _nlp_initialized
    return _nlp_initialized
