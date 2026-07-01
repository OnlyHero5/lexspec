"""
LexSpec 日志配置
================================
集中式日志设置，含控制台与文件处理器。

所有模块应使用 `get_logger(__name__)` 获取预配置格式的日志器。
确保流水线各阶段（extraction、linguistic、correction、annotation、
evaluation）的日志输出格式一致且便于解析。

任意模块中的用法:
    from src.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Processing clause %s", clause_id)
    logger.debug("UD parse: %s", tree.text)

应用启动时应调用一次 setup_logging:
    from src.utils.logging import setup_logging
    setup_logging(log_dir="outputs", level=logging.DEBUG)

控制台处理器显示 INFO 及以上消息，格式简洁便于快速浏览。
文件处理器捕获 DEBUG 及以上消息，含完整时间戳与源码位置，供事后调试。
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# 模块级标志，跟踪 setup_logging 是否已调用（幂等性）。
_setup_called: bool = False

# 最近一次 setup 调用的时间戳——用于日志文件名。
_setup_timestamp: str = ""

# ------------------------------------------------------------------
# 日志格式字符串
# ------------------------------------------------------------------

# 控制台格式：简短、无时间戳——面向实时监控。
# 仅显示级别、模块名与消息，便于扫读输出。
_CONSOLE_FORMAT = "[%(levelname)s] %(name)s: %(message)s"

# 文件格式：含时间戳、级别、模块与行号的完整详情——
# 面向事后调试与基于 grep 的分析。
_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"

# 文件日志时间戳的日期格式（ISO 8601，无微秒，
# 批量流水线日志秒级精度已足够）。
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_dir: str | Path = "outputs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    使用控制台与文件处理器配置根日志器。

    设置两个处理器：
      1. 控制台（stderr）：最低级别 INFO，简短格式供实时使用。
      2. 文件：              最低级别 DEBUG，含时间戳的完整格式。
         日志文件位于 log_dir 内，命名为 lexspec_YYYY-MM-DD.log。

    本函数具有幂等性——多次调用不会添加重复处理器。
    要更改日志级别，以不同的 `level` 参数再次调用；现有处理器将更新。

    参数:
        log_dir: 日志文件目录。不存在则创建。
                 默认为 "outputs"（相对于工作目录）。
        level:   控制台处理器的日志级别。默认为 logging.INFO。
                 文件处理器始终使用 logging.DEBUG 以捕获完整详情供事后分析。

    返回:
        已附加处理器的根日志器（logging.root）。

    副作用:
        - 若 log_dir 不存在则创建。
        - 向根日志器添加 StreamHandler 与 FileHandler。
        - 设置全局标志以防止后续调用重复注册处理器。
    """
    global _setup_called, _setup_timestamp

    log_dir = Path(log_dir)
    # 创建文件处理器前先确保日志目录存在。
    # parents=True 与 exist_ok=True 可安全重复调用。
    log_dir.mkdir(parents=True, exist_ok=True)

    # 生成带时间戳的文件名，使每天拥有独立日志文件。
    # 防止日志无限增长，便于定位特定运行日期的日志。
    today = datetime.now().strftime("%Y-%m-%d")
    log_filename = f"lexspec_{today}.log"
    log_path = log_dir / log_filename
    _setup_timestamp = today

    root_logger = logging.getLogger()
    # 将根日志器级别设为要捕获的最低级别（DEBUG），
    # 避免在日志器级别过滤消息。各处理器应用自己的级别过滤器。
    root_logger.setLevel(logging.DEBUG)

    # ---- 幂等性检查 ----
    # 若 setup 已调用过，更新处理器级别而非添加重复处理器。
    # 允许运行中调整详细程度而不产生冗余日志输出。
    if _setup_called:
        for handler in root_logger.handlers:
            handler.setLevel(level)
        return root_logger

    # ---- 控制台处理器（stderr）----
    # 写入 stderr 使日志输出与 stdout 分离，stdout 通常用于
    # 流水线数据输出（JSONL、报告等）。防止日志消息污染数据流。
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)  # 尊重调用方选择的级别
    console_formatter = logging.Formatter(_CONSOLE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # ---- 文件处理器 ----
    # 文件处理器始终捕获 DEBUG 及以上，无论控制台级别如何。
    # 即使控制台设为 WARNING 或 ERROR 以保持安静，也确保完整审计轨迹。
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    _setup_called = True

    # 记录初始化本身——作为日志文件中的会话开始标记。
    root_logger.info("LexSpec logging initialized — log file: %s", log_path)
    root_logger.debug("Console log level: %s", logging.getLevelName(level))
    root_logger.debug("File log level: DEBUG (always)")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    获取继承根日志器配置的模块专用日志器。

    所有 LexSpec 模块的标准入口。用法：
        logger = get_logger(__name__)
        logger.info("Extracting triplets from %d clauses", len(clauses))

    返回的日志器继承 setup_logging 配置的处理器与格式。
    模块侧无需额外配置。

    若尚未调用 setup_logging，则发出警告并临时附加基本控制台处理器，
    确保日志消息不丢失。

    参数:
        name: 日志器名称，惯例为调用模块的 __name__。

    返回:
        可直接使用的 logging.Logger 实例。
    """
    logger = logging.getLogger(name)

    # 若从未调用 setup_logging，至少确保有基本处理器，
    # 避免日志消息被静默丢弃。防止模块在应用启动例程之前
    # 被导入时常见的「日志不显示」问题。
    if not _setup_called and not logger.handlers:
        # 附加最小 stderr 处理器作为回退，级别 WARNING 以便可见但不喧宾夺主。
        fallback_handler = logging.StreamHandler(sys.stderr)
        fallback_handler.setLevel(logging.WARNING)
        fallback_formatter = logging.Formatter(_CONSOLE_FORMAT)
        fallback_handler.setFormatter(fallback_formatter)
        logger.addHandler(fallback_handler)
        logger.warning(
            "setup_logging() has not been called — using fallback WARNING handler. "
            "Call setup_logging() at application startup for full logging configuration."
        )

    return logger
