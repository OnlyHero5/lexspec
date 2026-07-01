"""
Logging Configuration for LexSpec
=================================
Centralized logging setup with console and file handlers.

All modules should use `get_logger(__name__)` to obtain a logger
pre-configured with consistent formatting. This ensures that log output
from all pipeline stages (extraction, linguistic, correction, annotation,
evaluation) follows the same format and can be easily parsed.

Usage in any module:
    from src.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Processing clause %s", clause_id)
    logger.debug("UD parse: %s", tree.text)

Setup should be called once at application startup:
    from src.utils.logging import setup_logging
    setup_logging(log_dir="outputs", level=logging.DEBUG)

The console handler shows INFO+ messages with minimal formatting for
at-a-glance readability. The file handler captures DEBUG+ messages with
full timestamps and source locations for post-hoc debugging.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Module-level flag to track whether setup_logging has been called.
# Used to prevent duplicate handler registration (idempotency).
_setup_called: bool = False

# Timestamp of the most recent setup call — used in the log filename.
_setup_timestamp: str = ""

# ------------------------------------------------------------------
# Log format strings
# ------------------------------------------------------------------

# Console format: brief, no timestamp — designed for real-time monitoring.
# Shows only the level, module name, and message so the output is scannable.
_CONSOLE_FORMAT = "[%(levelname)s] %(name)s: %(message)s"

# File format: full detail with timestamp, level, module, and line number —
# designed for post-mortem debugging and grep-based analysis.
_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"

# Date format for file log timestamps (ISO 8601 without microseconds,
# since second precision is sufficient for batch pipeline logs).
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_dir: str | Path = "outputs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure the root logger with console and file handlers.

    Sets up two handlers:
      1. Console (stderr): Min level = INFO, brief format for real-time use.
      2. File:              Min level = DEBUG, full format with timestamps.
         The log file is named lexspec_YYYY-MM-DD.log inside log_dir.

    This function is idempotent — calling it multiple times will not add
    duplicate handlers. To change the log level, call with a different
    `level` argument; existing handlers will be updated.

    Args:
        log_dir: Directory for log files. Created if it does not exist.
                 Defaults to "outputs" (relative to working directory).
        level:   Logging level for the console handler. Defaults to logging.INFO.
                 The file handler always uses logging.DEBUG to capture the
                 full detail for post-hoc analysis.

    Returns:
        The root logger (logging.root), with handlers attached.

    Side effects:
        - Creates log_dir if it does not exist.
        - Adds StreamHandler and FileHandler to the root logger.
        - Sets a global flag to prevent duplicate handler registration
          on subsequent calls.
    """
    global _setup_called, _setup_timestamp

    log_dir = Path(log_dir)
    # Ensure the log directory exists before creating the file handler.
    # mkdir with parents=True and exist_ok=True is safe to call repeatedly.
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate a timestamped filename so each day gets its own log file.
    # This prevents logs from growing unbounded and makes it easy to
    # locate logs for a specific run date.
    today = datetime.now().strftime("%Y-%m-%d")
    log_filename = f"lexspec_{today}.log"
    log_path = log_dir / log_filename
    _setup_timestamp = today

    root_logger = logging.getLogger()
    # Set the root logger level to the lowest level we want to capture
    # (DEBUG), so no messages are filtered at the logger level. Individual
    # handlers apply their own level filters.
    root_logger.setLevel(logging.DEBUG)

    # ---- Idempotency check ----
    # If setup has already been called, update handler levels instead of
    # adding duplicate handlers. This allows callers to adjust verbosity
    # mid-run without creating redundant log output.
    if _setup_called:
        for handler in root_logger.handlers:
            handler.setLevel(level)
        return root_logger

    # ---- Console Handler (stderr) ----
    # Writing to stderr keeps log output separate from stdout, which is
    # typically used for pipeline data output (JSONL, reports, etc.).
    # This prevents log messages from corrupting data streams.
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)  # Respect the caller's chosen level
    console_formatter = logging.Formatter(_CONSOLE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # ---- File Handler ----
    # The file handler always captures DEBUG and above, regardless of the
    # console level. This ensures a complete audit trail even when the
    # console is set to WARNING or ERROR for quieter output.
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    _setup_called = True

    # Log the initialization itself — this serves as a header in the
    # log file, marking the start of a new session.
    root_logger.info("LexSpec logging initialized — log file: %s", log_path)
    root_logger.debug("Console log level: %s", logging.getLevelName(level))
    root_logger.debug("File log level: DEBUG (always)")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a module-specific logger that inherits the root logger's configuration.

    This is the standard entry point for all LexSpec modules. Use it as:
        logger = get_logger(__name__)
        logger.info("Extracting triplets from %d clauses", len(clauses))

    The returned logger inherits handlers and formatting from the root
    logger (configured via setup_logging). No additional configuration
    is needed on the module side.

    If setup_logging has not been called yet, a warning is emitted and
    a basic console handler is temporarily attached to ensure log messages
    are not lost.

    Args:
        name: Logger name, conventionally __name__ from the calling module.

    Returns:
        A logging.Logger instance ready for use.
    """
    logger = logging.getLogger(name)

    # If setup_logging has never been called, ensure there's at least
    # a basic handler so messages are not silently dropped. This prevents
    # the common pitfall of "my logs aren't showing up" when a module
    # is imported before the application's startup routine runs.
    if not _setup_called and not logger.handlers:
        # Attach a minimal stderr handler as a fallback, set to WARNING
        # so it is unobtrusive but visible.
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
