"""
Plain JSON and Filesystem Utilities for LexSpec
================================================
Single-object JSON reading/writing and directory management.

This module handles plain (non-line-delimited) JSON files — typically used
for configuration, metadata, and small structured data files. It also
includes the ensure_dir filesystem utility used across the project.

Design decisions:
  - All paths accept str | Path, but internally use pathlib.Path.
  - JSON read/write uses ensure_ascii=False to preserve Unicode.
  - Write operations create parent directories automatically.
  - Read operations raise FileNotFoundError with clear messages.
"""

import json
from pathlib import Path

# Default encoding used for all file I/O
_ENCODING = "utf-8"


# ============================================================================
# Plain JSON — For configuration, metadata, and small data files
# ============================================================================


def read_json(file_path: str | Path) -> dict:
    """
    Read a single JSON object from a file.

    The file must contain a single JSON object (not JSONL / newline-delimited).
    Use this for configuration files, metadata, and other small structured data.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Parsed dict from the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file does not contain valid JSON.
    """
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, "r", encoding=_ENCODING) as fh:
        return json.load(fh)


def write_json(file_path: str | Path, data: dict, indent: int = 2) -> None:
    """
    Write a dict to a JSON file with pretty-printing.

    Creates parent directories if needed. Uses ensure_ascii=False to
    preserve Unicode characters in legal text (e.g., section symbols,
    non-English party names). The default indent of 2 spaces makes the
    output human-readable.

    Args:
        file_path: Path for the output JSON file.
        data:      Dict to serialize (must be JSON-serializable).
        indent:    Number of spaces for indentation (default 2). Set to
                   None for compact single-line output.

    Raises:
        OSError: If the file cannot be written.
        TypeError: If data contains non-JSON-serializable values.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding=_ENCODING) as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)


# ============================================================================
# Filesystem Utilities
# ============================================================================


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists, creating it (and all parent directories)
    if necessary.

    Idempotent — calling this on an already-existing directory is a no-op
    (no error, no modification). This is a thin but intentional wrapper
    around Path.mkdir to provide a consistent interface for all modules.

    Args:
        path: Directory path as string or Path.

    Returns:
        Path object pointing to the ensured directory.

    Raises:
        OSError: If the directory cannot be created (e.g., permission denied,
                 or a file already exists at the given path).
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
