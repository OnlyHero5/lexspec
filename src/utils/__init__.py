"""
LexSpec Utilities Package
==========================

Shared utility modules used across the entire LexSpec project.

Public API:
  - jsonl_io: JSONL reading/writing and Pydantic model serialization.
  - json_io:  Plain JSON reading/writing and directory management.
  - io:       Backward-compatibility shim that re-exports from jsonl_io
              and json_io (existing code continues to work unchanged).
  - logging:  Centralized logging configuration with console and file
              handlers. All modules use get_logger(__name__) for
              consistent formatting.
"""

from src.utils.jsonl_io import (
    read_jsonl,
    read_jsonl_stream,
    write_jsonl,
    append_jsonl,
    load_pydantic_list,
    save_pydantic_list,
)

from src.utils.json_io import (
    read_json,
    write_json,
    ensure_dir,
)

from src.utils.logging import (
    setup_logging,
    get_logger,
)

__all__ = [
    # I/O — JSONL
    "read_jsonl",
    "read_jsonl_stream",
    "write_jsonl",
    "append_jsonl",
    "load_pydantic_list",
    "save_pydantic_list",
    # I/O — JSON
    "read_json",
    "write_json",
    "ensure_dir",
    # Logging
    "setup_logging",
    "get_logger",
]
