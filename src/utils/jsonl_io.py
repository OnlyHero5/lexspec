"""
JSONL I/O Utilities for LexSpec
================================
JSONL reading/writing and Pydantic serialization helpers.

This module provides the canonical JSONL I/O layer. All JSONL file access
goes through these functions — no module reads or writes JSONL files directly.
This ensures consistent error handling, encoding, and path management across
the pipeline.

Design decisions:
  - All paths accept str | Path, but internally use pathlib.Path.
  - JSONL (JSON Lines) is the primary serialization format — one JSON
    object per line, suitable for streaming and incremental processing.
  - Pydantic serialization uses model_dump(mode='json') to ensure
    enum values are serialized as strings, not Python enum objects.
  - Write operations create parent directories automatically.
  - Read operations raise FileNotFoundError with clear messages.
"""

import json
from typing import List, Iterator, Any, TypeVar, Type
from pathlib import Path

# Generic type variable for Pydantic model classes
T = TypeVar("T")

# Default encoding used for all file I/O
_ENCODING = "utf-8"


# ============================================================================
# JSONL — Line-delimited JSON (primary data format)
# ============================================================================


def read_jsonl(file_path: str | Path) -> List[dict]:
    """
    Read all records from a JSONL file into a list of dicts.

    Each line in the file must be a valid JSON object. Blank lines are
    skipped silently. This method loads the entire file into memory and
    is suitable for small-to-medium datasets. For large files, use
    read_jsonl_stream() instead.

    Args:
        file_path: Path to the JSONL file.

    Returns:
        List of dicts, one per JSONL record. Empty list if the file
        contains only blank lines.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If any non-blank line is not valid JSON.
    """
    file_path = Path(file_path)  # Normalize to pathlib.Path

    if not file_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    records: List[dict] = []
    with open(file_path, "r", encoding=_ENCODING) as fh:
        for line_num, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                # Skip blank lines — common in hand-edited JSONL files
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                # Attach file and line context to the error for debugging
                raise json.JSONDecodeError(
                    f"Invalid JSON at {file_path}:{line_num}: {exc.msg}",
                    exc.doc,
                    exc.pos,
                ) from exc
    return records


def read_jsonl_stream(file_path: str | Path) -> Iterator[dict]:
    """
    Stream records from a JSONL file one at a time.

    Memory-efficient generator for large files — only one record is
    held in memory at a time. Use this when processing datasets that
    exceed available RAM.

    Args:
        file_path: Path to the JSONL file.

    Yields:
        Dict for each JSONL record.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If any non-blank line is not valid JSON.
    """
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    with open(file_path, "r", encoding=_ENCODING) as fh:
        for line_num, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"Invalid JSON at {file_path}:{line_num}: {exc.msg}",
                    exc.doc,
                    exc.pos,
                ) from exc


def write_jsonl(file_path: str | Path, records: List[dict]) -> None:
    """
    Write a list of dicts to a JSONL file.

    Creates parent directories if they do not exist. Overwrites the file
    if it already exists. Each dict is serialized as a single JSON line
    with no extra whitespace.

    Args:
        file_path: Path for the output JSONL file.
        records:   List of dicts to write. Empty list produces an empty file.

    Raises:
        OSError: If the file cannot be created or written to.
        TypeError: If any record contains non-JSON-serializable data.
    """
    file_path = Path(file_path)
    # Ensure parent directories exist before writing
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding=_ENCODING) as fh:
        for record in records:
            # ensure_ascii=False preserves Unicode characters in legal text
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(file_path: str | Path, record: dict) -> None:
    """
    Append a single record to a JSONL file.

    Creates the file and parent directories if they do not exist.
    Uses append mode so existing data is preserved. Thread-safe for
    POSIX systems when each write is a single line.

    Args:
        file_path: Path to the JSONL file.
        record:    A single dict to append as a new JSONL line.

    Raises:
        OSError: If the file cannot be opened for appending.
        TypeError: If the record contains non-JSON-serializable data.
    """
    file_path = Path(file_path)
    # Ensure the directory chain exists; mkdir won't raise if they do
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "a", encoding=_ENCODING) as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================================
# Pydantic Model Serialization — Type-safe I/O
# ============================================================================


def load_pydantic_list(file_path: str | Path, model_class: Type[T]) -> List[T]:
    """
    Read a JSONL file and parse each record into a Pydantic model instance.

    Uses model_validate (Pydantic v2) for validation during deserialization.
    This ensures that all data entering the system conforms to the schema.

    Args:
        file_path:    Path to the JSONL file.
        model_class:  A Pydantic BaseModel subclass to parse each record into.

    Returns:
        List of validated Pydantic model instances.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If any record fails schema validation.
        json.JSONDecodeError: If any line is not valid JSON.
    """
    records = read_jsonl(file_path)
    # model_validate is the Pydantic v2 replacement for parse_obj
    return [model_class.model_validate(record) for record in records]


def save_pydantic_list(file_path: str | Path, instances: List[Any]) -> None:
    """
    Serialize a list of Pydantic models to a JSONL file.

    Uses model_dump(mode='json') (Pydantic v2) to convert each instance
    to a JSON-serializable dict. The mode='json' parameter ensures:
      - Enum values are serialized as their string values, not enum objects
      - datetime objects (if any) are converted to ISO 8601 strings
      - Path objects are converted to strings

    Args:
        file_path: Path for the output JSONL file.
        instances: List of Pydantic BaseModel instances.

    Raises:
        OSError: If the file cannot be written.
        AttributeError: If any item lacks model_dump (i.e., is not a Pydantic model).
    """
    # Convert each model to a plain dict using Pydantic's serialization
    records = [instance.model_dump(mode="json") for instance in instances]
    write_jsonl(file_path, records)
