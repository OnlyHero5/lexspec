"""
Disagreement Log I/O
=====================
Serialization and persistence for AnnotationDisagreement records
to JSONL files for downstream analysis, reporting, and quality monitoring.

Exported:
  - save_disagreement_log:    Write a list of records to a JSONL file (overwrite)
  - append_disagreement_log:  Append a single record to an existing JSONL file
"""

from __future__ import annotations

from typing import List

from src.extraction.schema import AnnotationDisagreement
from src.utils.io import append_jsonl, write_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)


def save_disagreement_log(
    disagreements: List[AnnotationDisagreement],
    output_path: str = "data/processed/annotation_log.jsonl",
) -> None:
    """Save a list of AnnotationDisagreement records to a JSONL file.

    Each record is serialized as a single JSON line using Pydantic's
    model_dump(mode="json") to ensure enum values are serialized as
    strings. The output file can be read back with:
        records = load_pydantic_list(output_path, AnnotationDisagreement)

    If the output file already exists, it is OVERWRITTEN. Use
    append_disagreement_log() to add to an existing file without
    overwriting.

    Args:
        disagreements: List of AnnotationDisagreement records to persist.
        output_path: Path to the output JSONL file. Parent directories
                     are created automatically. Defaults to
                     "data/processed/annotation_log.jsonl".

    Raises:
        OSError: If the file cannot be written (e.g., permission denied).
        TypeError: If any record is not an AnnotationDisagreement instance.
    """
    if not disagreements:
        logger.warning(
            "save_disagreement_log called with empty list -- writing empty file"
        )

    # Convert each Pydantic model to a JSON-serializable dict.
    # mode="json" ensures enums become strings, datetimes become ISO strings, etc.
    records: List[dict] = []
    for i, disagreement in enumerate(disagreements):
        if not isinstance(disagreement, AnnotationDisagreement):
            logger.error(
                "Item %d is not an AnnotationDisagreement (type=%s) -- skipping",
                i, type(disagreement).__name__,
            )
            continue
        try:
            records.append(disagreement.model_dump(mode="json"))
        except Exception as exc:
            logger.error(
                "Failed to serialize AnnotationDisagreement %d: %s -- skipping",
                i, exc,
            )
            continue

    # Write all records to the JSONL file.
    # write_jsonl creates parent directories and overwrites existing files.
    write_jsonl(output_path, records)

    logger.info(
        "Saved %d disagreement records to %s",
        len(records),
        output_path,
    )


def append_disagreement_log(
    disagreement: AnnotationDisagreement,
    output_path: str = "data/processed/annotation_log.jsonl",
) -> None:
    """Append a single AnnotationDisagreement record to an existing JSONL file.

    Creates the file and parent directories if they do not exist.
    Uses append mode so existing data is preserved.

    This is useful for streaming annotation pipelines where disagreements
    are logged one at a time as they are discovered, rather than
    accumulated in memory and written in bulk.

    Args:
        disagreement: A single AnnotationDisagreement record to append.
        output_path: Path to the JSONL file. Defaults to
                     "data/processed/annotation_log.jsonl".

    Raises:
        OSError: If the file cannot be opened for appending.
        TypeError: If the record cannot be serialized to JSON.
    """
    if not isinstance(disagreement, AnnotationDisagreement):
        raise TypeError(
            f"Expected AnnotationDisagreement, got {type(disagreement).__name__}"
        )

    try:
        record = disagreement.model_dump(mode="json")
    except Exception as exc:
        logger.error("Failed to serialize AnnotationDisagreement: %s", exc)
        raise TypeError(
            f"Cannot serialize AnnotationDisagreement: {exc}"
        ) from exc

    append_jsonl(output_path, record)

    logger.debug(
        "Appended disagreement record for clause '%s' to %s",
        disagreement.clause_id,
        output_path,
    )
