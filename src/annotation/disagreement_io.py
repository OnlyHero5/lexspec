"""
分歧日志读写
============
将 AnnotationDisagreement 记录序列化并持久化到 JSONL，
供下游分析、报告与质量监控。

导出：
  - save_disagreement_log:    将记录列表写入 JSONL（覆盖）
  - append_disagreement_log:  向已有 JSONL 追加单条记录
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
    """将 AnnotationDisagreement 记录列表保存到 JSONL 文件。

    每条记录序列化为单行 JSON，使用 Pydantic 的
    model_dump(mode="json")，确保枚举序列化为字符串。
    可用以下方式读回：
        records = load_pydantic_list(output_path, AnnotationDisagreement)

    若输出文件已存在，将被**覆盖**。要向已有文件追加而不覆盖，
    请使用 append_disagreement_log()。

    参数：
        disagreements: 待持久化的 AnnotationDisagreement 列表。
        output_path: 输出 JSONL 路径。父目录自动创建。
                     默认为 "data/processed/annotation_log.jsonl"。

    抛出：
        OSError: 无法写入文件（如权限不足）。
        TypeError: 某条记录不是 AnnotationDisagreement 实例。
    """
    if not disagreements:
        logger.warning(
            "save_disagreement_log called with empty list -- writing empty file"
        )

    # 将各 Pydantic 模型转为可 JSON 序列化的 dict。
    # mode="json" 使枚举为字符串、日期时间为 ISO 字符串等。
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

    # 将全部记录写入 JSONL。
    # write_jsonl 会创建父目录并覆盖已有文件。
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
    """向已有 JSONL 文件追加单条 AnnotationDisagreement 记录。

    若文件或父目录不存在则创建。
    使用追加模式，保留已有数据。

    适用于流式标注流水线：发现分歧时逐条记录，
    而非在内存中累积后批量写入。

    参数：
        disagreement: 待追加的单条 AnnotationDisagreement。
        output_path: JSONL 文件路径。默认为
                     "data/processed/annotation_log.jsonl"。

    抛出：
        OSError: 无法以追加方式打开文件。
        TypeError: 记录无法序列化为 JSON。
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
