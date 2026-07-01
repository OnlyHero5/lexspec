"""
长耗时批处理任务的 tqdm 进度监控工具。

提供统一的 ``progress_bar`` 迭代器：终端显示 tqdm 进度条，
并可选写入 ``.progress`` 文件供 ``tail -f`` 监控。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_progress_file(
    progress_path: Path,
    *,
    desc: str,
    completed: int,
    total: int,
    extra: str = "",
) -> None:
    """写入一行进度信息。"""
    pct = (100.0 * completed / total) if total else 100.0
    suffix = f" {extra}" if extra else ""
    line = (
        f"{desc}: {completed}/{total} ({pct:.1f}%){suffix} "
        f"updated={_now_iso()}\n"
    )
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(line, encoding="utf-8")


def progress_bar(
    iterable: Iterable[T],
    *,
    desc: str,
    unit: str = "it",
    total: Optional[int] = None,
    progress_path: Optional[str] = None,
    disable: bool = False,
) -> Iterator[T]:
    """带 tqdm 的迭代器，可选同步写入 ``.progress`` 文件。

    参数:
        iterable: 待迭代对象。
        desc: tqdm 描述文本。
        unit: tqdm 单位（如 ``clause``、``sample``）。
        total: 总数；为 None 且 ``iterable`` 有 ``__len__`` 时自动推断。
        progress_path: 可选进度文件路径（如 ``outputs/foo.progress``）。
        disable: 为 True 时不显示 tqdm（仍可按需写 progress 文件）。
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None

    path = Path(progress_path) if progress_path else None
    completed = 0

    pbar = tqdm(
        iterable,
        desc=desc,
        unit=unit,
        total=total,
        disable=disable,
        dynamic_ncols=True,
        mininterval=0.3,
    )

    for item in pbar:
        completed += 1
        if path is not None and total:
            if completed == total or completed % 5 == 0:
                write_progress_file(
                    path, desc=desc, completed=completed, total=total,
                )
        yield item

    pbar.close()
    if path is not None and total:
        write_progress_file(path, desc=desc, completed=completed, total=total)
