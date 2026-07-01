"""
Concurrent task execution framework for the annotation pipeline.

Provides progress tracking, console log suppression during tqdm display,
and a worker pool that supports both serial and concurrent execution.
"""

from __future__ import annotations

import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, List, Tuple

from tqdm import tqdm

from src.annotation.cli_utils import _now_iso
from src.utils.io import append_jsonl

logger = logging.getLogger(__name__)


# ======================================================================
# Progress file helpers
# ======================================================================


def _progress_file_path(output_path: str) -> Path:
    """Return the progress file path corresponding to an output file."""
    return Path(output_path).with_suffix(".progress")


def _write_progress_file(
    progress_path: Path,
    *,
    desc: str,
    completed: int,
    total: int,
    success: int,
    failed: int,
    lock: threading.Lock,
) -> None:
    """Write one progress line, suitable for ``tail -f`` monitoring."""
    pct = (100.0 * completed / total) if total else 100.0
    line = (
        f"{desc}: {completed}/{total} ({pct:.1f}%) "
        f"ok={success} fail={failed} updated={_now_iso()}\n"
    )
    with lock:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(line, encoding="utf-8")


def _suppress_console_logging() -> List[Tuple[logging.Handler, int]]:
    """Temporarily suppress console log output to avoid breaking tqdm bars."""
    saved: List[Tuple[logging.Handler, int]] = []
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stderr:
            saved.append((handler, handler.level))
            handler.setLevel(logging.ERROR)
    return saved


def _restore_console_logging(saved: List[Tuple[logging.Handler, int]]) -> None:
    """Restore previously suppressed console log output levels."""
    for handler, level in saved:
        handler.setLevel(level)


# ======================================================================
# Task pool executor
# ======================================================================


def _run_task_pool(
    tasks: List[Any],
    worker_fn: Callable[..., dict],
    output_path: str,
    workers: int,
    desc: str,
    *,
    show_progress: bool = True,
) -> None:
    """Execute a list of tasks with optional concurrency, appending each
    result to a JSONL file as it completes.

    Args:
        tasks:         Task list (each passed as first arg to worker_fn).
        worker_fn:     Worker function with signature fn(task, index, total) -> dict.
        output_path:   JSONL output path for append writes.
        workers:       Number of concurrent worker threads (<=1 for serial).
        desc:          Progress bar description.
        show_progress: Whether to display a tqdm progress bar.
    """
    if not tasks:
        return

    total = len(tasks)
    progress_path = _progress_file_path(output_path)
    lock = threading.Lock()
    completed = 0
    success_count = 0
    failed_count = 0

    def _on_result(record: dict) -> None:
        """Callback on single task completion: append + update progress."""
        nonlocal completed, success_count, failed_count
        with lock:
            append_jsonl(output_path, record)
        completed += 1
        if record.get("success") is True:
            success_count += 1
        else:
            failed_count += 1
        pbar.update(1)
        if completed == total or completed % 5 == 0:
            pbar.set_postfix(ok=success_count, fail=failed_count, refresh=True)
        _write_progress_file(
            progress_path, desc=desc, completed=completed, total=total,
            success=success_count, failed=failed_count, lock=lock,
        )

    pbar = tqdm(
        total=total, desc=desc, unit="clause",
        file=sys.stdout, dynamic_ncols=True, mininterval=0.3,
        disable=not show_progress,
    )

    saved_handlers = _suppress_console_logging()
    try:
        if workers <= 1:
            for i, task in enumerate(tasks, start=1):
                _on_result(worker_fn(task, i, total))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(worker_fn, task, 0, total) for task in tasks
                ]
                for fut in as_completed(futures):
                    _on_result(fut.result())
    finally:
        pbar.close()
        _restore_console_logging(saved_handlers)
        if show_progress:
            print(
                f"\n{desc} complete: {completed}/{total} "
                f"(success={success_count}, failed={failed_count})",
                flush=True,
            )
            print(f"Progress file: {progress_path}", flush=True)
