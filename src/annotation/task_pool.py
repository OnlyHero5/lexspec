"""
标注流水线的并发任务执行框架。

提供进度跟踪、tqdm 显示期间抑制控制台日志，
以及支持串行与并发执行的工作池。
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
# 进度文件辅助函数
# ======================================================================


def _progress_file_path(output_path: str) -> Path:
    """返回与输出文件对应的进度文件路径。"""
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
    """写入一行进度信息，便于 ``tail -f`` 监控。"""
    pct = (100.0 * completed / total) if total else 100.0
    line = (
        f"{desc}: {completed}/{total} ({pct:.1f}%) "
        f"ok={success} fail={failed} updated={_now_iso()}\n"
    )
    with lock:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(line, encoding="utf-8")


def _suppress_console_logging() -> List[Tuple[logging.Handler, int]]:
    """临时抑制控制台日志，避免打断 tqdm 进度条。"""
    saved: List[Tuple[logging.Handler, int]] = []
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stderr:
            saved.append((handler, handler.level))
            handler.setLevel(logging.ERROR)
    return saved


def _restore_console_logging(saved: List[Tuple[logging.Handler, int]]) -> None:
    """恢复先前抑制的控制台日志级别。"""
    for handler, level in saved:
        handler.setLevel(level)


# ======================================================================
# 任务池执行器
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
    """执行任务列表，可选并发，每完成一条即追加到 JSONL。

    参数：
        tasks:         任务列表（每项作为 worker_fn 的第一个参数）。
        worker_fn:     工作函数，签名为 fn(task, index, total) -> dict。
        output_path:   用于追加写入的 JSONL 输出路径。
        workers:       并发工作线程数（<=1 为串行）。
        desc:          进度条描述。
        show_progress: 是否显示 tqdm 进度条。
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
        """单任务完成回调：追加结果并更新进度。"""
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
