#!/usr/bin/env python3
"""
LexSpec 步骤 02: 分阶段双模型标注流水线
========================================

本脚本分四个阶段运行，每次仅在远程服务器上加载一个模型：

  阶段 1 —— Gemma 独立标注，结果保存至本地：
    python experiments/step_02_annotate_gold.py annotate --model gemma

  阶段 2 —— 切换服务器至 Qwen 后：
    python experiments/step_02_annotate_gold.py annotate --model qwen
    python experiments/step_02_annotate_gold.py review --reviewer qwen --source gemma

  阶段 3 —— 切换服务器回 Gemma 后：
    python experiments/step_02_annotate_gold.py review --reviewer gemma --source qwen

  阶段 4 —— 合并为金标准（无需大语言模型）：
    python experiments/step_02_annotate_gold.py merge

默认所有输出写入 data/annotations/。支持 --resume 断点续跑。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 将项目根目录加入 Python 路径，以便正确解析 src 导入。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.annotation.commands import build_parser, cmd_annotate, cmd_review
from src.annotation.merge_engine import cmd_merge
from src.utils.logging import setup_logging


def main() -> None:
    """主入口：解析命令行参数并分发至对应子命令。"""
    parser = build_parser()
    args = parser.parse_args()
    import logging as _logging
    setup_logging(log_dir="outputs/logs", level=_logging.INFO)

    if args.command == "annotate":
        cmd_annotate(args)
    elif args.command == "review":
        cmd_review(args)
    elif args.command == "merge":
        cmd_merge(args)


if __name__ == "__main__":
    main()
