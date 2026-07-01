#!/usr/bin/env python3
"""
LexSpec Step 02: Staged dual-model annotation pipeline
========================================================

This script runs in four stages, loading only one model on the remote
server at a time:

  Stage 1 -- Gemma independent annotation, results saved locally:
    python experiments/step_02_annotate_gold.py annotate --model gemma

  Stage 2 -- After switching server to Qwen:
    python experiments/step_02_annotate_gold.py annotate --model qwen
    python experiments/step_02_annotate_gold.py review --reviewer qwen --source gemma

  Stage 3 -- After switching server back to Gemma:
    python experiments/step_02_annotate_gold.py review --reviewer gemma --source qwen

  Stage 4 -- Merge into gold standard (no LLM needed):
    python experiments/step_02_annotate_gold.py merge

All output is written to data/annotations/ by default. Supports --resume
for checkpoint/resume.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path so src imports resolve correctly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.annotation.commands import build_parser, cmd_annotate, cmd_review
from src.annotation.merge_engine import cmd_merge
from src.utils.logging import setup_logging


def main() -> None:
    """Main entry point: parse CLI args and dispatch to the appropriate subcommand."""
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
