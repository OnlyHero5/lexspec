#!/usr/bin/env python3
"""从 curated gold_500 抽取 100 条，生成对齐的测试集与金标。

输出:
  data/processed/gold_triplets_100.jsonl  — 完整金标（含 triplet / qwen / gemma / curated）
  data/processed/gold_testset_100.jsonl   — 抽取用测试集（仅 text + phenomena）
  data/processed/lexspec_100.jsonl        — 与 gold_testset_100 同步（兼容 step_03–07 默认路径）

默认取 gold_triplets_500.jsonl 的前 N=100 条（与 gold_testset_500 同序）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD_500 = ROOT / "data/processed/gold_triplets_500.jsonl"
TEST_500 = ROOT / "data/processed/gold_testset_500.jsonl"
GOLD_100 = ROOT / "data/processed/gold_triplets_100.jsonl"
TEST_100 = ROOT / "data/processed/gold_testset_100.jsonl"
LEXSPEC_100 = ROOT / "data/processed/lexspec_100.jsonl"
MANIFEST = ROOT / "data/processed/gold_100_manifest.json"
DEFAULT_N = 100


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    for p in (GOLD_500, TEST_500):
        if not p.exists():
            print(f"缺少: {p}", file=sys.stderr)
            sys.exit(1)

    gold_all = load_jsonl(GOLD_500)
    test_all = load_jsonl(TEST_500)
    if len(gold_all) != len(test_all):
        print("gold_500 与 test_500 行数不一致", file=sys.stderr)
        sys.exit(1)
    if len(gold_all) < n:
        print(f"gold_500 仅 {len(gold_all)} 条，无法抽取 {n} 条", file=sys.stderr)
        sys.exit(1)

    gold_ids = [r["clause_id"] for r in gold_all]
    test_ids = [r["clause_id"] for r in test_all]
    if gold_ids != test_ids:
        print("gold_500 与 test_500 clause_id 顺序不一致", file=sys.stderr)
        sys.exit(1)

    gold_sub = gold_all[:n]
    test_sub = test_all[:n]

    write_jsonl(GOLD_100, gold_sub)
    write_jsonl(TEST_100, test_sub)
    write_jsonl(LEXSPEC_100, test_sub)

    manifest = {
        "source_gold": str(GOLD_500.relative_to(ROOT)),
        "source_testset": str(TEST_500.relative_to(ROOT)),
        "selection": f"first_{n}_in_file_order",
        "count": n,
        "clause_ids": [r["clause_id"] for r in gold_sub],
        "curated": all(r.get("curated") for r in gold_sub),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"已写入 {n} 条 curated 金标:")
    print(f"  {GOLD_100.name}")
    print(f"  {TEST_100.name}")
    print(f"  {LEXSPEC_100.name} (已覆盖旧 lexspec_100)")
    print(f"  {MANIFEST.name}")
    print(f"  ID 范围: {manifest['clause_ids'][0]} … {manifest['clause_ids'][-1]}")


if __name__ == "__main__":
    main()
