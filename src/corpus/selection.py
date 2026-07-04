"""
构建 LexSpec 评估语料库的条款选择策略。
"""

from __future__ import annotations

import random
from typing import Dict, List

from src.linguistic.stanza_parser import StanzaParser
from src.corpus.clause_processor import build_clause_records
from src.utils.logging import get_logger

logger = get_logger(__name__)


def select_all_clauses(clause_records: List[Dict]) -> List[Dict]:
    """全量模式：返回全部有效已解析条款（排除定义性条款）。

    参数:
        clause_records: ``build_clause_records`` 产出的记录列表，每条含
            ``text``、``phenomena``、``source`` 等键。

    返回:
        过滤后的条款 dict 列表；每条新增 ``clause_id``（格式 ``C-00001``），
        且 ``phenomena.is_definition == True`` 的条目已排除。
    """
    result: List[Dict] = []
    for i, record in enumerate(clause_records):
        if record["phenomena"].get("is_definition", False):
            continue
        out = dict(record)
        out["clause_id"] = f"C-{i + 1:05d}"
        result.append(out)
    return result


def select_balanced_testset(
    parser: StanzaParser,
    clauses: List[str],
    target_count: int,
    phenomenon_quotas: Dict[str, int],
    random_seed: int,
    long_distance_mdd: float,
    source_label: str = "cuad_v1",
    progress_path: str | None = None,
) -> List[Dict]:
    """通过分层抽样选取覆盖各语言现象的平衡测试集。

    先按 ``phenomenon_quotas`` 为被动、条件、关系从句等现象填充配额，
    再按多现象重叠度优先填充剩余名额至 ``target_count``。

    参数:
        parser: 已初始化的 ``StanzaParser``，用于依存解析与现象检测。
        clauses: 候选条款文本字符串列表（通常来自 CUAD 分句结果）。
        target_count: 目标测试集总条数。
        phenomenon_quotas: 现象池名 → 最低条数 的字典（由
            ``get_phenomenon_quotas`` 生成）。
        random_seed: 随机种子，控制洗牌与抽样顺序的可复现性。
        long_distance_mdd: 判定长距离依存现象的平均依存距离阈值。
        source_label: 写入每条记录的 ``source`` 字段（如 ``"cuad_v1"``）。

    返回:
        已选条款 dict 列表，每条含 ``clause_id``、``text``、``phenomena``、
        ``source``；``clause_id`` 格式为 ``C-0001``。
    """
    rng = random.Random(random_seed)
    rng.shuffle(clauses)

    clause_records, phenomenon_pools = build_clause_records(
        parser, clauses, source_label,
        long_distance_mdd=long_distance_mdd,
        progress_path=progress_path,
    )

    def_indices = {
        i for i, r in enumerate(clause_records)
        if r["phenomena"].get("is_definition", False)
    }
    for pool_name in phenomenon_pools:
        phenomenon_pools[pool_name] = [
            i for i in phenomenon_pools[pool_name] if i not in def_indices
        ]
    if def_indices:
        logger.info("Excluded %d definition clauses from candidate pool", len(def_indices))

    selected: set = set()
    fill_order = sorted(phenomenon_quotas.keys(), key=lambda k: len(phenomenon_pools.get(k, [])))

    for phen_name in fill_order:
        quota = phenomenon_quotas[phen_name]
        pool = phenomenon_pools.get(phen_name, [])
        rng.shuffle(pool)

        for idx in pool:
            current_count = sum(
                1 for i in selected
                if clause_records[i]["phenomena"].get(phen_name, False)
            )
            if current_count >= quota:
                break
            if idx not in selected:
                selected.add(idx)

    remaining_indices = [
        i for i in range(len(clause_records))
        if i not in selected and i not in def_indices
    ]
    remaining_indices.sort(
        key=lambda i: -sum(
            1 for k, v in clause_records[i]["phenomena"].items()
            if k != "is_definition" and v
        )
    )

    while len(selected) < target_count and remaining_indices:
        selected.add(remaining_indices.pop(0))

    result: List[Dict] = []
    for i, idx in enumerate(sorted(selected)):
        record = dict(clause_records[idx])
        record["clause_id"] = f"C-{i + 1:05d}"
        result.append(record)

    phen_counts = {
        phen: sum(1 for r in result if r["phenomena"][phen])
        for phen in phenomenon_quotas
    }
    logger.info("Selected %d clauses: %s", len(result), phen_counts)
    return result
