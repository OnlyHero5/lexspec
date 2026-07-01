"""
条款提取与语言现象标注。

使用 Stanza 将合同上下文切分为单句（条款），
再解析每条条款并标注检测到的语言现象。
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.linguistic.stanza_parser import StanzaParser
from src.corpus.phenomena_detector import detect_phenomena, is_boilerplate_clause
from src.utils.progress import progress_bar
from src.utils.logging import get_logger

logger = get_logger(__name__)


def split_into_clauses(
    parser: StanzaParser,
    contexts: List[str],
    max_contexts: Optional[int] = None,
    progress_path: Optional[str] = None,
) -> List[str]:
    """将段落上下文切分为单句（条款）。

    使用 Stanza 分句器将各上下文切分为句子，
    返回扁平句子列表。每句为候选条款。

    对超大 CUAD 上下文可先随机抽样以控制运行时间。
    抽样使用固定种子 42 以保证可复现。

    过滤规则：
      - 保留 5–200 个 token 的句子
      - 排除模板文本（页眉、签名等）

    参数：
        parser:        已配置的 StanzaParser 实例。
        contexts:      段落上下文字符串列表。
        max_contexts:  最多处理的上下文数（0 或 None 表示全部）。

    返回：
        提取出的句子字符串扁平列表。
    """
    if max_contexts and max_contexts > 0 and len(contexts) > max_contexts:
        rng = random.Random(42)
        contexts = rng.sample(contexts, max_contexts)
        logger.info("Sampled %d contexts for sentence splitting", max_contexts)

    clauses: List[str] = []
    import stanza as _stanza  # noqa: F811

    for i, ctx in enumerate(
        progress_bar(
            contexts,
            desc="Splitting contexts into sentences",
            unit="ctx",
            progress_path=progress_path,
        )
    ):
        try:
            doc = parser.nlp(ctx)
            for sent in doc.sentences:
                sent_text = sent.text.strip()
                token_count = len(sent.words)
                if 5 <= token_count <= 200:
                    if not is_boilerplate_clause(sent_text):
                        clauses.append(sent_text)
        except Exception as exc:
            logger.debug(
                "Skipping context %d/%d (len=%d): %s",
                i + 1, len(contexts), len(ctx), exc,
            )
            continue

    logger.info(
        "Extracted %d clauses from %d contexts", len(clauses), len(contexts)
    )
    return clauses


def build_clause_records(
    parser: StanzaParser,
    clauses: List[str],
    source_label: str,
    long_distance_mdd: float,
    progress_path: Optional[str] = None,
) -> Tuple[List[Dict], Dict[str, List[int]]]:
    """解析条款列表并附加语言现象标注。

    对每条条款做依存解析并检测语言现象，
    构建候选池。解析失败时（极罕见）静默跳过。

    参数:
        parser:       已配置的 ``StanzaParser`` 实例。
        clauses:      待解析的条款文本字符串列表。
        source_label: 写入每条记录 ``source`` 字段的数据源标识。
        long_distance_mdd: 平均依存距离超过此值时标记为长距离依存现象。

    返回:
        二元组 ``(clause_records, phenomenon_pools)``：
          - ``clause_records``: 含 ``text``、``phenomena``、``source`` 的 dict 列表；
          - ``phenomenon_pools``: 现象名 → 匹配记录在 ``clause_records`` 中索引的列表。
    """
    clause_records: List[Dict] = []
    phenomenon_pools: Dict[str, List[int]] = defaultdict(list)

    logger.info("Parsing %d candidate clauses for phenomenon detection...", len(clauses))

    for idx, clause_text in enumerate(
        progress_bar(
            clauses,
            desc="Detecting language phenomena",
            unit="clause",
            progress_path=progress_path,
        )
    ):
        if is_boilerplate_clause(clause_text):
            continue
        try:
            tree = parser.parse(clause_text)
            if tree.token_count < 3:
                continue
            phen = detect_phenomena(tree, long_distance_mdd=long_distance_mdd)
        except Exception as exc:
            logger.debug(
                "Clause %d parse failed (len=%d): %s", idx, len(clause_text), exc,
            )
            continue

        record = {
            "clause_id": "",
            "text": clause_text,
            "phenomena": phen,
            "source": source_label,
        }
        clause_records.append(record)
        record_idx = len(clause_records) - 1

        # 将条款索引加入现象池（is_definition 不参与
        # 分层抽样）。
        if phen["passive"]:
            phenomenon_pools["passive"].append(record_idx)
        if phen["conditional"]:
            phenomenon_pools["conditional"].append(record_idx)
        if phen["relative_clause"]:
            phenomenon_pools["relative_clause"].append(record_idx)
        if phen["long_distance"]:
            phenomenon_pools["long_distance"].append(record_idx)
        if phen["negation"]:
            phenomenon_pools["negation"].append(record_idx)

    logger.info(
        "Parsing complete: %d valid clauses. Phenomenon pool sizes: %s",
        len(clause_records),
        {k: len(v) for k, v in phenomenon_pools.items()},
    )
    return clause_records, phenomenon_pools
