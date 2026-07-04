"""
标注流水线的 CLI 工具函数。

提供配置加载、客户端构建、模型名解析、
输出路径选择，以及标注去重/续跑支持。
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

import yaml

from src.extraction.client import LLMClient, ClientConfig
from src.extraction.schema import LegalTriplet
from src.utils.config import 构建标注客户端
from src.utils.io import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认文件路径
# ---------------------------------------------------------------------------
# 正式评测（消融/对比实验）统一使用 curated gold_500 前 100 条：
#   lexspec_100.jsonl + gold_triplets_100.jsonl（见 scripts/build_gold_100_from_500.py）
CURATED_TESTSET_100 = "data/processed/lexspec_100.jsonl"
CURATED_GOLD_100 = "data/processed/gold_triplets_100.jsonl"
DEFAULT_TESTSET = CURATED_TESTSET_100
ANNOT_DIR = "data/annotations"
GEMMA_ANNOT = f"{ANNOT_DIR}/gemma_annotations.jsonl"
QWEN_ANNOT = f"{ANNOT_DIR}/qwen_annotations.jsonl"
QWEN_REVIEW_GEMMA = f"{ANNOT_DIR}/qwen_review_gemma.jsonl"
GEMMA_REVIEW_QWEN = f"{ANNOT_DIR}/gemma_review_qwen.jsonl"
# 历史双模型 merge 输出（step_02 默认）；与 curated gold_100 评测集无关。
GOLD_OUT = "data/processed/gold_triplets.jsonl"
DISAGREE_OUT = f"{ANNOT_DIR}/needs_human_review.jsonl"

# 模型别名映射 — 同时支持 primary/secondary 与具体模型名。
MODEL_ALIASES = {
    "gemma": "secondary",
    "qwen": "primary",
    "primary": "primary",
    "secondary": "secondary",
}


# ======================================================================
# 工具函数
# ======================================================================


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件；缺失时抛出 FileNotFoundError。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_client(config: dict, model_key: str) -> LLMClient:
    """为 model_key 所标识的标注模型构建 LLM 客户端。

    委托 src.utils.config.构建标注客户端 进行统一配置加载。
    """
    role = MODEL_ALIASES.get(model_key, model_key)
    if role not in ("primary", "secondary"):
        raise ValueError(f"Unknown model identifier: {model_key}")
    return 构建标注客户端(config=config, model_role=role)


def model_role_name(config: dict, model_key: str) -> str:
    """从配置解析模型的显示名称。"""
    role = MODEL_ALIASES.get(model_key, model_key)
    ann = config.get("models", {}).get("annotation", {})
    if role == "primary":
        return ann.get("primary", {}).get("name", "qwen")
    return ann.get("secondary", {}).get("name", "gemma")


def output_path_for_model(model_key: str) -> str:
    """返回模型标识对应的默认标注输出路径。"""
    role = MODEL_ALIASES.get(model_key, model_key)
    if role in ("primary", "qwen"):
        return QWEN_ANNOT
    return GEMMA_ANNOT


def review_output_path(reviewer: str, source: str) -> str:
    """返回审查者/被审查源组合的默认审查输出路径。"""
    rev = MODEL_ALIASES.get(reviewer, reviewer)
    src = MODEL_ALIASES.get(source, source)
    if rev in ("primary", "qwen") and src in ("secondary", "gemma"):
        return QWEN_REVIEW_GEMMA
    if rev in ("secondary", "gemma") and src in ("primary", "qwen"):
        return GEMMA_REVIEW_QWEN
    raise ValueError(f"Unsupported review combination: {reviewer} reviews {source}")


def triplet_to_dict(t: LegalTriplet) -> dict:
    """LegalTriplet -> 可序列化 dict。"""
    return t.model_dump(mode="json")


def dict_to_triplet(d: dict) -> LegalTriplet:
    """Dict -> LegalTriplet。"""
    return LegalTriplet.model_validate(d)


# ======================================================================
# 标注去重 — 处理续跑产生的重复
# ======================================================================


def _deduplicate_annotations(output_path: str) -> int:
    """对标注文件去重，每个 clause_id 保留最佳记录。

    选择策略：成功记录优先于失败；同状态下保留最后一条。
    文件原地覆盖。

    参数：
        output_path: 标注 JSONL 文件路径。

    返回：
        移除的重复记录数。
    """
    path = Path(output_path)
    if not path.exists():
        return 0

    records = read_jsonl(output_path)
    if not records:
        return 0

    best: Dict[str, dict] = {}
    for rec in records:
        cid = rec.get("clause_id", "")
        if not cid:
            continue
        if cid not in best:
            best[cid] = rec
        else:
            existing = best[cid]
            # 成功优先；同状态保留较新的一条。
            if rec.get("success") and not existing.get("success"):
                best[cid] = rec
            elif not rec.get("success") and existing.get("success"):
                pass  # 保留已有成功记录。
            else:
                best[cid] = rec

    removed = len(records) - len(best)
    if removed > 0:
        deduped = [best[cid] for cid in sorted(best)]
        write_jsonl(output_path, deduped)
        logger.info(
            "Annotation dedup complete %s: %d -> %d (removed %d duplicates)",
            output_path, len(records), len(deduped), removed,
        )
    return removed


# ======================================================================
# 已完成标注跟踪 — 用于 --resume 模式
# ======================================================================


def _load_completed_ids(output_path: str) -> Set[str]:
    """从文件中读取已成功完成标注的 clause_id。

    仅统计 success=True 且三元组含实质性内容的记录。
    """
    path = Path(output_path)
    if not path.exists():
        return set()
    done: Set[str] = set()
    for rec in read_jsonl(path):
        cid = rec.get("clause_id")
        if not cid or rec.get("success") is not True:
            continue
        triplet = rec.get("triplet") or {}
        subj = (triplet.get("subject") or {}).get("text", "")
        pred = (triplet.get("action") or {}).get("predicate", "")
        obj = (triplet.get("action") or {}).get("object", "")
        if subj.strip() or pred.strip() or obj.strip():
            done.add(cid)
    return done
