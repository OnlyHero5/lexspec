"""
语言学错误分析 — 两级分类。

按 PDF 要求：「必须从语言学角度分类归因，每个错误案例需附上语言学解释」
（须从语言学视角分类归因，每个错误案例附语言学解释并引用具体 UD 关系）。

分类体系：

  主类（语言学现象）— 何种句法结构导致错误：
    - 被动语态错误：         使用 nsubj:pass 而非 nsubj；受事与施事混淆
    - 条件边界错误：         advcl/mark 范围识别错误
    - 关系从句混淆：         acl:relcl 嵌套使抽取器混淆
    - 长距离依存：           谓词与论元间依存路径 > 3 条边
    - 否定/例外错误：        否定词或 except/unless 改变角色
    - 其他错误：             兜底类

  次类（字段错误类型）— 哪些字段受影响：
    - 主语错误：             subject.text 或 subject.role 错误
    - 角色错误：             subject.role 错误（文本可能正确）
    - 谓词错误：             action.predicate 错误
    - 宾语错误：             action.object 错误
    - 条件遗漏：             应有条件而缺失
    - 条件过度扩展：         条件文本超出正确边界

错误案例可序列化为 JSONL 供报告使用，并可交叉制表得到错误分布统计。
"""

from __future__ import annotations

import os
from typing import Optional, List, Dict, Any

from src.extraction.schema import (
    LegalTriplet, DependencyTree, Token, ValidationResult,
    ErrorCase, ErrorCategory, FieldErrorType,
)
from src.evaluation.error_detection import detect_field_errors, determine_secondary_category
from src.evaluation.error_classification import determine_primary_category
from src.evaluation.error_summary import generate_error_id
from src.evaluation.error_explanations import generate_explanation
from src.utils.io import append_jsonl, ensure_dir, write_json
from src.utils.constraints import get_validation_thresholds, load_constraints_config
from src.utils.progress import progress_bar
from src.utils.logging import get_logger

logger = get_logger(__name__)
# =============================================================================
# 单条错误案例生成
# =============================================================================


def generate_error_report(
    prediction: LegalTriplet,
    gold: LegalTriplet,
    tree: Optional[DependencyTree] = None,
    validation_result: Optional[ValidationResult] = None,
    error_id: Optional[str] = None,
    constraints_path: str = "configs/constraints.yaml",
) -> Optional[ErrorCase]:
    """生成含语言学解释的详细错误分析案例。

    核心分析函数。对单个 (prediction, gold) 对：
      1. 逐字段比较以确定次级（字段级）类别。
      2. 分析 UD 树（若可用）以确定主类（语言学现象）类别。
      3. 生成中英双语语言学解释，引用具体 UD 关系。

    仅当存在实际错误（任字段不匹配）时生成报告。
    预测与金标完全一致时返回 None。

    主类分类逻辑：

    规则 1 — 被动语态错误：
        - 树含 nsubj:pass + aux:pass（检测到被动）
        - 且预测 subject.text 与 nsubj:pass 文本高重叠
          （受事），而金标主语匹配 obl:agent（真施事）
        -> LLM 将句法主语（受事）与法律主语（施事）混淆。

    规则 2 — 条件边界错误：
        - 树中存在 advcl 关系
        - 且预测条件词元与 UD 条件片段 IoU 低（< 0.5）
        -> LLM 误识别条件从句范围。

    规则 3 — 关系从句混淆：
        - 树中存在 acl:relcl 关系
        - 且谓词或主语/宾语文本匹配关系从句子树内词元
        -> LLM 从关系从句内而非主句抽取。

    规则 4 — 长距离依存错误：
        - 计算根谓词与其主语/宾语在树中的依存距离
        - 当距离 > 3 且预测在远距离论元上有误
        -> 句法距离使抽取困难。

    规则 5 — 否定/例外错误：
        - 树中存在 neg 关系
        - 或条件类型为 EXCEPTION
        - 且 subject.role 错误（应为 prohibited_party，得 obligor）
        -> 否定或例外从句混淆法律角色赋值。

    规则 6 — 其他错误：
        - 未检测到特定语言学模式；兜底分类。

    参数:
        prediction: 系统预测（可能已修正或原始）。
        gold: 金标准三元组。
        tree: UD 依存树（可选，支持更丰富的语言学分析）。
        validation_result: 验证结果（可选，为解释提供修正证据）。
        error_id: 可选错误 ID；为 None 时自动生成 ``E-{index}``。
        constraints_path: 约束 YAML 路径，用于读取长距离依存等分析阈值。

    返回:
        含完整两级分类与双语解释的 ErrorCase，
        或无错误（全部字段匹配）时返回 None。

    示例:
        >>> err = generate_error_report(pred, gold, tree)
        >>> if err:
        ...     print(err.linguistic_explanation)
        # 被动语态错误：系统错误地将受事识别为主语……
    """
    # -------------------------------------------------------------------
    # 步骤 1：检测是否存在错误。
    # -------------------------------------------------------------------
    field_errors = detect_field_errors(prediction, gold)
    if not field_errors:
        # 完全匹配 — 无需错误报告。
        return None

    # -------------------------------------------------------------------
    # 步骤 2：确定次级类别（字段级）。
    # -------------------------------------------------------------------
    secondary = determine_secondary_category(field_errors)

    # -------------------------------------------------------------------
    # 步骤 3：确定主类（语言学现象）。
    # -------------------------------------------------------------------
    primary = ErrorCategory.OTHER_ERROR  # 默认兜底
    ud_evidence: Dict[str, Any] = {}     # 收集 UD 证据供解释使用。

    if tree is not None and tree.token_count > 0:
        thresholds = get_validation_thresholds(
            load_constraints_config(constraints_path), constraints_path
        )
        long_distance_threshold = int(thresholds["long_distance_tokens"])
        primary, ud_evidence = determine_primary_category(
            prediction,
            gold,
            tree,
            field_errors,
            long_distance_token_threshold=long_distance_threshold,
        )

    # -------------------------------------------------------------------
    # 步骤 4：生成双语语言学解释。
    # -------------------------------------------------------------------
    explanation = generate_explanation(
        prediction, gold, tree, primary, secondary, field_errors, ud_evidence,
        validation_result,
    )

    # -------------------------------------------------------------------
    # 步骤 5：组装 ErrorCase。
    # -------------------------------------------------------------------
    error_case = ErrorCase(
        error_id=error_id or generate_error_id(),
        text=gold.subject.text if gold.subject.text else "[no text]",
        prediction=prediction,
        gold=gold,
        primary_category=primary,
        secondary_category=secondary,
        linguistic_explanation=explanation,
    )

    return error_case


# =============================================================================
# 批量错误分类
# =============================================================================


def classify_errors(
    predictions: List[LegalTriplet],
    gold: List[LegalTriplet],
    trees: Optional[List[DependencyTree]] = None,
    validation_results: Optional[List[ValidationResult]] = None,
    constraints_path: str = "configs/constraints.yaml",
) -> List[ErrorCase]:
    """对完整预测集分类全部错误。

    处理每个 (prediction, gold) 对，对有错误的对生成 ErrorCase，
    并返回完整列表。为错误分析的主批量入口。

    参数:
        predictions: 预测 LegalTriplet 列表。
        gold: 金标准 LegalTriplet 列表（等长）。
        trees: 可选 DependencyTree 列表（等长）。
               提供时可做更丰富的主类分类。
        validation_results: 可选 ValidationResult 列表（等长）。
            提供时可丰富解释中的修正证据。
        constraints_path: 约束 YAML 路径，用于主类分类中的距离阈值。

    返回:
        全部有误样本的 ErrorCase 列表。全部匹配时为空列表。

    异常:
        ValueError: 输入列表长度不一致。
    """
    n = len(predictions)
    if len(gold) != n:
        raise ValueError(
            f"predictions and gold must have the same length. "
            f"Got {len(predictions)} and {len(gold)}."
        )
    if trees is not None and len(trees) != n:
        raise ValueError(
            f"trees must have the same length as predictions. "
            f"Got {len(trees)} vs {n}."
        )
    if validation_results is not None and len(validation_results) != n:
        raise ValueError(
            f"validation_results must have the same length as predictions. "
            f"Got {len(validation_results)} vs {n}."
        )

    error_cases: List[ErrorCase] = []

    for i in progress_bar(range(n), desc="Classifying errors", unit="sample"):
        pred = predictions[i]
        g = gold[i]
        t = trees[i] if trees is not None else None
        vr = validation_results[i] if validation_results is not None else None

        error_case = generate_error_report(
            prediction=pred,
            gold=g,
            tree=t,
            validation_result=vr,
            error_id=f"E-{i + 1:04d}",
            constraints_path=constraints_path,
        )

        if error_case is not None:
            error_cases.append(error_case)

    logger.info(
        "Error classification complete: %d errors found out of %d samples (%.1f%%)",
        len(error_cases), n, (len(error_cases) / n * 100) if n > 0 else 0,
    )

    return error_cases


# =============================================================================
# 错误案例持久化
# =============================================================================


def save_error_cases(
    error_cases: List[ErrorCase],
    output_dir: str = "outputs/error_cases",
) -> None:
    """将错误案例保存为分类 JSONL 文件。

    按主错误类别分别写入 JSONL，并写入汇总文件。
    便于针对特定错误类型的分析。

    输出文件:
      - passive_voice_errors.jsonl
      - conditional_boundary_errors.jsonl
      - relative_clause_errors.jsonl
      - long_distance_dependency_errors.jsonl
      - negation_exception_errors.jsonl
      - other_errors.jsonl
      - all_errors.jsonl（完整集合）

    参数:
        error_cases: 待保存的 ``ErrorCase`` 列表。
        output_dir: 输出目录路径；不存在则自动创建。

    返回:
        无（``None``）。按主错误类别写入多个 JSONL 文件及 ``error_summary.json``。

    异常:
        OSError: 目录创建或文件写入失败时由底层 I/O 抛出。
    """
    if not error_cases:
        logger.info("No error cases to save.")
        return

    ensure_dir(output_dir)

    # 主类别值 -> 输出文件名。
    category_files: Dict[str, str] = {
        "passive_voice": "passive_voice_errors.jsonl",
        "conditional_boundary": "conditional_boundary_errors.jsonl",
        "relative_clause": "relative_clause_errors.jsonl",
        "long_distance_dependency": "long_distance_dependency_errors.jsonl",
        "negation_exception": "negation_exception_errors.jsonl",
        "other": "other_errors.jsonl",
    }

    # 各类别计数供日志。
    counts: Dict[str, int] = {cat: 0 for cat in category_files}

    for case in progress_bar(error_cases, desc="Saving error cases", unit="case"):
        # 用 Pydantic model_dump 序列化为 JSON 兼容 dict。
        record = case.model_dump(mode="json")

        # 追加到类别专用文件。
        category = case.primary_category.value
        filename = category_files.get(category, "other_errors.jsonl")
        filepath = os.path.join(output_dir, filename)
        append_jsonl(filepath, record)
        counts[category] = counts.get(category, 0) + 1

        # 追加到汇总文件。
        all_filepath = os.path.join(output_dir, "all_errors.jsonl")
        append_jsonl(all_filepath, record)

    # 记录已写文件摘要。
    for cat, count in counts.items():
        if count > 0:
            logger.info("Saved %d errors to %s/%s", count, output_dir, category_files[cat])

    # 写入含计数的摘要索引文件。
    summary_path = os.path.join(output_dir, "error_summary.json")
    write_json(summary_path, {
        "total_errors": len(error_cases),
        "categories": {cat: count for cat, count in counts.items() if count > 0},
    })

    logger.info(
        "Error cases saved to %s: %d total across %d categories",
        output_dir, len(error_cases), sum(1 for c in counts.values() if c > 0),
    )
