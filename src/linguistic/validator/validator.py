"""
约束校验器 —— 核心算法
=======================
将大语言模型抽取的法律三元组与 UD 句法结构进行比对，
执行校验、自动修正，或标记为需要 Reflexion 重生成。

这是整个 LexSpec 系统的核心模块（设计文档 §6.8），实现 7 步校验算法:

  步骤 1: 定位根谓词 —— 通过 UD 解析找到主句动词
  步骤 2: 检测被动语态 —— 恢复语义施事/受事映射
  步骤 3: 校验主语 —— 对比大语言模型输出与 UD 主语
  步骤 4: 校验宾语 —— 对比大语言模型输出与 UD 宾语
  步骤 5: 校验条件从句 —— 计算大语言模型条件文本与 UD 条件片段的 IoU
  步骤 6: 校验情态/角色 —— 对比大语言模型角色与 UD 推导的角色
  步骤 7: 确定输出状态 —— VALID / CORRECTED / REFLEXION_REQUIRED

输入: LegalTriplet（来自大语言模型） + 原始文本 + DependencyTree（来自 Stanza）
输出: ValidationResult，含状态、语言学证据、修正列表和反馈

设计原则: 校验器绝不直接修改大语言模型。它只产生修正和反馈。
实验脚本决定是使用修正结果（Ours-Dep 路径）还是触发 Reflexion（Ours-Reflexion 路径）。
"""

from __future__ import annotations

from typing import Optional, List, Tuple

from src.extraction.schema import (
    DependencyTree,
    Token,
    LegalTriplet,
    ValidationStatus,
    ValidationResult,
    LinguisticEvidence,
    FieldCorrection,
    LegalRole,
    ConditionType,
    ConditionSpan,
)
from src.linguistic.stanza_parser import StanzaParser
from src.linguistic.condition_extractor import ConditionExtractor
from src.linguistic.polarity_detector import PolarityDetector
from src.linguistic.validator._thresholds import load_validation_thresholds
from src.linguistic.validator._validate import run_validation
from src.linguistic.validator._depth import compute_depth_metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ConstraintValidator:
    """约束校验器 —— LexSpec 系统的核心算法。

    接收大语言模型产生的 LegalTriplet，对照 UD 依存树逐字段校验/修正。

    协调以下语言学分析组件:
      - StanzaParser: 从文本生成 DependencyTree
      - PassiveDetector: 检测被动语态，恢复论元
      - ConditionExtractor: 提取并分类条件从句
      - PolarityDetector: 从情态/极性推导法律角色

    使用示例::

        parser = StanzaParser()
        validator = ConstraintValidator(parser=parser)
        tree = parser.parse(clause_text)
        result = validator.validate(llm_triplet, clause_text, tree)
        if result.status == ValidationStatus.VALID:
            ...  # 直接使用
        elif result.status == ValidationStatus.CORRECTED:
            ...  # 使用自动修正后的结果
        else:
            ...  # 触发 Reflexion
    """

    def __init__(
        self,
        parser: Optional[StanzaParser] = None,
        condition_extractor: Optional[ConditionExtractor] = None,
        polarity_detector: Optional[PolarityDetector] = None,
        constraints_path: str = "configs/constraints.yaml",
    ):
        """Initialize validator with linguistic analysis components.

        All components are optional — if not provided, defaults are
        created. This supports both production use (shared components)
        and testing (isolated components with mocks).

        Args:
            parser: StanzaParser instance. Created if None.
            condition_extractor: ConditionExtractor instance. Created if None.
            polarity_detector: PolarityDetector instance. Created if None.
            constraints_path: Path to constraints YAML for thresholds.
        """
        self._parser = parser
        self._condition_extractor = condition_extractor
        self._polarity_detector = polarity_detector

        # Thresholds from configuration.
        self._condition_overlap, self._subject_match, self._object_match = (
            load_validation_thresholds(constraints_path)
        )

        # Lazy initialization flags.
        self._parser_owned = parser is None
        self._extractor_owned = condition_extractor is None
        self._detector_owned = polarity_detector is None

    @property
    def parser(self) -> StanzaParser:
        """Get or lazily create the StanzaParser."""
        if self._parser is None:
            self._parser = StanzaParser()
        return self._parser

    @property
    def condition_extractor(self) -> ConditionExtractor:
        """Get or lazily create the ConditionExtractor."""
        if self._condition_extractor is None:
            self._condition_extractor = ConditionExtractor()
        return self._condition_extractor

    @property
    def polarity_detector(self) -> PolarityDetector:
        """Get or lazily create the PolarityDetector."""
        if self._polarity_detector is None:
            self._polarity_detector = PolarityDetector()
        return self._polarity_detector

    # ==================================================================
    # Main Validation Entry Point — The 7-Step Algorithm
    # ==================================================================

    def validate(
        self,
        triplet: LegalTriplet,
        text: str,
        tree: Optional[DependencyTree] = None,
    ) -> ValidationResult:
        """Run the complete 7-step constraint validation algorithm.

        This is the main entry point for all validation. Delegates to
        ``run_validation()`` in ``_validate.py`` which orchestrates the
        7-step pipeline.

        Args:
            triplet: LLM-extracted legal triplet to validate.
            text: Original contract clause text (exact input sent to LLM).
            tree: Pre-parsed dependency tree. Parsed from text if None.

        Returns:
            ValidationResult with full validation details.
        """
        return run_validation(self, triplet, text, tree)

    # ==================================================================
    # Depth Analysis — Additional Linguistic Metrics
    # ==================================================================

    def compute_depth_metrics(
        self,
        tree: DependencyTree,
        predicate_idx: int,
    ) -> dict:
        """Compute additional linguistic depth metrics for analysis.

        These metrics quantify sentence complexity and help explain
        WHY certain sentences cause LLM extraction errors. Used by
        the evaluation module for phenomenon-based error analysis.

        Args:
            tree: Dependency tree.
            predicate_idx: 1-based index of the predicate.

        Returns:
            Dict with keys:
              - mean_dependency_distance (float): MDD for the sentence.
              - predicate_to_subject_distance (int): Token distance from
                predicate to its subject.
              - predicate_to_object_distance (int): Token distance from
                predicate to its object.
              - has_long_distance (bool): Any dependency > 5 tokens.
              - has_acl_relcl (bool): Relative clause present.
              - dependency_depth (int): Max depth from root to leaf.
        """
        return compute_depth_metrics(tree, predicate_idx)
