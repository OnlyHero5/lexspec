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
        """使用语言学分析组件初始化校验器。

        所有组件均为可选 —— 未提供时创建默认实例。
        既支持生产使用（共享组件），也支持测试（带 mock 的隔离组件）。

        参数：
            parser: StanzaParser 实例。为 None 时自动创建。
            condition_extractor: ConditionExtractor 实例。为 None 时自动创建。
            polarity_detector: PolarityDetector 实例。为 None 时自动创建。
            constraints_path: 阈值配置用的约束 YAML 路径。
        """
        self._parser = parser
        self._condition_extractor = condition_extractor
        self._polarity_detector = polarity_detector

        # 来自配置的阈值。
        self._condition_overlap, self._subject_match, self._object_match = (
            load_validation_thresholds(constraints_path)
        )

        # 延迟初始化标志。
        self._parser_owned = parser is None
        self._extractor_owned = condition_extractor is None
        self._detector_owned = polarity_detector is None

    @property
    def parser(self) -> StanzaParser:
        """获取或延迟创建 Stanza 依存解析器。

        返回:
            ``StanzaParser`` 实例；若构造时未注入则首次访问时自动创建。
        """
        if self._parser is None:
            self._parser = StanzaParser()
        return self._parser

    @property
    def condition_extractor(self) -> ConditionExtractor:
        """获取或延迟创建条件从句提取器。

        返回:
            ``ConditionExtractor`` 实例，用于从 UD 树识别 advcl+mark 条件边界。
        """
        if self._condition_extractor is None:
            self._condition_extractor = ConditionExtractor()
        return self._condition_extractor

    @property
    def polarity_detector(self) -> PolarityDetector:
        """获取或延迟创建情态/极性检测器。

        返回:
            ``PolarityDetector`` 实例，用于从助动词与否定推导法律角色。
        """
        if self._polarity_detector is None:
            self._polarity_detector = PolarityDetector()
        return self._polarity_detector

    # ==================================================================
    # 主校验入口 —— 七步算法
    # ==================================================================

    def validate(
        self,
        triplet: LegalTriplet,
        text: str,
        tree: Optional[DependencyTree] = None,
    ) -> ValidationResult:
        """运行完整的七步约束校验算法。

        所有校验的主入口。委托至 ``_validate.py`` 中的
        ``run_validation()``，由其编排七步流水线。

        参数：
            triplet: 待校验的大语言模型抽取法律三元组。
            text: 原始合同从句文本（发送给大语言模型的精确输入）。
            tree: 预解析依存树。为 None 时从 text 解析。

        返回：
            含完整校验详情的 ValidationResult。
        """
        return run_validation(self, triplet, text, tree)

    # ==================================================================
    # 深度分析 —— 附加语言学度量
    # ==================================================================

    def compute_depth_metrics(
        self,
        tree: DependencyTree,
        predicate_idx: int,
    ) -> dict:
        """计算附加语言学深度度量供分析使用。

        这些度量量化句子复杂度，帮助解释
        某些句子导致大语言模型抽取错误的原因。
        供评估模块进行基于现象的错误分析。

        参数：
            tree: 依存树。
            predicate_idx: 谓词的 1 基索引。

        返回：
            含以下键的字典：
              - mean_dependency_distance (float): 句子的 MDD。
              - predicate_to_subject_distance (int): 谓词到主语的距离。
              - predicate_to_object_distance (int): 谓词到宾语的距离。
              - has_long_distance (bool): 是否存在 >5 词元的依存。
              - has_acl_relcl (bool): 是否存在关系从句。
              - dependency_depth (int): 从根到叶的最大深度。
        """
        return compute_depth_metrics(tree, predicate_idx)
