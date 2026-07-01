"""
极性与情态检测 —— 法律角色分类
============================================================
检测情态助动词与否定，分类主语的法律角色
（义务方、权利方、禁止方等）。

子模块：
  _config:     回退规则、YAML 配置加载、查找表构建
  _negation:   否定检测（_is_negated、_has_lexical_negation）
  _detector:   主检测接口（detect、detect_modality、detect_role_with_voice、get_modality_evidence）
"""

from __future__ import annotations

from src.linguistic.polarity._config import (
    _load_rules as _load_rules_fn,
    _build_lookup as _build_lookup_fn,
)
from src.linguistic.polarity._negation import (
    _is_negated as _is_negated_fn,
    _has_lexical_negation as _has_lexical_negation_fn,
)
from src.linguistic.polarity._detector import (
    detect as _detect_fn,
    detect_modality as _detect_modality_fn,
    detect_role_with_voice as _detect_role_with_voice_fn,
    get_modality_evidence as _get_modality_evidence_fn,
)


class PolarityDetector:
    """检测情态与极性以确定法律角色。

    从 configs/constraints.yaml 加载情态规则，将法律从句主语
    分类为以下之一：
      - OBLIGOR: 负有义务的当事方（shall/must + 肯定）
      - RIGHT_HOLDER: 享有权利的当事方（may + 肯定）
      - PROHIBITED_PARTY: 受禁止的当事方（情态 + 否定）
      - INDEMNIFYING_PARTY: 负有赔偿义务的当事方
        （"indemnify" 的词项覆盖）
      - OTHER: 无法确定（无情态、歧义或边界情况）

    分类为基于规则且确定性的，以 UD 句法结构为依据。
    提供与大语言模型角色赋值比对的「真值」。

    使用示例::

        detector = PolarityDetector("configs/constraints.yaml")
        role, polarity = detector.detect(tree, predicate_idx, "deliver")
        # role 为 LegalRole.OBLIGOR，polarity 为 "positive"
    """

    # 从子模块导入 —— 绑定为实例方法：
    _load_rules = _load_rules_fn
    _build_lookup = _build_lookup_fn
    _is_negated = _is_negated_fn
    _has_lexical_negation = _has_lexical_negation_fn
    detect = _detect_fn
    detect_modality = _detect_modality_fn
    detect_role_with_voice = _detect_role_with_voice_fn
    get_modality_evidence = _get_modality_evidence_fn

    def __init__(self, constraints_path: str = "configs/constraints.yaml"):
        """从约束 YAML 加载道义情态规则并构建查找表。

        参数:
            constraints_path: 约束配置文件路径，须含 ``modality_rules`` 节；
                文件缺失时将回退至内置硬编码规则。

        异常:
            KeyError: YAML 存在但 ``modality_rules`` 格式非法时。
        """
        # 内部查找表：{role_name: {"aux_verbs": [...], "negated": bool}}
        self._modality_rules: dict = {}
        # 快速查找：(aux_lemma, is_negated) -> LegalRole
        self._lookup: dict = {}

        self._load_rules(constraints_path)
