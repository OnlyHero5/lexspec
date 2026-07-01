"""
UD 特征提取
======================
从 UD 依存树提取直接映射到法律三元组字段
（主语、动作、条件）的句法特征。

使用的 UD 关系（含语言学依据）：

    nsubj       —— 名词性主语（主动语态施事）。
    obj         —— 直接宾语（主动语态受事）。
    nsubj:pass  —— 被动名词性主语（表层主语，语义受事）。
    obl:agent   —— 斜格施事（被动中的 by 短语，语义行为者）。
    advcl       —— 状语从句修饰语（条件/触发候选）。
    mark        —— 从句标记（识别 if/unless/provided that）。
    acl:relcl   —— 关系从句修饰语（嵌入式修饰）。
    aux         —— 助动词（情态：shall/may/must）。
    aux:pass    —— 被动助动词（被动构造中的 be/get）。
    neg         —— 否定标记（not/never，极性反转）。
    xcomp       —— 开放从句补语（控制/提升）。
    ccomp       —— 从句补语（嵌入式从句宾语）。

理论来源：
  - Tesnière (1959). Éléments de syntaxe structurale.
  - de Marneffe & Manning (2014). Stanford Typed Dependencies Manual.
  - Universal Dependencies Guidelines v2. https://universaldependencies.org/u/dep/
  - Nivre et al. (2020). Universal Dependencies v2: An Ever-growing Multilingual
    Treebank Collection. LREC 2020.
"""

from __future__ import annotations

from src.linguistic.ud_features._predicate import (
    find_root_predicate as _find_root_predicate,
    find_all_predicates as _find_all_predicates,
)
from src.linguistic.ud_features._arguments import (
    find_nsubj as _find_nsubj,
    find_obj as _find_obj,
    find_nsubj_pass as _find_nsubj_pass,
    find_obl_agent as _find_obl_agent,
)
from src.linguistic.ud_features._clause import (
    find_advcl_with_mark as _find_advcl_with_mark,
    _matches_marker,
    _extract_condition_span_text,
)
from src.linguistic.ud_features._modifiers import (
    find_aux_verb as _find_aux_verb,
    find_aux_pass as _find_aux_pass,
    find_negation as _find_negation,
    has_acl_relcl as _has_acl_relcl,
    find_acl_relcl_head as _find_acl_relcl_head,
    get_noun_phrase_span as _get_noun_phrase_span,
    is_copular_construction as _is_copular_construction,
)
from src.linguistic.ud_features._topology import (
    get_dependency_path as _get_dependency_path,
    compute_mean_dependency_distance as _compute_mean_dependency_distance,
    find_long_distance_dependencies as _find_long_distance_dependencies,
    get_conjuncts as _get_conjuncts,
    get_conjunct_text as _get_conjunct_text,
)


class UDFeatureExtractor:
    """从 UD 依存树提取与法律三元组相关的句法特征（静态方法集合）。

    方法分组:
        谓词: ``find_root_predicate``、``find_all_predicates``
        论元: ``find_nsubj``、``find_obj``、``find_nsubj_pass``、``find_obl_agent``
        从句: ``find_advcl_with_mark``
        修饰: ``find_aux_verb``、``find_aux_pass``、``find_negation``、``has_acl_relcl``
        拓扑: ``get_dependency_path``、``compute_mean_dependency_distance``
        工具: ``get_noun_phrase_span``、``get_conjuncts``、``is_copular_construction``

    各静态方法的 ``参数``/``返回`` 说明见对应子模块实现
    （``_predicate.py``、``_arguments.py`` 等）。
    """

    # ==================================================================
    # 谓词识别
    # ==================================================================
    find_root_predicate = staticmethod(_find_root_predicate)
    find_all_predicates = staticmethod(_find_all_predicates)

    # ==================================================================
    # 主语提取 —— 主动与被动
    # ==================================================================
    find_nsubj = staticmethod(_find_nsubj)
    find_obj = staticmethod(_find_obj)
    find_nsubj_pass = staticmethod(_find_nsubj_pass)
    find_obl_agent = staticmethod(_find_obl_agent)

    # ==================================================================
    # 从句关系 —— advcl、xcomp、ccomp、acl:relcl
    # ==================================================================
    find_advcl_with_mark = staticmethod(_find_advcl_with_mark)
    _matches_marker = staticmethod(_matches_marker)
    _extract_condition_span_text = staticmethod(_extract_condition_span_text)

    # ==================================================================
    # 助动词与否定检测
    # ==================================================================
    find_aux_verb = staticmethod(_find_aux_verb)
    find_aux_pass = staticmethod(_find_aux_pass)
    find_negation = staticmethod(_find_negation)

    # ==================================================================
    # 关系从句检测
    # ==================================================================
    has_acl_relcl = staticmethod(_has_acl_relcl)
    find_acl_relcl_head = staticmethod(_find_acl_relcl_head)

    # ==================================================================
    # 依存路径与距离分析
    # ==================================================================
    get_dependency_path = staticmethod(_get_dependency_path)
    compute_mean_dependency_distance = staticmethod(_compute_mean_dependency_distance)
    find_long_distance_dependencies = staticmethod(_find_long_distance_dependencies)

    # ==================================================================
    # 并列处理
    # ==================================================================
    get_conjuncts = staticmethod(_get_conjuncts)
    get_conjunct_text = staticmethod(_get_conjunct_text)

    # ==================================================================
    # 工具：从中心词获取名词短语完整跨度
    # ==================================================================
    get_noun_phrase_span = staticmethod(_get_noun_phrase_span)

    # ==================================================================
    # 工具：从句类型分类
    # ==================================================================
    is_copular_construction = staticmethod(_is_copular_construction)


# ======================================================================
# 模块级便捷函数
# ======================================================================
# 将静态方法再导出为模块级函数，使调用方可
# `from src.linguistic.ud_features import find_nsubj`
# 而无需引用类。两种接口均支持。

find_root_predicate = UDFeatureExtractor.find_root_predicate
find_nsubj = UDFeatureExtractor.find_nsubj
find_obj = UDFeatureExtractor.find_obj
find_nsubj_pass = UDFeatureExtractor.find_nsubj_pass
find_obl_agent = UDFeatureExtractor.find_obl_agent
find_advcl_with_mark = UDFeatureExtractor.find_advcl_with_mark
find_aux_verb = UDFeatureExtractor.find_aux_verb
find_aux_pass = UDFeatureExtractor.find_aux_pass
find_negation = UDFeatureExtractor.find_negation
has_acl_relcl = UDFeatureExtractor.has_acl_relcl
get_dependency_path = UDFeatureExtractor.get_dependency_path
compute_mean_dependency_distance = UDFeatureExtractor.compute_mean_dependency_distance

__all__ = [
    "UDFeatureExtractor",
    "find_root_predicate",
    "find_nsubj",
    "find_obj",
    "find_nsubj_pass",
    "find_obl_agent",
    "find_advcl_with_mark",
    "find_aux_verb",
    "find_aux_pass",
    "find_negation",
    "has_acl_relcl",
    "get_dependency_path",
    "compute_mean_dependency_distance",
]
