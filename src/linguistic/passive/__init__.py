"""
被动语态检测与论元恢复
===============================================
检测法律文本中的被动构造，从表层句法恢复
语义施事<->受事映射。

子模块：
  _detection:     is_passive、is_passive_loose、get_passive_features
  _restoration:   restore_passive_args、get_active_args
"""

from __future__ import annotations

from src.linguistic.passive._detection import (
    is_passive,
    is_passive_loose,
    get_passive_features,
)
from src.linguistic.passive._restoration import (
    restore_passive_args,
    get_active_args,
)


class PassiveDetector:
    """被动语态检测与语义论元恢复的工具类（全部为静态方法）。

    主要方法:
        is_passive(tree, predicate_idx): 判断是否被动构造。
        restore_passive_args(tree, predicate_idx): 恢复语义施事与受事
            （``obl:agent`` → 施事，``nsubj:pass`` → 受事）。
        get_active_args(tree, predicate_idx): 主动句下的主语/宾语提取。
        get_passive_features(tree, predicate_idx): 返回被动相关 UD 特征摘要。
    """

    is_passive = staticmethod(is_passive)
    is_passive_loose = staticmethod(is_passive_loose)
    restore_passive_args = staticmethod(restore_passive_args)
    get_active_args = staticmethod(get_active_args)
    get_passive_features = staticmethod(get_passive_features)
