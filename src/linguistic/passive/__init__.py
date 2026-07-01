"""
Passive Voice Detection & Argument Restoration
===============================================
Detects passive constructions in legal text and restores
the semantic agent<->patient mapping from surface syntax.

Submodules:
  _detection:     is_passive, is_passive_loose, get_passive_features
  _restoration:   restore_passive_args, get_active_args
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
    """Detect passive constructions and restore semantic argument mapping.

    This class encapsulates all passive-related logic: detection of passive
    voice, argument restoration (swapping nsubj:pass <-> obl:agent), and
    fallback handling for agentless passives.

    The restored arguments are used by the validator to correct LLM triplets
    where the subject and object have been reversed due to passive voice.

    Usage:
        detector = PassiveDetector()
        if detector.is_passive(tree, pred_idx):
            agent, patient = detector.restore_args(tree, pred_idx)
            # agent -> use as triplet subject
            # patient -> use as triplet object
    """

    is_passive = staticmethod(is_passive)
    is_passive_loose = staticmethod(is_passive_loose)
    restore_passive_args = staticmethod(restore_passive_args)
    get_active_args = staticmethod(get_active_args)
    get_passive_features = staticmethod(get_passive_features)
