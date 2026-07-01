"""
Polarity & Modality Detection for Legal Role Classification
============================================================
Detects modal auxiliaries and negation to classify the legal role
of the subject (obligor, right_holder, prohibited_party, etc.).

Submodules:
  _config:     Fallback rules, YAML config loading, lookup table construction
  _negation:   Negation detection (_is_negated, _has_lexical_negation)
  _detector:   Main detection interface (detect, detect_modality, detect_role_with_voice, get_modality_evidence)
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
    """Detect modality and polarity to determine legal role.

    Loads modality rules from configs/constraints.yaml and uses them
    to classify the subject of a legal clause into one of:
      - OBLIGOR: The party with a duty (shall/must + positive)
      - RIGHT_HOLDER: The party with a right (may + positive)
      - PROHIBITED_PARTY: The party under a prohibition (modal + negative)
      - INDEMNIFYING_PARTY: The party with indemnification obligations
        (lexical override for "indemnify")
      - OTHER: Cannot determine (no modal, ambiguous, or edge case)

    The classification is rule-based and deterministic, grounded in
    UD syntactic structure. It provides the "ground truth" against
    which the LLM's role assignment is validated.

    Usage:
        detector = PolarityDetector("configs/constraints.yaml")
        role, polarity = detector.detect(tree, predicate_idx, "deliver")
        # role = LegalRole.OBLIGOR, polarity = "positive"
    """

    # Imported from submodules — bound as instance methods:
    _load_rules = _load_rules_fn
    _build_lookup = _build_lookup_fn
    _is_negated = _is_negated_fn
    _has_lexical_negation = _has_lexical_negation_fn
    detect = _detect_fn
    detect_modality = _detect_modality_fn
    detect_role_with_voice = _detect_role_with_voice_fn
    get_modality_evidence = _get_modality_evidence_fn

    def __init__(self, constraints_path: str = "configs/constraints.yaml"):
        """Initialize with modality rules from constraints config.

        Args:
            constraints_path: Path to constraints YAML config.
                             Falls back to hardcoded rules if unavailable.
        """
        # Internal lookup: {role_name: {"aux_verbs": [...], "negated": bool}}
        self._modality_rules: dict = {}
        # Fast lookup: (aux_lemma, is_negated) -> LegalRole
        self._lookup: dict = {}

        self._load_rules(constraints_path)
