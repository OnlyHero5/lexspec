"""
UD Feature Extraction
======================
Extracts syntactic features from UD dependency trees that directly map
to the legal triplet fields (subject, action, condition).

UD Relations Used (with linguistic justification):

    nsubj       — Nominal subject (active voice actor).
    obj         — Direct object (active voice patient).
    nsubj:pass  — Passive nominal subject (surface subject, semantic patient).
    obl:agent   — Oblique agent (by-phrase in passive, semantic actor).
    advcl       — Adverbial clause modifier (condition/trigger candidate).
    mark        — Clause marker (identifies if/unless/provided that).
    acl:relcl   — Relative clause modifier (embedded modification).
    aux         — Auxiliary verb (modality: shall/may/must).
    aux:pass    — Passive auxiliary (be/get in passive construction).
    neg         — Negation marker (not/never, polarity reversal).
    xcomp       — Open clausal complement (control/raising).
    ccomp       — Clausal complement (embedded clause object).

Theory Source:
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
    """Extract syntactic features from UD dependency trees.

    This class provides static/instance methods for navigating UD parse
    trees and extracting the syntactic arguments relevant to legal triplet
    construction. Every method cites its UD relation source and provides
    a legal-domain example.

    These functions are the building blocks for the higher-level analysis
    in PassiveDetector, ConditionExtractor, and PolarityDetector. They
    operate directly on the DependencyTree model and never reference
    Stanza internals.

    Usage:
        tree = parser.parse("Seller shall deliver the Goods.")
        pred = UDFeatureExtractor.find_root_predicate(tree)
        subj = UDFeatureExtractor.find_nsubj(tree, pred.index)
    """

    # ==================================================================
    # Predicate Identification
    # ==================================================================
    find_root_predicate = staticmethod(_find_root_predicate)
    find_all_predicates = staticmethod(_find_all_predicates)

    # ==================================================================
    # Subject Extraction — Active and Passive
    # ==================================================================
    find_nsubj = staticmethod(_find_nsubj)
    find_obj = staticmethod(_find_obj)
    find_nsubj_pass = staticmethod(_find_nsubj_pass)
    find_obl_agent = staticmethod(_find_obl_agent)

    # ==================================================================
    # Clause Relations — advcl, xcomp, ccomp, acl:relcl
    # ==================================================================
    find_advcl_with_mark = staticmethod(_find_advcl_with_mark)
    _matches_marker = staticmethod(_matches_marker)
    _extract_condition_span_text = staticmethod(_extract_condition_span_text)

    # ==================================================================
    # Auxiliary and Negation Detection
    # ==================================================================
    find_aux_verb = staticmethod(_find_aux_verb)
    find_aux_pass = staticmethod(_find_aux_pass)
    find_negation = staticmethod(_find_negation)

    # ==================================================================
    # Relative Clause Detection
    # ==================================================================
    has_acl_relcl = staticmethod(_has_acl_relcl)
    find_acl_relcl_head = staticmethod(_find_acl_relcl_head)

    # ==================================================================
    # Dependency Path and Distance Analysis
    # ==================================================================
    get_dependency_path = staticmethod(_get_dependency_path)
    compute_mean_dependency_distance = staticmethod(_compute_mean_dependency_distance)
    find_long_distance_dependencies = staticmethod(_find_long_distance_dependencies)

    # ==================================================================
    # Coordination Handling
    # ==================================================================
    get_conjuncts = staticmethod(_get_conjuncts)
    get_conjunct_text = staticmethod(_get_conjunct_text)

    # ==================================================================
    # Utility: Find the full span of a noun phrase from its head
    # ==================================================================
    get_noun_phrase_span = staticmethod(_get_noun_phrase_span)

    # ==================================================================
    # Utility: Clause Type Classification
    # ==================================================================
    is_copular_construction = staticmethod(_is_copular_construction)


# ======================================================================
# Module-level convenience functions
# ======================================================================
# These re-export the static methods as module-level functions so that
# callers can do `from src.linguistic.ud_features import find_nsubj`
# without needing to reference the class. Both interfaces are supported.

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
