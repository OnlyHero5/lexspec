"""
LexSpec Linguistic Constraint Module
=====================================
The CORE of the LexSpec project — Universal Dependencies (UD) syntactic
constraints as a post-hoc validator for LLM-extracted legal triplets.

This package provides the complete linguistic analysis pipeline:
  1. StanzaParser:      Wraps Stanza for UD dependency parsing of legal text.
  2. UDFeatureExtractor: Extracts syntactic features from UD trees that map
                         to legal triplet fields (subject, object, condition).
  3. PassiveDetector:   Detects passive voice and restores semantic argument
                         mapping (agent/patient) from surface syntax.
  4. ConditionExtractor: Extracts condition clause boundaries using advcl+mark
                         patterns and classifies by legal-domain taxonomy.
  5. PolarityDetector:  Detects modal auxiliaries and negation to classify
                         the legal role (obligor, right_holder, etc.).
  6. ConstraintValidator: The CORE ALGORITHM — a 7-step validator that compares
                         LLM triplets against UD syntactic structure, producing
                         validated, corrected, or Reflexion-flagged output.

Design Principle:
  All modules consume DependencyTree objects from src.extraction.schema.
  No raw Stanza objects are exposed outside this package. This isolation
  allows the parser backend to be swapped without affecting any downstream
  code.

UD Theory Foundation:
  - de Marneffe & Manning (2014). Stanford Typed Dependencies Manual.
  - Nivre et al. (2020). Universal Dependencies v2 Guidelines.
  - Tesnière (1959). Éléments de syntaxe structurale.

Usage:
    from src.linguistic import StanzaParser, ConstraintValidator
    parser = StanzaParser()
    validator = ConstraintValidator(parser=parser)
    tree = parser.parse("Seller shall deliver the Goods within 30 days.")
    result = validator.validate(triplet, text, tree)
"""

from src.linguistic.stanza_parser import StanzaParser
from src.linguistic.ud_features import UDFeatureExtractor
from src.linguistic.passive_detector import PassiveDetector
from src.linguistic.condition_extractor import ConditionExtractor
from src.linguistic.polarity_detector import PolarityDetector
from src.linguistic.validator import ConstraintValidator

__all__ = [
    "StanzaParser",
    "UDFeatureExtractor",
    "PassiveDetector",
    "ConditionExtractor",
    "PolarityDetector",
    "ConstraintValidator",
]
