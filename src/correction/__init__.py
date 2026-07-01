"""
LexSpec Correction (Reflexion) Package
=======================================
Self-correction module that uses structured error feedback to prompt
the LLM to re-extract legal triplets when the UD constraint validator
detects structural errors.

All prompt templates and error hints are loaded from configs/prompts.yaml.
No hardcoded fallbacks.

Package structure:
  - reflexion.py:           ReflexionGenerator class (main orchestrator)
  - prompt_loader.py:        YAML prompt loading (no fallbacks)
  - error_analyzer.py:       Error type determination from validation results
  - response_parser.py:      LLM response parsing for corrections

Public API:
  - ReflexionGenerator: Main class orchestrating the Reflexion loop
"""

from src.correction.reflexion import ReflexionGenerator

__all__ = [
    "ReflexionGenerator",
]
