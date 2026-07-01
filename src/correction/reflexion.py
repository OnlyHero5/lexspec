"""
Reflexion Feedback Generator for Correcting LLM Extraction Errors
==================================================================
Orchestrates the Reflexion self-correction loop: when the UD constraint
validator returns REFLEXION_REQUIRED status, this module generates a
targeted feedback prompt with linguistic hints, calls the LLM for
re-generation, and returns the corrected triplet.

All prompt templates and error hints are loaded from configs/prompts.yaml.
No hardcoded fallbacks — a missing or incomplete config is an error.

Usage:
    from src.extraction.client import LLMClient
    from src.correction.reflexion import ReflexionGenerator

    client = LLMClient(config)
    reflexion = ReflexionGenerator(client)
    corrected = reflexion.correct(clause_text, validation_result)
"""

from __future__ import annotations

import json
from typing import Optional, TYPE_CHECKING

import yaml

from src.extraction.schema import (
    LegalTriplet,
    ValidationResult,
)
from src.correction.prompt_loader import load_reflexion_prompt, load_system_prompt
from src.correction.error_analyzer import determine_error_types
from src.correction.response_parser import parse_llm_response
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.extraction.client import LLMClient

logger = get_logger(__name__)


def _load_error_hints(prompts_path: str) -> dict[str, str]:
    """Load error hints from the prompts YAML config.

    Reads ``reflexion.error_hints`` from the configuration file.

    Args:
        prompts_path: Path to the prompts YAML file.

    Returns:
        Dict mapping error type keys to hint strings.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If the error_hints section is missing or incomplete.
    """
    with open(prompts_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    reflexion_section = config.get("reflexion", {})
    hints = reflexion_section.get("error_hints", {})

    if not hints:
        raise KeyError(
            f"No 'reflexion.error_hints' in '{prompts_path}'. "
            "The prompts YAML must define error hints for all categories."
        )

    if "default" not in hints:
        raise KeyError(
            f"'default' error hint missing from '{prompts_path}'. "
            "A fallback hint with key 'default' is required."
        )

    return hints


class ReflexionGenerator:
    """Generate feedback prompts and orchestrate Reflexion re-generation.

    When the constraint validator returns REFLEXION_REQUIRED status,
    this module:
    1. Analyzes the validation result to determine error types
    2. Maps errors to specific hints from the YAML config
    3. Formats the feedback prompt with all context
    4. Calls the LLM for re-generation
    5. Returns the corrected triplet

    Iteration limit: max 1 iteration.
    """

    def __init__(
        self,
        client: "LLMClient",
        prompts_path: str = "configs/prompts.yaml",
    ) -> None:
        """Initialize the Reflexion generator with an LLM client.

        Loads the Reflexion prompt template and error hints from the
        YAML prompts config. No fallback — raises on missing config.

        Args:
            client: LLM client for re-generation calls.
            prompts_path: Path to the prompts YAML configuration file.

        Raises:
            FileNotFoundError: If prompts_path does not exist.
            KeyError: If required YAML sections are missing.
        """
        self.client = client
        self.reflexion_prompt_template = load_reflexion_prompt(prompts_path)
        self.system_prompt = load_system_prompt(prompts_path)
        self.error_hints = _load_error_hints(prompts_path)

        logger.info(
            "ReflexionGenerator initialized (prompts_path=%s, error_hints=%d)",
            prompts_path, len(self.error_hints),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_feedback(
        self,
        validation_result: ValidationResult,
        clause: str = "",
    ) -> str:
        """Generate a feedback prompt from a validation result.

        Maps validation errors to specific linguistic hints and fills
        the feedback template with all relevant context.

        Args:
            validation_result: Result from ConstraintValidator.validate().
            clause: The original clause text.

        Returns:
            Formatted feedback prompt string ready for LLM consumption.
        """
        error_types = determine_error_types(validation_result)
        primary_error = error_types[0] if error_types else "default"
        logger.debug(
            "Identified error types: %s -> primary=%s", error_types, primary_error
        )

        specific_hint = self.error_hints.get(
            primary_error, self.error_hints["default"]
        )

        clause_text = clause if clause else self._derive_clause_text(validation_result)

        prediction_json = json.dumps(
            validation_result.original_prediction.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        )

        evidence_json = json.dumps(
            validation_result.linguistic_evidence.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        )

        feedback = self.reflexion_prompt_template.format(
            error_type=primary_error,
            text=clause_text,
            prediction=prediction_json,
            linguistic_evidence=evidence_json,
            specific_hint=specific_hint,
        )

        logger.debug(
            "Generated feedback prompt (%d chars) for error type=%s",
            len(feedback), primary_error,
        )
        return feedback

    def correct(
        self,
        clause: str,
        validation_result: ValidationResult,
    ) -> Optional[LegalTriplet]:
        """Run one iteration of Reflexion correction.

        Generates a feedback prompt from the validation result, sends it
        to the LLM for re-extraction, and parses the response into a
        corrected LegalTriplet.

        Args:
            clause: Original clause text that was extracted.
            validation_result: Validation result indicating what went wrong.

        Returns:
            Corrected LegalTriplet if the LLM produced valid output,
            or None if parsing failed.
        """
        feedback_prompt = self.generate_feedback(validation_result, clause)

        logger.info("Calling LLM for Reflexion correction on clause: %.80s...", clause)
        response = self.client.complete(
            system_prompt=self.system_prompt,
            user_prompt=feedback_prompt,
        )

        logger.debug("LLM Reflexion response length: %d chars", len(response))

        corrected_triplet = parse_llm_response(response)

        if corrected_triplet is not None:
            logger.info(
                "Reflexion correction succeeded: subject=%s, predicate=%s",
                corrected_triplet.subject.text,
                corrected_triplet.action.predicate,
            )
        else:
            logger.warning(
                "Reflexion correction failed: could not parse LLM response into LegalTriplet"
            )

        return corrected_triplet

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_clause_text(validation_result: ValidationResult) -> str:
        """Best-effort derivation of clause text from the validation result."""
        pred = validation_result.original_prediction
        if pred.action.predicate:
            subject_text = pred.subject.text or "unknown party"
            return (
                f"Clause involving '{subject_text}' performing "
                f"action '{pred.action.predicate}'"
            )
        return ""
