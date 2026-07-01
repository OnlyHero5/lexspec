"""
LLM-Based Contract Clause Annotator for Gold-Standard Test Set Construction
============================================================================
Uses LLMs (Qwen3.6 27B or Gemma4 31B) to annotate contract clauses with
legal triplets. Each annotation model independently annotates every clause.

**Key constraint**: The annotation models are COMPLETELY ISOLATED from the
experiment model (Qwen3.5 9B) used in Phases 2-3. This module is used ONLY
in Phase 1 (test set construction) and never appears in Phases 2-3.

Usage::

    from src.extraction.client import LLMClient, ClientConfig
    from src.annotation.llm_annotator import LLMAnnotator

    client = LLMClient(ClientConfig(
        base_url="http://10.0.16.254:8080/v1",
        model="gemma-4-31B-it-Q8_0.gguf",
        timeout=1200,
    ))
    annotator = LLMAnnotator(client)
    triplet = annotator.annotate("Seller shall deliver the Goods.")
"""

from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING

from src.extraction.schema import LegalTriplet
from src.utils.logging import get_logger

from src.annotation.prompts import load_annotation_prompts
from src.annotation.response_parser import parse_llm_response

if TYPE_CHECKING:
    from src.extraction.client import LLMClient

logger = get_logger(__name__)


class LLMAnnotator:
    """Annotate contract clauses with legal triplets using an LLM.

    Supports dual-model annotation -- each model annotates independently,
    and results are reconciled through field-level voting consensus
    (see src/annotation/consensus.py).

    Prompts are loaded from configs/prompts.yaml (see annotation.system
    and annotation.user fields). Raises if the config is unavailable.

    Attributes:
        client: The LLM client configured for the annotation model.
        system_prompt: System prompt for annotation calls.
        user_template: User prompt template containing {sentence} placeholder.
    """

    def __init__(
        self,
        client: "LLMClient",
        prompts_path: str = "configs/prompts.yaml",
    ) -> None:
        """Initialize the annotator.

        Loads annotation prompt templates from the YAML config file.
        Raises an error if the file is missing or malformed.

        Args:
            client: LLM client configured for the annotation model
                    (e.g., Qwen3.6 27B or Gemma4 31B).
            prompts_path: Path to the prompts YAML configuration file.
                          Defaults to "configs/prompts.yaml".
        """
        self.client = client
        self.system_prompt, self.user_template = load_annotation_prompts(
            prompts_path
        )
        logger.info(
            "LLMAnnotator initialized (prompts_path=%s)", prompts_path
        )
        logger.debug(
            "System prompt length: %d chars, user template length: %d chars",
            len(self.system_prompt),
            len(self.user_template),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(self, clause: str) -> LegalTriplet:
        """Annotate a single contract clause.

        Injects the clause text into the prompt template, sends it to the
        LLM, and parses the response into a LegalTriplet.

        Args:
            clause: The contract clause text to annotate.

        Returns:
            A LegalTriplet with the extracted subject, action, and condition.

        Raises:
            ValueError: If the LLM response cannot be parsed into a valid
                        LegalTriplet.
        """
        logger.debug("Annotating clause: %.80s...", clause)

        # Step 1: Inject clause text into the user prompt template.
        user_prompt = self.user_template.format(sentence=clause)

        # Step 2: Call the LLM.
        try:
            response = self.client.complete_structured(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            logger.error("LLM call failed during annotation: %s", exc)
            raise ValueError(
                f"LLM annotation call failed, clause: {clause[:100]}..."
            ) from exc

        logger.debug("LLM annotation response length: %d chars", len(response))

        # Step 3: Parse the response into a LegalTriplet.
        triplet = parse_llm_response(response)

        if triplet is None:
            logger.error(
                "Could not parse LLM annotation response. First 200 chars: %.200s",
                response,
            )
            raise ValueError(
                f"Cannot parse annotation response, clause: {clause[:100]}..."
            )

        if not self._is_substantive_triplet(triplet):
            logger.error(
                "LLM returned an empty or invalid triplet for clause: %.80s...",
                clause,
            )
            raise ValueError(
                f"Annotation triplet is empty or invalid, clause: {clause[:100]}..."
            )

        logger.debug(
            "Annotation complete: subject=%s (role=%s), predicate=%s, object=%s, condition=%s",
            triplet.subject.text,
            triplet.subject.role.value,
            triplet.action.predicate,
            triplet.action.object,
            triplet.condition.type.value if triplet.condition.text else "none",
        )

        return triplet

    @staticmethod
    def _is_substantive_triplet(triplet: LegalTriplet) -> bool:
        """Check that a triplet contains substantive content.

        Rejects shells where parsing succeeded but all core fields are empty.

        Rules:
          - Operative clause: needs at least subject.text + action.predicate
          - Definitional clause ("X means Y"): needs predicate + object even
            without a party.

        Args:
            triplet: The LegalTriplet to validate.

        Returns:
            True if the triplet has meaningful content, False otherwise.
        """
        has_subject = bool(triplet.subject.text.strip())
        has_predicate = bool(triplet.action.predicate.strip())
        has_object = bool(triplet.action.object.strip())
        if has_subject and has_predicate:
            return True
        if has_predicate and has_object:
            return True
        return False

    def annotate_batch(
        self,
        clauses: List[str],
        show_progress: bool = True,
    ) -> List[LegalTriplet]:
        """Annotate a batch of contract clauses.

        Each clause is annotated independently. Displays a tqdm progress bar
        by default. Single-clause failures do not abort the batch; failed
        entries are filled with a placeholder triplet so the output list
        always has the same length as the input.

        Args:
            clauses: List of clause texts to annotate.
            show_progress: Whether to show a tqdm progress bar. Default True.

        Returns:
            List of LegalTriplets, same length as the input. Failed entries
            have subject.text="ERROR".
        """
        iterator = clauses
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(clauses, desc="Annotating", unit="clause")
            except ImportError:
                logger.debug("tqdm not installed -- running without progress bar")

        results: List[LegalTriplet] = []
        for i, clause in enumerate(iterator):
            try:
                triplet = self.annotate(clause)
                results.append(triplet)
            except Exception as exc:
                logger.error(
                    "Annotation failed for item %d: %s. Clause: %.80s...",
                    i, exc, clause,
                )
                from src.extraction.schema import Subject, Action, Condition
                from src.extraction.schema import LegalRole, ConditionType

                placeholder = LegalTriplet(
                    subject=Subject(text="ERROR", role=LegalRole.OTHER),
                    action=Action(predicate="", object=""),
                    condition=Condition(text="", type=ConditionType.NONE),
                )
                results.append(placeholder)

        logger.info(
            "Batch annotation complete: %d/%d succeeded",
            sum(1 for t in results if t.subject.text != "ERROR"),
            len(clauses),
        )
        return results
