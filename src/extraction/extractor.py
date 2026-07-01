"""
法律三元组抽取器 —— 基于大语言模型的结构化信息抽取
====================================================

从合同条款中抽取 (subject, action, condition) 三元组，
使用结构化提示词 + JSON Schema 约束的大语言模型调用。

架构:
  1. 提示词加载 —— 从 configs/prompts.yaml 加载模板
  2. 大语言模型调用 —— LLMClient 发送格式化提示词
  3. 响应解析 —— 多策略 JSON 提取 + 回退
  4. 校验与规范化 —— Pydantic 模型校验 + 枚举值规范化

使用示例::

    from src.extraction.client import LLMClient, ClientConfig
    from src.extraction.extractor import LegalTripletExtractor

    config = ClientConfig(base_url="http://localhost:8080/v1", model="qwen3.5-9b")
    client = LLMClient(config)
    extractor = LegalTripletExtractor(client)
    triplet = extractor.extract("Seller shall deliver the goods within 30 days.")
"""

from __future__ import annotations

from typing import List

from src.extraction.schema import LegalTriplet
from src.extraction.client import LLMClient
from src.utils.logging import get_logger

from ._prompts import load_prompts
from ._parsing import parse_llm_response
from ._validation import (
    validate_and_normalize_triplet,
    build_fallback_triplet,
)

logger = get_logger(__name__)


# =============================================================================
# LegalTripletExtractor
# =============================================================================


class LegalTripletExtractor:
    """Extract legal action triplets from contract clauses via LLM.

    Loads prompt templates from ``configs/prompts.yaml``.  Raises an error
    hard-coded defaults), formats them with the target clause, sends them
    to the LLM, and parses the structured JSON response into validated
    ``LegalTriplet`` Pydantic models.

    The extractor is designed to be **robust against LLM failures**: if the
    model returns malformed JSON, incomplete fields, or refuses to answer,
    a fallback triplet is constructed so that batch processing continues
    uninterrupted.  Failed extractions are logged with full context for
    post-hoc debugging.

    Attributes:
        client:             The configured ``LLMClient`` instance.
        system_prompt:      The system prompt template (loaded or default).
        user_prompt_template:  The user prompt template with ``{clause}`` or
                               ``{sentence}`` placeholder.
        prompts_source:     Human-readable description of where the prompts
                            were loaded from (for logging).
    """

    def __init__(
        self,
        client: LLMClient,
        prompts_path: str = "configs/prompts.yaml",
    ):
        """Initialize the extractor with an LLM client and prompt configuration.

        Loads prompt templates from ``prompts_path`` (YAML).  Raises an
        error if the configuration file is missing or incomplete — no
        silent fallback to hardcoded defaults.

        Args:
            client:        A configured ``LLMClient`` instance pointing at
                           the llama.cpp server.
            prompts_path:  Path to the ``prompts.yaml`` configuration file.
                           Defaults to ``configs/prompts.yaml``.

        Raises:
            FileNotFoundError: If prompts_path does not exist.
            KeyError: If the YAML is missing required keys.
            yaml.YAMLError: If the YAML is malformed.
        """
        self.client = client

        loaded = load_prompts(prompts_path)
        self.system_prompt = loaded["system"]
        self.user_prompt_template = loaded["user"]
        self.prompts_source = loaded["source"]

        logger.info(
            "LegalTripletExtractor initialized (prompts_source=%s, model=%s)",
            self.prompts_source,
            client.config.model,
        )

    # ---------------------------------------------------------------------
    # Main Extraction Entry Points
    # ---------------------------------------------------------------------

    def extract(self, clause: str) -> LegalTriplet:
        """Extract a legal triplet from a single contract clause.

        This is the main entry point.  It performs the following steps:

        1. **Format the prompt** — Inserts the clause text into the user
           prompt template (handles both ``{clause}`` and ``{sentence}``
           placeholders depending on which template is loaded).
        2. **Call the LLM** — Sends the formatted prompt via
           ``LLMClient.complete_structured()``.
        3. **Parse the response** — Robust JSON parsing with multiple
           fallback strategies.
        4. **Validate** — Checks the parsed dict against the
           ``LegalTriplet`` Pydantic model.
        5. **Normalize** — Trims whitespace, handles empty conditions,
           ensures enum values are valid.

        Args:
            clause:  A contract clause string, e.g.,
                     ``"Seller shall deliver the goods within 30 days."``

        Returns:
            A validated ``LegalTriplet`` with extracted subject, action,
            and condition.

        Raises:
            ValueError:  If the LLM response cannot be parsed or validated
                         after all fallback attempts.  This is rare — the
                         method is designed to return a fallback triplet in
                         most failure modes.
        """
        # --- Step 1: Format the user prompt with the clause text ---
        # We need to handle two possible placeholder formats:
        #   - {clause}   — the DEFAULT template uses this
        #   - {sentence} — the YAML config (prompts.yaml) uses this
        # The formatter picks the correct key based on which placeholder
        # appears in the template.
        user_prompt = self._format_prompt(clause)

        logger.debug(
            "Extracting triplet from clause (len=%d): %s",
            len(clause),
            clause[:120] + ("..." if len(clause) > 120 else ""),
        )

        # --- Step 2: Call the LLM ---
        try:
            raw_response = self.client.complete_structured(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
            )
            logger.debug(
                "LLM raw response (len=%d): %s",
                len(raw_response),
                raw_response[:200] + ("..." if len(raw_response) > 200 else ""),
            )
        except RuntimeError as exc:
            # The LLM client exhausted all retries.  Build a fallback triplet
            # so the caller can continue processing other clauses.
            logger.error(
                "LLM request failed for clause: %s",
                exc,
            )
            return build_fallback_triplet(
                clause=clause,
                error=f"LLM request failed: {exc}",
            )

        # --- Step 3: Parse the JSON response ---
        parsed = parse_llm_response(raw_response)

        # If parsing returned an empty dict, the response was completely
        # unparseable.  Build a fallback triplet.
        if not parsed:
            logger.warning(
                "Could not parse LLM response into JSON dict. "
                "Raw response: %s",
                raw_response[:200],
            )
            return build_fallback_triplet(
                clause=clause,
                error="JSON parse failure: no valid JSON found in response",
            )

        # --- Step 4 & 5: Validate and normalize ---
        try:
            triplet = validate_and_normalize_triplet(parsed, clause)
            logger.debug(
                "Extracted triplet: subject=(text=%r, role=%s), "
                "action=(predicate=%r, object=%r), "
                "condition=(text=%r, type=%s)",
                triplet.subject.text,
                triplet.subject.role.value,
                triplet.action.predicate,
                triplet.action.object,
                triplet.condition.text,
                triplet.condition.type.value,
            )
            return triplet

        except (ValueError, TypeError, KeyError) as exc:
            # Validation against the Pydantic model failed.  This can happen
            # when the LLM returns valid JSON that does not conform to the
            # expected schema (e.g., missing required fields, wrong types).
            logger.warning(
                "Triplet validation failed for clause: %s. Parsed: %s",
                exc,
                parsed,
            )
            return build_fallback_triplet(
                clause=clause,
                error=f"Validation failure: {exc}",
            )

    def extract_batch(self, clauses: List[str]) -> List[LegalTriplet]:
        """Extract triplets from multiple clauses sequentially.

        Each clause is processed independently via ``self.extract()``.
        Errors propagate immediately — no silent catch-all.

        Args:
            clauses:  List of contract clause strings.

        Returns:
            List of ``LegalTriplet`` objects, one per input clause, in the
            same order.

        Raises:
            RuntimeError: If the LLM request fails.
            ValueError: If parsing or validation fails.
        """
        results: List[LegalTriplet] = []

        for clause in clauses:
            triplet = self.extract(clause)
            results.append(triplet)

        logger.info(
            "Batch extraction complete: %d/%d clauses processed",
            len(results),
            len(clauses),
        )
        return results

    # ---------------------------------------------------------------------
    # Prompt Formatting
    # ---------------------------------------------------------------------

    def _format_prompt(self, clause: str) -> str:
        """Insert the clause text into the user prompt template.

        Handles both ``{clause}`` (default template) and ``{sentence}``
        (prompts.yaml template) placeholders automatically by checking
        which placeholder appears in the loaded template string.

        The template is formatted using Python's ``str.format()``.  The
        ``{{`` and ``}}`` escape sequences in the default template become
        literal ``{`` and ``}`` characters in the final prompt — these
        are part of the JSON example shown to the LLM.

        Args:
            clause:  The contract clause text to insert.

        Returns:
            The fully formatted prompt string ready for the LLM.
        """
        template = self.user_prompt_template

        # --- Determine which placeholder to use ---
        # The YAML config uses {sentence}, the default uses {clause}.
        # We detect the format from the template string itself so that
        # the extractor works regardless of which prompt source is active.
        if "{sentence}" in template:
            return template.format(sentence=clause)
        elif "{clause}" in template:
            return template.format(clause=clause)
        else:
            # If the template has no recognized placeholder, we append the
            # clause on a new line as a last resort.  This should not happen
            # with the standard templates but guards against misconfigured
            # custom prompts.
            logger.warning(
                "Prompt template has no {clause} or {sentence} placeholder. "
                "Appending clause to the end."
            )
            return template + "\n\n" + clause
