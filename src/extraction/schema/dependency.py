"""
LexSpec 数据模型 —— 语言学模型
=============================

Universal Dependencies 依存句法树的结构化表示。
提供 Token, ClauseSpan, ConditionSpan 三个核心数据类。
DependencyTree 已移至 ``._dependency_tree`` 模块。
"""

from typing import Optional, List, Dict

from pydantic import BaseModel, Field

from .enums import ConditionType


class Token(BaseModel):
    """
    A single token from a Universal Dependencies (UD) parse.

    Wraps one row of a CoNLL-U table into a typed Python object.
    The `head` field uses 1-based indexing with 0 = root, following UD conventions.

    All linguistic analysis operates on these Token objects — raw Stanza output
    is never exposed outside the linguistic module.
    """
    index: int = Field(
        description="1-based token index in the sentence"
    )
    text: str = Field(
        description="Surface form of the token as it appears in the sentence"
    )
    lemma: str = Field(
        description="Canonical lemma form, e.g. 'be' for 'was', 'deliver' for 'delivered'"
    )
    upos: str = Field(
        description="Universal POS tag, e.g. 'VERB', 'NOUN', 'ADP', 'CCONJ'"
    )
    xpos: str = Field(
        default="",
        description="Language-specific POS tag (treebank-specific); empty if unavailable"
    )
    deprel: str = Field(
        description="Universal dependency relation label, e.g. 'nsubj', 'obj', 'advcl', 'mark'"
    )
    head: int = Field(
        description="Head token index (1-based); 0 means this token is the syntactic root"
    )
    feats: Dict[str, str] = Field(
        default_factory=dict,
        description="Morphological features as key-value pairs, e.g. {'Tense': 'Past', 'Voice': 'Pass'}"
    )


class ClauseSpan(BaseModel):
    """
    A contiguous or non-contiguous span of tokens representing a syntactic clause.

    Used to represent condition clauses, relative clauses, and other sub-sentential
    units extracted from the dependency tree.

    The `mark_token` field captures the subordinating conjunction or complementizer
    that introduces the clause (e.g., 'if', 'unless', 'that', 'which').
    """
    tokens: List[int] = Field(
        description="Ordered list of 1-based token indices belonging to this span"
    )
    text: str = Field(
        description="Surface text of the span — tokens joined by whitespace"
    )
    deprel: str = Field(
        default="",
        description="Dependency relation of the span's head token to its governor"
    )
    mark_token: Optional[Token] = Field(
        default=None,
        description="Mark word introducing this clause (e.g. 'if', 'unless', 'that'), if present"
    )


class ConditionSpan(ClauseSpan):
    """
    A condition clause span with its semantic type classification.

    Extends ClauseSpan to add the condition_type derived from the mark word
    and the syntactic context (advcl with mark='if' -> TRIGGER, etc.).
    """
    condition_type: ConditionType = Field(
        default=ConditionType.NONE,
        description="Classified condition type based on mark word and syntactic structure"
    )
    mark_text: str = Field(
        default="",
        description="The surface text of the mark word, e.g. 'if', 'unless', 'after'"
    )
