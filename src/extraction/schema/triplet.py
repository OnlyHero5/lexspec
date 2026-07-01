"""
LexSpec 数据模型 —— 核心三元组模型
===================================

法律动作框架的核心数据结构：Subject, Action, Condition, LegalTriplet。
"""

from pydantic import BaseModel, Field

from .enums import LegalRole, ConditionType


class Subject(BaseModel):
    """
    Legal action subject — the party who performs, owes, or holds the
    obligation, right, restriction, or indemnity.

    Examples:
      - "Seller" as OBLIGOR:  "Seller shall deliver the Goods"
      - "Buyer" as RIGHT_HOLDER: "Buyer may terminate this Agreement"
      - "the Company" as PROHIBITED_PARTY: "the Company shall not assign..."
    """
    text: str = Field(
        description="Party name/identifier, e.g. 'Seller', 'the Buyer', 'each Indemnifying Party'"
    )
    role: LegalRole = Field(
        description="Legal role classification derived from modality and syntactic position"
    )


class Action(BaseModel):
    """
    Core legal action — the predicate (main verb in lemma form) and its
    direct object.

    The predicate is always in lemma form (e.g., 'deliver' not 'delivered')
    for consistent comparison across tense/aspect variations.
    """
    predicate: str = Field(
        description="Main verb in lemma form, e.g. 'deliver' not 'delivered', 'indemnify' not 'indemnifies'"
    )
    object: str = Field(
        description="What is being acted upon, e.g. 'the Goods', 'the Agreement', 'all Losses'"
    )


class Condition(BaseModel):
    """
    Condition clause — a precondition, trigger event, temporal limitation,
    or exception that qualifies the legal action.

    When no condition is present, `text` defaults to "" and `type` to NONE.
    """
    text: str = Field(
        default="",
        description="Full condition clause text; empty string if no condition is present"
    )
    type: ConditionType = Field(
        default=ConditionType.NONE,
        description="Semantic classification of the condition clause"
    )


class LegalTriplet(BaseModel):
    """
    Complete legal action frame extracted from a single contract clause.

    This is the fundamental unit of analysis in the LexSpec pipeline.
    Every clause is decomposed into exactly one triplet representing:
      WHO (Subject) does WHAT (Action) under WHICH circumstances (Condition).

    Example:
      "Seller shall deliver the Goods within 30 days of Closing"
      -> Subject("Seller", OBLIGOR), Action("deliver", "the Goods"), Condition("within 30 days...", TEMPORAL)
    """
    subject: Subject = Field(
        description="Legal action subject — the party who performs/receives the action"
    )
    action: Action = Field(
        description="Core legal action — predicate + object"
    )
    condition: Condition = Field(
        default_factory=Condition,
        description="Condition clause qualifying the action; may be empty/NONE",
    )
