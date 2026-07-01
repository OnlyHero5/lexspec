"""
LexSpec 数据模型 —— 枚举类型定义
=================================

项目所有模块共享的受控词汇表。
"""

from enum import Enum


class LegalRole(str, Enum):
    """
    Legal role of the subject party in a contract clause.

    Determined from the combination of modality (shall/may/must not/agrees to)
    and syntactic structure (active/passive, subject position).
    Used by the UD validator to correct LLM-assigned roles.
    """
    OBLIGOR = "obligor"
    RIGHT_HOLDER = "right_holder"
    PROHIBITED_PARTY = "prohibited_party"
    INDEMNIFYING_PARTY = "indemnifying_party"
    OTHER = "other"


class ConditionType(str, Enum):
    """
    Semantic type of a condition clause attached to a legal action.

    - TEMPORAL:  Time-based condition (e.g., "within 30 days", "after Closing")
    - TRIGGER:   Event-based precondition (e.g., "if Buyer defaults", "upon delivery")
    - EXCEPTION: Carve-out from an obligation (e.g., "unless waived", "except as provided")
    - NONE:      No condition present in the clause
    """
    TEMPORAL = "temporal"
    TRIGGER = "trigger"
    EXCEPTION = "exception"
    NONE = "none"


class ValidationStatus(str, Enum):
    """
    Outcome of the UD-based constraint validator on an LLM prediction.

    - VALID:              Prediction matches UD evidence; no corrections needed.
    - CORRECTED:          Prediction had minor errors that were automatically fixed.
    - REFLEXION_REQUIRED: Prediction has structural errors requiring LLM re-extraction
                          (used as feedback in iterative Reflexion loops).
    """
    VALID = "VALID"
    CORRECTED = "CORRECTED"
    REFLEXION_REQUIRED = "REFLEXION_REQUIRED"


class ErrorCategory(str, Enum):
    """
    Primary error category — the linguistic phenomenon causing the extraction error.

    These map to specific UD syntactic patterns:
    - PASSIVE_VOICE:            nsubj:pass instead of nsubj; object in subject position
    - CONDITIONAL_BOUNDARY:     advcl/mark scope misidentified by LLM
    - RELATIVE_CLAUSE:          acl:relcl embedding confused the extractor
    - LONG_DISTANCE_DEPENDENCY: Dependency path > 3 edges between predicate and argument
    - NEGATION_EXCEPTION:       Negation particle or "except/unless" altered the role
    - OTHER_ERROR:              Catch-all for errors not fitting the above categories
    """
    PASSIVE_VOICE = "passive_voice"
    CONDITIONAL_BOUNDARY = "conditional_boundary"
    RELATIVE_CLAUSE = "relative_clause"
    LONG_DISTANCE_DEPENDENCY = "long_distance_dependency"
    NEGATION_EXCEPTION = "negation_exception"
    OTHER_ERROR = "other"


class FieldErrorType(str, Enum):
    """
    Secondary error type — which field(s) of the LegalTriplet were affected.

    Used to cross-tabulate error rates by field and by linguistic phenomenon.
    """
    SUBJECT = "subject"
    ROLE = "role"
    PREDICATE = "predicate"
    OBJECT = "object"
    CONDITION_OMISSION = "condition_omission"
    CONDITION_OVEREXTENSION = "condition_overextension"
