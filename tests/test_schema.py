"""src.extraction.schema 中核心 Pydantic 模型的基础测试。"""

import pytest

from src.extraction.schema import (
    LegalTriplet,
    Subject,
    Action,
    Condition,
    DependencyTree,
    Token,
    LegalRole,
    ConditionType,
)


class TestSubject:
    """Subject 模型的测试。"""

    def test_creation_with_valid_role(self):
        subj = Subject(text="Seller", role=LegalRole.OBLIGOR)
        assert subj.text == "Seller"
        assert subj.role == LegalRole.OBLIGOR

    def test_role_is_enum(self):
        subj = Subject(text="Buyer", role=LegalRole.RIGHT_HOLDER)
        assert isinstance(subj.role, LegalRole)

    def test_field_descriptions_present(self):
        assert Subject.model_fields["text"].description is not None
        assert Subject.model_fields["role"].description is not None


class TestAction:
    """Action 模型的测试。"""

    def test_creation(self):
        action = Action(predicate="deliver", object="the Goods")
        assert action.predicate == "deliver"
        assert action.object == "the Goods"

    def test_field_descriptions_present(self):
        assert Action.model_fields["predicate"].description is not None
        assert Action.model_fields["object"].description is not None


class TestCondition:
    """Condition 模型的测试。"""

    def test_defaults_to_none(self):
        cond = Condition()
        assert cond.text == ""
        assert cond.type == ConditionType.NONE

    def test_creation_with_values(self):
        cond = Condition(text="within 30 days", type=ConditionType.TEMPORAL)
        assert cond.text == "within 30 days"
        assert cond.type == ConditionType.TEMPORAL


class TestToken:
    """Token 模型的测试。"""

    def test_creation(self):
        tok = Token(
            index=1,
            text="Seller",
            lemma="seller",
            upos="NOUN",
            deprel="nsubj",
            head=3,
        )
        assert tok.index == 1
        assert tok.text == "Seller"
        assert tok.lemma == "seller"
        assert tok.upos == "NOUN"
        assert tok.deprel == "nsubj"
        assert tok.head == 3

    def test_defaults(self):
        tok = Token(
            index=2,
            text="shall",
            lemma="shall",
            upos="AUX",
            deprel="aux",
            head=3,
        )
        assert tok.xpos == ""
        assert tok.feats == {}


class TestDependencyTree:
    """DependencyTree 模型的测试。"""

    @pytest.fixture
    def simple_tree(self):
        """构建用于测试的最小依存树。"""
        tokens = [
            Token(index=1, text="Seller", lemma="seller", upos="NOUN", deprel="nsubj", head=3),
            Token(index=2, text="shall", lemma="shall", upos="AUX", deprel="aux", head=3),
            Token(index=3, text="deliver", lemma="deliver", upos="VERB", deprel="root", head=0),
            Token(index=4, text="Goods", lemma="goods", upos="NOUN", deprel="obj", head=3),
        ]
        return DependencyTree(text="Seller shall deliver Goods", tokens=tokens)

    def test_creation(self, simple_tree):
        assert simple_tree.text == "Seller shall deliver Goods"
        assert simple_tree.token_count == 4

    def test_root_index(self, simple_tree):
        assert simple_tree.root_index == 3

    def test_get_token(self, simple_tree):
        tok = simple_tree.get_token(1)
        assert tok is not None
        assert tok.text == "Seller"

    def test_get_token_missing(self, simple_tree):
        assert simple_tree.get_token(99) is None

    def test_get_children(self, simple_tree):
        children = simple_tree.get_children(3)
        child_texts = {t.text for t in children}
        assert "Seller" in child_texts
        assert "shall" in child_texts
        assert "Goods" in child_texts

    def test_get_children_filtered(self, simple_tree):
        subjects = simple_tree.get_children(3, deprel="nsubj")
        assert len(subjects) == 1
        assert subjects[0].text == "Seller"

    def test_get_head(self, simple_tree):
        head = simple_tree.get_head(1)  # "Seller"（索引 1）的中心词
        assert head is not None
        assert head.index == 3
        assert head.text == "deliver"

    def test_get_head_of_root_returns_none(self, simple_tree):
        assert simple_tree.get_head(3) is None  # 根节点没有中心词

    def test_get_path_to_root(self, simple_tree):
        path = simple_tree.get_path_to_root(1)  # 从 "Seller" 向上
        assert len(path) == 2
        assert path[0].text == "Seller"
        assert path[1].text == "deliver"

    def test_find_tokens_by_upos(self, simple_tree):
        verbs = simple_tree.find_tokens_by_upos("VERB")
        assert len(verbs) == 1
        assert verbs[0].text == "deliver"

    def test_find_tokens_by_deprel(self, simple_tree):
        subjects = simple_tree.find_tokens_by_deprel("nsubj")
        assert len(subjects) == 1
        assert subjects[0].text == "Seller"

    def test_has_deprel(self, simple_tree):
        assert simple_tree.has_deprel("nsubj") is True
        assert simple_tree.has_deprel("iobj") is False

    def test_get_dependency_distance(self, simple_tree):
        distance = simple_tree.get_dependency_distance(1, 3)
        assert distance == 2

    def test_get_subtree_span(self, simple_tree):
        span = simple_tree.get_subtree_span(3)  # 根节点下的整棵子树
        assert "Seller" in span.text
        assert "deliver" in span.text
        assert "Goods" in span.text

    def test_get_subtree_tokens(self, simple_tree):
        tokens = simple_tree.get_subtree_tokens(3)
        assert len(tokens) == 4


class TestLegalTriplet:
    """LegalTriplet 模型的测试——核心抽取输出。"""

    @pytest.fixture
    def sample_triplet(self):
        return LegalTriplet(
            subject=Subject(text="Seller", role=LegalRole.OBLIGOR),
            action=Action(predicate="deliver", object="the Goods"),
            condition=Condition(text="within 30 days", type=ConditionType.TEMPORAL),
        )

    def test_creation(self, sample_triplet):
        assert sample_triplet.subject.text == "Seller"
        assert sample_triplet.subject.role == LegalRole.OBLIGOR
        assert sample_triplet.action.predicate == "deliver"
        assert sample_triplet.condition.type == ConditionType.TEMPORAL

    def test_model_dump(self, sample_triplet):
        data = sample_triplet.model_dump()
        assert data["subject"]["text"] == "Seller"
        assert data["subject"]["role"] == "obligor"
        assert data["action"]["predicate"] == "deliver"
        assert data["condition"]["text"] == "within 30 days"
        assert data["condition"]["type"] == "temporal"

    def test_roundtrip_via_json(self, sample_triplet):
        json_str = sample_triplet.model_dump_json()
        recreated = LegalTriplet.model_validate_json(json_str)
        assert recreated.subject.text == sample_triplet.subject.text
        assert recreated.subject.role == sample_triplet.subject.role
        assert recreated.action.predicate == sample_triplet.action.predicate

    def test_condition_defaults_when_bare_condition(self):
        """裸 Condition() 具有合理的默认值（空文本、NONE 类型）。"""
        cond = Condition()
        assert cond.text == ""
        assert cond.type == ConditionType.NONE

    def test_condition_is_required_field(self):
        """LegalTriplet 要求显式传入 `condition`。"""
        triplet = LegalTriplet(
            subject=Subject(text="Buyer", role=LegalRole.RIGHT_HOLDER),
            action=Action(predicate="terminate", object="the Agreement"),
            condition=Condition(),
        )
        assert triplet.condition.text == ""
        assert triplet.condition.type == ConditionType.NONE
