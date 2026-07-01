"""
修饰语提取：aux、aux:pass、否定、acl:relcl、名词短语跨度、系词检查。
"""

from __future__ import annotations

from typing import Optional, List, Tuple

from src.extraction.schema import DependencyTree, Token
from src.utils.logging import get_logger

logger = get_logger(__name__)


def find_aux_verb(tree: DependencyTree,
                  predicate_idx: int) -> Optional[Token]:
    """查找修饰谓词的助动词。

    UD: aux(predicate, auxiliary) —— 助动词。
    法律文本中，助动词承载道义情态：
      "Seller SHALL deliver" -> aux(deliver, shall)
      "Buyer MAY terminate"  -> aux(terminate, may)
      "Party MUST pay"       -> aux(pay, must)

    助动词对法律角色分类至关重要：
      shall/must  -> 义务 -> OBLIGOR
      may         -> 许可 -> RIGHT_HOLDER
      shall not   -> 禁止 -> PROHIBITED_PARTY

    参数：
        tree: 依存树。
        predicate_idx: 谓词词元的 1 基索引。

    返回：
        aux Token，无助动词时返回 None
        （如从属从句中的裸不定式）。
    """
    children = tree.get_children(predicate_idx, deprel="aux")
    if children:
        # 返回第一个 aux。复杂动词短语中可能有多个 aux
        #（"shall have been delivered"）—— 取第一个（最左），
        # 承载主要情态。
        return children[0]
    return None


def find_aux_pass(tree: DependencyTree,
                  predicate_idx: int) -> Optional[Token]:
    """查找谓词上的被动助动词（be/get）。

    UD: aux:pass(predicate, be_aux) —— 被动助动词。
    法律文本中：
      "the Goods ARE delivered"   -> aux:pass(delivered, are)
      "the Agreement was breached" -> aux:pass(breached, was)

    aux:pass 的存在确认构造为形态被动，
    而非形容词过去分词。

    参数：
        tree: 依存树。
        predicate_idx: 谓词词元的 1 基索引。

    返回：
        aux:pass Token，或 None。
    """
    children = tree.get_children(predicate_idx, deprel="aux:pass")
    if children:
        return children[0]
    return None


def find_negation(tree: DependencyTree,
                  predicate_idx: int) -> Optional[Token]:
    """查找谓词上的否定标记。

    UD: neg(predicate, negation) —— 否定修饰语。
    法律文本中：
      "shall NOT assign"  -> neg(assign, not)
      "may NOT disclose"  -> neg(disclose, not)
      "NO party shall"    -> neg(shall, no) [少见 —— 通常 neg 附于动词]

    否定对区分义务与禁止至关重要：
      "shall deliver"    -> 义务（须交付）
      "shall NOT deliver" -> 禁止（不得交付）

    参数：
        tree: 依存树。
        predicate_idx: 谓词词元的 1 基索引。

    返回：
        neg Token，谓词未否定时返回 None。
    """
    children = tree.get_children(predicate_idx, deprel="neg")
    if children:
        return children[0]
    return None


def has_acl_relcl(tree: DependencyTree, token_idx: int) -> bool:
    """检查词元是否有关系从句修饰语。

    UD: acl:relcl(head, clause) —— 修饰名词的关系从句。
    法律文本中：
      "the Party WHO receives notice" -> acl:relcl(Party, receives)
      "any amount THAT exceeds the cap" -> acl:relcl(amount, exceeds)

    关系从句在名词短语内嵌入谓词-论元结构，
    当大语言模型试图从关系从句而非主句抽取谓词时，
    可能干扰抽取。

    参数：
        tree: 依存树。
        token_idx: 待检查关系从句修饰语的词元索引。

    返回：
        词元至少有一个 acl:relcl 依存时返回 True。
    """
    children = tree.get_children(token_idx, deprel="acl:relcl")
    return len(children) > 0


def find_acl_relcl_head(tree: DependencyTree,
                        token_idx: int) -> Optional[Token]:
    """获取修饰词元的关系从句头。

    参数：
        tree: 依存树。
        token_idx: 中心名词的词元索引。

    返回：
        acl:relcl 头 Token，或 None。
    """
    children = tree.get_children(token_idx, deprel="acl:relcl")
    if children:
        return children[0]
    return None


def get_noun_phrase_span(tree: DependencyTree,
                         head_idx: int) -> Tuple[str, List[int]]:
    """给定句法中心词，提取名词短语的完整文本。

    遍历名词子树以收集限定词、形容词、
    介词修饰语及属于名词短语的关系从句。
    产出大语言模型抽取器应作为主语或宾语文本
    的「最大名词短语」跨度。

    示例：对 "all outstanding amounts due under this Agreement"：
      head = "amounts"（obj 或 nsubj）
      span = "all outstanding amounts due under this Agreement"

    参数：
        tree: 依存树。
        head_idx: 名词短语中心词的索引。

    返回：
        (full_np_text, list_of_token_indices) 元组。
    """
    # 收集中心名词的完整子树。
    subtree_indices = set(tree._collect_subtree(head_idx))

    # 也包含可能作为 `det`、`amod`、`nummod` 依存附着的前置修饰语。
    # 它们已是头的子树的一部分（通过 _collect_subtree），
    # 因它们是中心名词的依存。
    #
    # 对复杂名词短语，还需包含通过 `nmod`、`obl`、`acl` 等
    # 依附于头的词元。_collect_subtree 已传递处理。

    sorted_indices = sorted(subtree_indices)
    tokens_sorted = [
        tree.get_token(i) for i in sorted_indices
    ]
    tokens_sorted = [t for t in tokens_sorted if t is not None]
    text = " ".join(t.text for t in tokens_sorted)

    return text, sorted_indices


def is_copular_construction(tree: DependencyTree,
                            predicate_idx: int) -> bool:
    """检查谓词是否为系词构造的一部分。

    UD: cop(predicate, be) —— 系词关系。
    法律文本："The Agreement IS binding" -> cop(binding, is)

    系词构造以 "be" 为助动词，形容词或名词作谓语。
    此时真实「动作」可能在补语从句或名词谓语本身。

    参数：
        tree: 依存树。
        predicate_idx: 潜在谓词的索引。

    返回：
        词元有 cop 依存时返回 True。
    """
    cop_children = tree.get_children(predicate_idx, deprel="cop")
    return len(cop_children) > 0
