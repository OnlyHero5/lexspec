"""
依赖树模型 —— Universal Dependencies 依存句法树
===============================================

封装 UD 解析器的输出，提供类型化的树遍历、子树提取
和查询接口。
"""

from __future__ import annotations

from typing import Optional, List

from pydantic import BaseModel, Field

from .dependency import Token, ClauseSpan


class DependencyTree(BaseModel):
    """单句 UD 依存树 —— 语言学模块的核心数据结构。

    字段:
        text: 原始句子文本。
        tokens: 含完整 UD 标注的有序 ``Token`` 列表。

    常用属性/方法:
        root_index: 句法根词元索引（``head == 0``）。
        token_count: 词元总数。
        get_token(index): 按 1 基索引取词元。
        get_children(head, deprel): 获取指定依存关系的子节点。
        get_subtree_span(index): 提取以某词元为根的子树文本跨度。
    """
    text: str = Field(
        description="原始句子文本，保留供参考"
    )
    tokens: List[Token] = Field(
        description="含完整 UD 依存标注的有序词元列表"
    )

    # ------------------------------------------------------------------
    # 属性 —— 从词元列表推导的信息
    # ------------------------------------------------------------------

    @property
    def root_index(self) -> Optional[int]:
        """
        句法根词元的索引（head == 0 的词元）。

        格式良好的 UD 树中通常为 MAIN 子句谓词（限定动词）。
        未找到根时返回 None（如词元列表为空）。
        """
        for t in self.tokens:
            if t.head == 0:
                return t.index
        return None

    @property
    def token_count(self) -> int:
        """句子中的词元总数。"""
        return len(self.tokens)

    # ------------------------------------------------------------------
    # 访问方法 —— 词元检索与导航
    # ------------------------------------------------------------------

    def get_token(self, index: int) -> Optional[Token]:
        """
        按 1 基索引检索词元。

        参数:
            index: 1 基词元索引。

        返回:
            找到则返回 Token 对象，否则返回 None。
        """
        for t in self.tokens:
            if t.index == index:
                return t
        return None

    def get_children(self, head_index: int, deprel: Optional[str] = None) -> List[Token]:
        """
        获取依存树中某词元的全部直接依存子节点。

        参数:
            head_index: 中心词的 1 基索引。
            deprel:     可选过滤器——仅返回具有该依存关系标签的子节点
                        （如 'nsubj'、'obj'、'advcl'）。

        返回:
            子 Token 对象列表。若无依存子节点或无匹配子节点则返回空列表。
        """
        result = [t for t in self.tokens if t.head == head_index]
        if deprel is not None:
            result = [t for t in result if t.deprel == deprel]
        return result

    def get_head(self, token_index: int) -> Optional[Token]:
        """
        获取某词元的支配词（中心词）。

        参数:
            token_index: 依存子节点的 1 基索引。

        返回:
            中心词 Token 对象；若 token_index 为根或未找到则返回 None。
        """
        token = self.get_token(token_index)
        if token is None or token.head == 0:
            return None
        return self.get_token(token.head)

    def get_path_to_root(self, token_index: int) -> List[Token]:
        """
        计算从某词元到根节点的依存路径。

        用于度量依存距离并识别对大语言模型抽取具有挑战性的长距离依存。

        参数:
            token_index: 起始词元的 1 基索引。

        返回:
            从给定词元（含）到根（含）的有序词元列表。
            词元未找到则返回空列表。
        """
        path = []
        current = self.get_token(token_index)
        if current is None:
            return path
        path.append(current)
        # 沿 head 指针向上遍历直至根（head == 0）
        while current.head != 0:
            current = self.get_token(current.head)
            if current is None:
                break  # 防御性：树结构损坏
            path.append(current)
        return path

    # ------------------------------------------------------------------
    # 子树方法 —— 提取子句跨度
    # ------------------------------------------------------------------

    def get_subtree_span(self, head_index: int) -> ClauseSpan:
        """
        提取以 head_index 为根的子树的完整文本跨度。

        执行深度优先遍历，收集给定中心词支配的全部词元（含中心词本身）。
        拼接前按表层顺序（1 基索引）排序。

        提取条件子句文本的主要方法：
            advcl_heads = [c for c in tree.get_children(root_idx) if c.deprel == 'advcl']
            for advcl in advcl_heads:
                span = tree.get_subtree_span(advcl.index)

        参数:
            head_index: 子树根词元的 1 基索引。

        返回:
            含排序后词元索引与拼接文本的 ClauseSpan。
        """
        subtree_indices = self._collect_subtree(head_index)
        subtree_indices.sort()
        tokens_sorted = [
            t for i in subtree_indices
            if (t := self.get_token(i)) is not None
        ]
        text = " ".join(t.text for t in tokens_sorted)
        # 确定子树根的依存关系
        head_token = self.get_token(head_index)
        head_deprel = head_token.deprel if head_token else ""
        return ClauseSpan(tokens=subtree_indices, text=text, deprel=head_deprel)

    def get_subtree_tokens(self, head_index: int) -> List[Token]:
        """
        返回以 head_index 为根的子树中全部 Token 对象，按表层顺序排序。

        参数:
            head_index: 子树根的 1 基索引。

        返回:
            子树中排序后的 Token 对象列表。
        """
        indices = sorted(self._collect_subtree(head_index))
        return [t for i in indices if (t := self.get_token(i)) is not None]

    def _collect_subtree(self, head_index: int) -> List[int]:
        """
        深度优先遍历，收集 head_index 支配的子树中全部词元索引（含中心词本身）。

        这是对 ``head`` 关系的传递闭包——收集依存路径经过 head_index 的每个词元。

        参数:
            head_index: 子树根的 1 基索引。

        返回:
            1 基词元索引列表（可能未排序）。
        """
        result = [head_index]
        for child in self.get_children(head_index):
            result.extend(self._collect_subtree(child.index))
        return result

    # ------------------------------------------------------------------
    # 查询方法 —— 树上的模式匹配
    # ------------------------------------------------------------------

    def find_tokens_by_upos(self, upos: str) -> List[Token]:
        """
        查找具有给定通用词性标签的全部词元。

        参数:
            upos: 通用词性标签，如 'VERB'、'NOUN'、'AUX'。

        返回:
            匹配的 Token 对象列表。
        """
        return [t for t in self.tokens if t.upos == upos]

    def find_tokens_by_deprel(self, deprel: str) -> List[Token]:
        """
        查找具有给定依存关系标签的全部词元。

        参数:
            deprel: 依存关系，如 'nsubj'、'obj'、'nsubj:pass'。

        返回:
            匹配的 Token 对象列表。
        """
        return [t for t in self.tokens if t.deprel == deprel]

    def has_deprel(self, deprel: str) -> bool:
        """
        检查树中是否存在具有给定依存关系的词元。

        参数:
            deprel: 待查找的依存关系标签。

        返回:
            至少有一个词元具有该 deprel 则为 True，否则为 False。
        """
        return any(t.deprel == deprel for t in self.tokens)

    def get_dependency_distance(self, child_index: int, head_index: int) -> int:
        """
        计算依存子节点与其中心词之间的线性距离（词元数）。

        较大距离（> 3）表示长距离依存，对大语言模型抽取与规则分析均具挑战性。

        参数:
            child_index: 依存子节点的 1 基索引。
            head_index:  中心词的 1 基索引。

        返回:
            词元索引的绝对差值。任一词元未找到则返回 -1。
        """
        child = self.get_token(child_index)
        head = self.get_token(head_index)
        if child is None or head is None:
            return -1
        return abs(child.index - head.index)
