"""
StanzaParser 包 —— 依存句法解析器封装
======================================
模块级单例状态 + StanzaParser 类门面，委托至各子模块。

子模块：
  _pipeline:   流水线初始化（__init__、nlp 属性、is_available）
  _parsing:    解析方法（parse、parse_batch）
  _conversion: Stanza Word 到 LexSpec Token 的内部转换
"""

from __future__ import annotations

from src.linguistic.parser._pipeline import (
    _nlp,
    _nlp_initialized,
    _init_pipeline,
    _get_nlp,
    _is_available,
)
from src.linguistic.parser._parsing import _parse, _parse_batch
from src.linguistic.parser._conversion import _convert_sentence_to_tokens


class StanzaParser:
    """Stanza 依存句法解析器 —— 用于英文法律文本的 UD 解析。

    使用模块级单例模式，重复实例化不会重新加载模型。

    Stanza 流水线组件:
      - tokenize:  句子分割和词语分词
      - mwt:       多词 token 展开（如 "don't" → "do" + "not"）
      - pos:       词性标注（UPOS + XPOS）
      - lemma:     词形还原（规范字典形式）
      - depparse:  UD 依存解析（head + deprel 标注）

    使用示例::

        parser = StanzaParser()
        tree = parser.parse("Seller shall deliver the goods within 30 days.")
        trees = parser.parse_batch([sentence1, sentence2, sentence3])
    """

    def __init__(
        self,
        model_path=None,
        lang="en",
        processors="tokenize,mwt,pos,lemma,depparse",
        download_method="REUSE_RESOURCES",
        use_gpu=True,
    ):
        """初始化 Stanza 流水线（模块级单例，重复调用不会重复加载模型）。

        参数:
            model_path: Stanza 模型目录路径；为 None 时使用默认下载位置。
            lang: 语言代码，法律合同语料为 ``"en"``。
            processors: 逗号分隔的处理器链，默认含分词、多词展开、
                词性标注、词形还原与依存解析。
            download_method: 模型下载策略；``"REUSE_RESOURCES"`` 表示优先
                复用已缓存模型，避免每次运行重复下载。
            use_gpu: 是否尝试使用 GPU 加速解析。
        """
        _init_pipeline(model_path, lang, processors, download_method, use_gpu)

    @property
    def nlp(self):
        """返回底层 Stanza ``Pipeline`` 实例。

        返回:
            已初始化的 Stanza 流水线对象，可直接调用 ``nlp(text)`` 获取
            原始 ``Document``。多数场景应优先使用 ``parse()`` /
            ``parse_batch()``，它们会转换为 ``DependencyTree``。
        """
        return _get_nlp()

    def parse(self, text):
        """解析单句并返回 UD 依存树。

        参数:
            text: 单条英语句子或合同条款文本；多句输入仅解析第一句。

        返回:
            ``DependencyTree``，每个词元含 UD 依存关系、词性、词元与索引。

        异常:
            ValueError: 输入为空或 Stanza 未产出任何句子。
            RuntimeError: 流水线尚未成功初始化。
        """
        return _parse(self, text)

    def parse_batch(self, texts):
        """批量解析多条句子，比循环调用 ``parse()`` 更高效。

        参数:
            texts: 待解析的句子字符串列表；空字符串会被跳过。

        返回:
            ``DependencyTree`` 列表，与 ``texts`` 中非空项一一对应、顺序一致。
        """
        return _parse_batch(self, texts)

    def is_available(self):
        """检查 Stanza 流水线是否已成功加载。

        返回:
            流水线可用且可解析时为 ``True``，初始化失败或未加载时为 ``False``。
        """
        return _is_available()
