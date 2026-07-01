"""
StanzaParser 包 —— 依存句法解析器封装
======================================
Singleton state + StanzaParser class facade that delegates to submodules.

Submodules:
  _pipeline:   Pipeline initialization (__init__, nlp property, is_available)
  _parsing:    Parsing methods (parse, parse_batch)
  _conversion: Internal conversion from Stanza Word to LexSpec Token
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
        _init_pipeline(model_path, lang, processors, download_method, use_gpu)

    @property
    def nlp(self):
        return _get_nlp()

    def parse(self, text):
        return _parse(self, text)

    def parse_batch(self, texts):
        return _parse_batch(self, texts)

    def is_available(self):
        return _is_available()
