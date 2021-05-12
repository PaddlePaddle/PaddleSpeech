from typing import Iterator
from typing import Text

from pypinyin.constants import PHRASES_DICT
"""
MMSEG: A Word Identification System for Mandarin Chinese Text Based on Two Variants of the Maximum Matching Algorithm
最大正向匹配分词
http://technology.chtsai.org/mmseg/
https://www.jianshu.com/p/e4ae8d194487
"""


class PrefixSet():
    def __init__(self):
        self._set = set()

    def train(self, word_s: Iterator[Text]):
        """更新 prefix set
        :param word_s: 词语库列表
        :type word_s: iterable
        :return: None
        """
        for word in word_s:
            # 把词语的每个前缀更新到 prefix_set 中
            for index in range(len(word)):
                self._set.add(word[:index + 1])

    def __contains__(self, key: Text) -> bool:
        return key in self._set


class Seg():
    """正向最大匹配分词(Simple)
    简单的正向最大匹配，即按能匹配上的最长词做切分；
    :type prefix_set: PrefixSet
    :param strict_phrases: 是否严格按照词语分词，不允许把非词语的词当做词语进行分词。 False, 不严格； True, 严格。
    :type strict_phrases: bool
    """

    def __init__(self, prefix_set: PrefixSet,
                 strict_phrases: bool=False) -> None:
        self._prefix_set = prefix_set
        self._strict_phrases = strict_phrases

    def cut(self, text: Text) -> Iterator[Text]:
        """分词
        :param text: 待分词的文本
        :yield: 单个词语
        """
        remain = text
        while remain:
            matched = ''
            # 一次加一个字的匹配
            for index in range(len(remain)):
                word = remain[:index + 1]
                if word in self._prefix_set:
                    matched = word
                else:
                    # 前面的字符串是个词语
                    if (matched and ((not self._strict_phrases) or
                                     matched in PHRASES_DICT)):
                        yield matched
                        matched = ''
                        remain = remain[index:]
                    else:  # 前面为空或不是真正的词语
                        # 严格按照词语分词的情况下，不是词语的词拆分为单个汉字
                        # 先返回第一个字，后面的重新参与分词，
                        # 处理前缀匹配导致无法识别输入尾部的词语，
                        # 支持简单的逆向匹配分词:
                        #   已有词语：金融寡头 行业
                        #   输入：金融行业
                        #   输出：金 融 行业
                        if self._strict_phrases:
                            yield word[0]
                            remain = remain[index + 1 - len(word) + 1:]
                        else:
                            yield word
                            remain = remain[index + 1:]

                    # 有结果了，剩余的重新开始匹配
                    matched = ''
                    break
            else:  # 整个文本就是一个词语，或者不包含任何词语
                if self._strict_phrases and remain not in PHRASES_DICT:
                    for x in remain:
                        yield x
                else:
                    yield remain
                break

    def train(self, words: Iterator[Text]):
        """训练分词器
        :param words: 词语列表
        """
        self._prefix_set.train(words)


p_set = PrefixSet()
p_set.train(PHRASES_DICT.keys())

#: 基于内置词库的(简单)最大正向匹配分词器。使用:
#:
#: .. code-block:: python
#:
#:     >>> from pinyin.seg.mmseg import Seg
#:     >>> text = '你好，我是中国人，我爱我的祖国'
#:     >>> seg.cut(text)
#:     <generator object Seg.cut at 0x10b2df2b0>
#:     >>> list(seg.cut(text))
#:     ['你好', '，', '我', '是', '中国人', '，', '我', '爱',
#:      '我的', '祖', '国']
#:     >>> seg.train(['祖国', '我是'])
#:     >>> list(seg.cut(text))
#:     ['你好', '，', '我是', '中国人', '，', '我', '爱',
#:      '我的', '祖国']
#:     >>>
seg = Seg(p_set, strict_phrases=True)


def retrain(seg_instance: Seg):
    """重新使用内置词典训练 seg_instance。
    比如在增加自定义词语信息后需要调用这个模块重新训练分词器
    :type seg_instance: Seg
    """
    seg_instance.train(PHRASES_DICT.keys())
