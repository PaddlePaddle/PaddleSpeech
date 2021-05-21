#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/17
"""
![phkit](phkit.png "phkit")

## phkit
phoneme toolkit: 拼音相关的文本处理工具箱，中文和英文的语音合成前端文本解决方案。

#### 安装

```
pip install -U phkit
```
"""
__version__ = "0.2.8"

version_doc = """
#### 版本
v{}
""".format(__version__)

history_doc = """
### 历史版本
#### v0.2.8
- 文本转拼音轻声用5表示音调。
- 文本转拼音确保文本和拼音一一对应，文本长度和拼音列表长度相同。
- 增加拼音格式转换，国标格式和字母数字格式相互转换。

#### v0.2.7
- 所有中文音素都能被映射到。

#### v0.2.5
- 修正拼音转音素的潜在bug。

#### v0.2.4
- 修正几个默认拼音。

#### v0.2.3
- 汉字转拼音轻量化。
- 词语拼音词典去除全都是默认拼音的词语。

#### v0.2.2
- 修正安装依赖报错问题。

#### v0.2.1
- 增加中文的text_to_sequence方法，可替换英文版本应对中文环境。
- 兼容v0.1.0（含）之前版本需要在python3.7（含）版本以上，否则请改为从phkit.chinese导入模块。

#### v0.2.0
- 增加文本转拼音的模块，依赖python-pinyin，jieba，phrase-pinyin-data模块。
- 中文的音素方案移动到chinese模块。

#### v0.1.0
- 增加英文版本的音素方案，包括英文字母和英文音素。
- 增加简单的数字转中文的方法。

#### todo

```
文本正则化处理
数字读法
字符读法
常见规则读法

文本转拼音
pypinyin
国标和alnum转换

anything转音素
字符
英文
汉字
OOV

进阶:
分词
命名实体识别
依存句法分析
```
"""

from phkit.chinese import __doc__ as doc_chinese
from phkit.chinese.symbol import __doc__ as doc_symbol
from phkit.chinese.sequence import __doc__ as doc_sequence
from phkit.chinese.pinyin import __doc__ as doc_pinyin
from phkit.chinese.phoneme import __doc__ as doc_phoneme
from phkit.chinese.number import __doc__ as doc_number
from phkit.chinese.convert import __doc__ as doc_convert
from phkit.chinese.style import __doc__ as doc_style
from .english import __doc__ as doc_english
from .pinyinkit import __doc__ as doc_pinyinkit

readme_docs = [__doc__, version_doc,
               doc_pinyinkit,
               doc_chinese, doc_symbol, doc_sequence, doc_pinyin, doc_phoneme, doc_number, doc_convert, doc_style,
               doc_english,
               history_doc]

from .chinese import text_to_sequence as chinese_text_to_sequence, sequence_to_text as chinese_sequence_to_text
from .english import text_to_sequence as english_text_to_sequence, sequence_to_text as english_sequence_to_text
from .pinyinkit import lazy_pinyin, pinyin, slug, initialize

# 兼容0.1.0之前的版本，python3.7以上版本支持。
from .chinese import convert, number, phoneme, sequence, symbol, style
from .chinese.style import guobiao2shengyundiao, shengyundiao2guobiao
from .chinese.convert import fan2jian, jian2fan, quan2ban, ban2quan
from .chinese.number import say_digit, say_decimal, say_number
from .chinese.pinyin import text2pinyin, split_pinyin
from .chinese.sequence import text2sequence, text2phoneme, pinyin2phoneme, phoneme2sequence, sequence2phoneme
from .chinese.sequence import symbol_chinese, ph2id_dict, id2ph_dict

if __name__ == "__main__":
    print(__file__)
