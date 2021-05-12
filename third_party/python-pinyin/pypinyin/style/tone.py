"""TONE 相关的几个拼音风格实现:
Style.TONE
Style.TONE2
Style.TONE3
"""
from typing import Any
from typing import Text

from pypinyin.constants import Style
from pypinyin.style import register
from pypinyin.style._constants import RE_TONE3
from pypinyin.style._utils import replace_symbol_to_number


class ToneConverter():
    def to_tone(self, pinyin: Text, **kwargs: Any) -> Text:
        return pinyin

    def to_tone2(self, pinyin: Text, **kwargs: Any) -> Text:
        # 用数字表示声调
        pinyin = replace_symbol_to_number(pinyin)
        return pinyin

    def to_tone3(self, pinyin: Text, **kwargs: Any) -> Text:
        pinyin = self.to_tone2(pinyin, **kwargs)
        # 将声调数字移动到最后
        return RE_TONE3.sub(r'\1\3\2', pinyin)


converter = ToneConverter()
register(Style.TONE, func=converter.to_tone)
register(Style.TONE2, func=converter.to_tone2)
register(Style.TONE3, func=converter.to_tone3)
