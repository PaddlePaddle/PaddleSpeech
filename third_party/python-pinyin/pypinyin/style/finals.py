"""韵母相关拼音风格:
Style.FINALS
Style.FINALS_TONE
Style.FINALS_TONE2
Style.FINALS_TONE3
"""
from typing import Text
from typing import Any

from pypinyin.constants import Style
from pypinyin.standard import convert_finals
from pypinyin.style import register
from pypinyin.style._constants import RE_TONE3
from pypinyin.style._utils import get_finals
from pypinyin.style._utils import has_finals
from pypinyin.style._utils import replace_symbol_to_no_symbol
from pypinyin.style._utils import replace_symbol_to_number


class FinalsConverter():
    def to_finals(self, pinyin: Text, **kwargs: Any) -> Text:
        if kwargs.get('strict'):
            pinyin = convert_finals(pinyin)
        has_fi = has_finals(pinyin)

        # 替换声调字符为无声调字符
        pinyin = replace_symbol_to_no_symbol(pinyin)
        if not has_fi:
            return pinyin
        # 获取韵母部分
        return get_finals(pinyin, strict=False)

    def to_finals_tone(self, pinyin: Text, **kwargs: Any) -> Text:
        if not has_finals(pinyin):
            return pinyin

        # 获取韵母部分
        return get_finals(pinyin, strict=kwargs.get('strict'))

    def to_finals_tone2(self, pinyin: Text, **kwargs: Any) -> Text:
        if kwargs.get('strict'):
            pinyin = convert_finals(pinyin)
        has_fi = has_finals(pinyin)

        # 用数字表示声调
        pinyin = replace_symbol_to_number(pinyin)
        if not has_fi:
            return pinyin
        # 获取韵母部分
        return get_finals(pinyin, strict=False)

    def to_finals_tone3(self, pinyin: Text, **kwargs: Any) -> Text:
        if kwargs.get('strict'):
            pinyin = convert_finals(pinyin)
        has_fi = has_finals(pinyin)

        # 用数字表示声调
        pinyin = replace_symbol_to_number(pinyin)
        # 将声调数字移动到最后
        pinyin = RE_TONE3.sub(r'\1\3\2', pinyin)

        if not has_fi:
            return pinyin
        # 获取韵母部分
        return get_finals(pinyin, strict=False)


converter = FinalsConverter()
register(Style.FINALS, func=converter.to_finals)
register(Style.FINALS_TONE, func=converter.to_finals_tone)
register(Style.FINALS_TONE2, func=converter.to_finals_tone2)
register(Style.FINALS_TONE3, func=converter.to_finals_tone3)
