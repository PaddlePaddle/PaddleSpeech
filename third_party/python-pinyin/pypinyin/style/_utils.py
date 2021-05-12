from pypinyin.standard import convert_finals
from pypinyin.style._constants import _INITIALS
from pypinyin.style._constants import _INITIALS_NOT_STRICT
from pypinyin.style._constants import PHONETIC_SYMBOL_DICT
from pypinyin.style._constants import PHONETIC_SYMBOL_DICT_KEY_LENGTH_NOT_ONE
from pypinyin.style._constants import RE_NUMBER
from pypinyin.style._constants import RE_PHONETIC_SYMBOL

from typing import Text


def get_initials(pinyin: Text, strict: bool) -> Text:
    """获取单个拼音中的声母.
    :param pinyin: 单个拼音
    :type pinyin: unicode
    :param strict: 是否严格遵照《汉语拼音方案》来处理声母和韵母
    :return: 声母
    :rtype: unicode
    """
    if strict:
        _initials = _INITIALS
    else:
        _initials = _INITIALS_NOT_STRICT

    for i in _initials:
        if pinyin.startswith(i):
            return i
    return ''


def get_finals(pinyin: Text, strict: bool) -> Text:
    """获取单个拼音中的韵母.
    :param pinyin: 单个拼音
    :type pinyin: unicode
    :param strict: 是否严格遵照《汉语拼音方案》来处理声母和韵母
    :return: 韵母
    :rtype: unicode
    """
    if strict:
        pinyin = convert_finals(pinyin)

    initials = get_initials(pinyin, strict=strict) or ''
    # 没有声母，整个都是韵母
    if not initials:
        return pinyin
    # 按声母分割，剩下的就是韵母
    return ''.join(pinyin.split(initials, 1))


def has_finals(pinyin: Text) -> bool:
    """判断是否有韵母"""
    # 鼻音: 'm̄', 'ḿ', 'm̀', 'ń', 'ň', 'ǹ ' 没有韵母
    for symbol in ['m̄', 'ḿ', 'm̀', 'ń', 'ň', 'ǹ', 'ê̄', 'ế', 'ê̌', 'ề']:
        if symbol in pinyin:
            return False

    return True


def replace_symbol_to_number(pinyin: Text) -> Text:
    """把声调替换为数字"""

    def _replace(match):
        symbol = match.group(0)  # 带声调的字符
        # 返回使用数字标识声调的字符
        return PHONETIC_SYMBOL_DICT[symbol]

    # 替换拼音中的带声调字符
    value = RE_PHONETIC_SYMBOL.sub(_replace, pinyin)
    for symbol, to in PHONETIC_SYMBOL_DICT_KEY_LENGTH_NOT_ONE.items():
        value = value.replace(symbol, to)

    return value


def replace_symbol_to_no_symbol(pinyin: Text) -> Text:
    """把带声调字符替换为没有声调的字符"""
    value = replace_symbol_to_number(pinyin)
    return RE_NUMBER.sub('', value)
