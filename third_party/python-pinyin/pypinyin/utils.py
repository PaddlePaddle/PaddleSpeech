import re

from typing import List
from typing import Text

from pypinyin import phonetic_symbol
from pypinyin.constants import RE_TONE2
from pypinyin.seg.simpleseg import simple_seg  # noqa

# 用于向后兼容，TODO: 废弃


def is_chinese_char(cp) -> bool:
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    # https://www.cnblogs.com/jacen789/p/10825350.html

    if ((cp >= 0x4E00 and cp <= 0x9FFF) or (cp >= 0x3400 and cp <= 0x4DBF) or
        (cp >= 0x20000 and cp <= 0x2A6DF) or
        (cp >= 0x2A700 and cp <= 0x2B73F) or
        (cp >= 0x2B740 and cp <= 0x2B81F) or
        (cp >= 0x2B820 and cp <= 0x2CEAF) or (cp >= 0xF900 and cp <= 0xFAFF) or
        (cp >= 0x2F800 and cp <= 0x2FA1F)):
        return True  # yapf: disable

    return False


def _replace_tone2_style_dict_to_default(string: Text) -> Text:
    regex = re.compile(RE_TONE2.pattern.replace('$', ''))
    d = phonetic_symbol.phonetic_symbol_reverse
    string = string.replace('ü', 'v').replace('5', '').replace('0', '')

    def _replace(m):
        s = m.group(0)
        return d.get(s) or s

    return regex.sub(_replace, string)


def _remove_dup_items(lst: List[Text]) -> List[Text]:
    new_lst = []
    for item in lst:
        if item not in new_lst:
            new_lst.append(item)
    return new_lst
