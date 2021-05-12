from typing import Optional
from typing import Text


def right_mark_index(pinyin_no_tone: Text) -> Optional[int]:
    """
    标调位置
        有 ɑ 不放过，
    　　没 ɑ 找 o、e；
    　　ɑ、o、e、i、u、ü
    　　标调就按这顺序；
    　　i、u 若是连在一起，
    　　谁在后面就标谁。

    有ɑ不放过（有ɑ一定要标在ɑ上）；
    无ɑ找oe（没有ɑ的时候标在o上,如果没有o则标在e上）;
    iu并列标在后（iu, ui的情况,标在后面的字母上,比如说iu应该标u,ui应该标i）；
    单个韵母不用说（只能标在单韵母上）

    http://www.hwjyw.com/resource/content/2010/06/04/8183.shtml
    https://www.zhihu.com/question/23655297
    https://github.com/mozillazg/python-pinyin/issues/160
    http://www.pinyin.info/rules/where.html
    """

    # 有 ɑ 不放过, 没 ɑ 找 o、e
    for c in ['a', 'o', 'e']:
        if c in pinyin_no_tone:
            return pinyin_no_tone.index(c)

    # i、u 若是连在一起，谁在后面就标谁
    for c in ['iu', 'ui']:
        if c in pinyin_no_tone:
            return pinyin_no_tone.index(c) + 1

    # ɑ、o、e、i、u、ü
    for c in ['i', 'u', 'v', 'ü']:
        if c in pinyin_no_tone:
            return pinyin_no_tone.index(c)

    # n, m, ê
    for c in ['n', 'm', 'ê']:
        if c in pinyin_no_tone:
            return pinyin_no_tone.index(c)
