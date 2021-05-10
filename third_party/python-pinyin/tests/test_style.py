from copy import deepcopy

from pypinyin import pinyin, Style
from pypinyin.style import register, convert


def test_custom_style_with_decorator():
    style_value = 'test_custom_style_with_decorator'

    @register(style_value)
    def func(pinyin, **kwargs):
        return pinyin + str(len(pinyin))

    hans = '北京'
    origin_pinyin_s = pinyin(hans)
    expected_pinyin_s = deepcopy(origin_pinyin_s)
    for pinyin_s in expected_pinyin_s:
        for index, py in enumerate(pinyin_s):
            pinyin_s[index] = func(py)

    assert pinyin(hans, style=style_value) == expected_pinyin_s


def test_custom_style_with_call():
    style_value = 'test_custom_style_with_call'

    def func(pinyin, **kwargs):
        return str(len(pinyin))

    register(style_value, func=func)

    hans = '北京'
    origin_pinyin_s = pinyin(hans)
    expected_pinyin_s = deepcopy(origin_pinyin_s)
    for pinyin_s in expected_pinyin_s:
        for index, py in enumerate(pinyin_s):
            pinyin_s[index] = func(py)

    assert pinyin(hans, style=style_value) == expected_pinyin_s


def test_finals_tone3_no_final():
    assert convert('ń', Style.FINALS_TONE3, True, None) == 'n2'


if __name__ == '__main__':
    import pytest
    pytest.cmdline.main()
