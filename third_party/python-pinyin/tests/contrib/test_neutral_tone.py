from pytest import mark

from pypinyin import lazy_pinyin, Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.contrib.uv import V2UMixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


class HerConverter(NeutralToneWith5Mixin, V2UMixin, DefaultConverter):
    pass


class TheyConverter(V2UMixin, NeutralToneWith5Mixin, DefaultConverter):
    pass


my_pinyin = Pinyin(MyConverter())
her_pinyin = Pinyin(HerConverter())
they_pinyin = Pinyin(TheyConverter())


def test_neutral_tone_with_5():
    assert lazy_pinyin('好了', style=Style.TONE2) == ['ha3o', 'le']
    assert my_pinyin.lazy_pinyin('好了', style=Style.TONE2) == ['ha3o', 'le5']
    assert lazy_pinyin(
        '好了', style=Style.TONE2,
        neutral_tone_with_five=True) == ['ha3o', 'le5']
    assert her_pinyin.lazy_pinyin('好了', style=Style.TONE2) == ['ha3o', 'le5']
    assert lazy_pinyin(
        '好了', style=Style.TONE2, neutral_tone_with_five=True,
        v_to_u=True) == ['ha3o', 'le5']
    assert they_pinyin.lazy_pinyin('好了', style=Style.TONE2) == ['ha3o', 'le5']
    assert lazy_pinyin(
        '好了', style=Style.TONE2, v_to_u=True,
        neutral_tone_with_five=True) == ['ha3o', 'le5']

    assert lazy_pinyin('好了绿', style=Style.TONE2) == ['ha3o', 'le', 'lv4']
    assert lazy_pinyin(
        '好了绿', style=Style.TONE2, v_to_u=True,
        neutral_tone_with_five=True) == ['ha3o', 'le5', 'lü4']

    assert lazy_pinyin('好了') == ['hao', 'le']
    assert my_pinyin.lazy_pinyin('好了') == ['hao', 'le']
    assert lazy_pinyin('好了', neutral_tone_with_five=True) == ['hao', 'le']
    assert her_pinyin.lazy_pinyin('好了') == ['hao', 'le']
    assert lazy_pinyin(
        '好了', neutral_tone_with_five=True, v_to_u=True) == ['hao', 'le']
    assert lazy_pinyin(
        '好了绿', v_to_u=True, neutral_tone_with_five=True) == ['hao', 'le', 'lü']


@mark.parametrize('input,style,expected_old, expected_new', [
    ['你好', Style.TONE2, ['ni3', 'ha3o'], ['ni3', 'ha3o']],
    ['你好', Style.FINALS_TONE2, ['i3', 'a3o'], ['i3', 'a3o']],
    ['你好', Style.TONE3, ['ni3', 'hao3'], ['ni3', 'hao3']],
    ['你好', Style.FINALS_TONE3, ['i3', 'ao3'], ['i3', 'ao3']],
    ['男孩儿', Style.TONE2, ['na2n', 'ha2i', 'er'], ['na2n', 'ha2i', 'e5r']],
    ['男孩儿', Style.FINALS_TONE2, ['a2n', 'a2i', 'er'], ['a2n', 'a2i', 'e5r']],
    ['男孩儿', Style.TONE3, ['nan2', 'hai2', 'er'], ['nan2', 'hai2', 'er5']],
    ['男孩儿', Style.FINALS_TONE3, ['an2', 'ai2', 'er'], ['an2', 'ai2', 'er5']],
    ['我们', Style.TONE2, ['wo3', 'men'], ['wo3', 'me5n']],
    ['我们', Style.FINALS_TONE2, ['uo3', 'en'], ['uo3', 'e5n']],
    ['我们', Style.TONE3, ['wo3', 'men'], ['wo3', 'men5']],
    ['我们', Style.FINALS_TONE3, ['uo3', 'en'], ['uo3', 'en5']],
    ['衣裳', Style.TONE2, ['yi1', 'shang'], ['yi1', 'sha5ng']],
    ['衣裳', Style.FINALS_TONE2, ['i1', 'ang'], ['i1', 'a5ng']],
    ['衣裳', Style.TONE3, ['yi1', 'shang'], ['yi1', 'shang5']],
    ['衣裳', Style.FINALS_TONE3, ['i1', 'ang'], ['i1', 'ang5']],
    ['好吧', Style.TONE2, ['ha3o', 'ba'], ['ha3o', 'ba5']],
    ['好吧', Style.FINALS_TONE2, ['a3o', 'a'], ['a3o', 'a5']],
    ['好吧', Style.TONE3, ['hao3', 'ba'], ['hao3', 'ba5']],
    ['好吧', Style.FINALS_TONE3, ['ao3', 'a'], ['ao3', 'a5']],
])
def test_neutral_tone_with_5_many_cases(input, style, expected_old,
                                        expected_new):
    assert lazy_pinyin(input, style=style) == expected_old
    assert my_pinyin.lazy_pinyin(input, style=style) == expected_new
    assert lazy_pinyin(
        input, style=style, neutral_tone_with_five=True) == expected_new
    assert lazy_pinyin(
        input, style=style, neutral_tone_with_five=True,
        v_to_u=True) == expected_new
