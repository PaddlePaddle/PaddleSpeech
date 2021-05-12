from pypinyin.constants import Style
from pypinyin.converter import DefaultConverter


def test_pre_convert_style_return_value():
    class A(DefaultConverter):
        def pre_convert_style(self, han, orig_pinyin, style, strict, **kwargs):
            return 'test'

    han = '测试'
    assert DefaultConverter().convert(han, Style.TONE2, False, 'ignore',
                                      True) == [['ce4'], ['shi4']]
    assert A().convert(han, Style.TONE2, False, 'ignore', True) == [['test'],
                                                                    ['test']]


def test_post_convert_style_return_value():
    class A(DefaultConverter):
        def post_convert_style(self, han, orig_pinyin, converted_pinyin, style,
                               strict, **kwargs):
            return 'test'

    han = '测试'
    assert DefaultConverter().convert(han, Style.TONE2, False, 'ignore',
                                      True) == [['ce4'], ['shi4']]
    assert A().convert(han, Style.TONE2, False, 'ignore', True) == [['test'],
                                                                    ['test']]


def test_pre_handle_nopinyin_return_value():
    class A(DefaultConverter):
        def pre_handle_nopinyin(self, chars, style, heteronym, errors, strict,
                                **kwargs):
            return 'abc'

    han = 'test'
    assert DefaultConverter().convert(han, Style.TONE2, False, 'default',
                                      True) == [['test']]
    assert A().convert(han, Style.TONE2, False, 'default', True) == [['abc']]


def test_post_handle_nopinyin_return_value():
    class A(DefaultConverter):
        def post_handle_nopinyin(self, chars, style, heteronym, errors, strict,
                                 pinyin, **kwargs):
            return 'abc'

    han = 'test'
    assert DefaultConverter().convert(han, Style.TONE2, False, 'default',
                                      True) == [['test']]
    assert A().convert(han, Style.TONE2, False, 'default', True) == [['abc']]


def test_post_pinyin_return_value_single_pinyin():
    class A(DefaultConverter):
        def post_pinyin(self, han, heteronym, pinyin, **kwargs):
            return {
                '测': [['zhāo']],
                '试': [['yáng']],
                '测试': [['zhāo'], ['yáng']],
            }[han]

    han = '测试'
    assert DefaultConverter().convert(han, Style.TONE3, False, 'ignore',
                                      True) == [['ce4'], ['shi4']]
    assert A().convert(han, Style.TONE3, False, 'ignore', True) == [['zhao1'],
                                                                    ['yang2']]


def test_post_pinyin_return_value_phrase_pinyin():
    class A(DefaultConverter):
        def post_pinyin(self, han, heteronym, pinyin, **kwargs):
            return {
                '北': [['zhāo']],
                '京': [['yáng']],
                '北京': [['zhāo'], ['yáng']],
            }[han]

    han = '北京'
    assert DefaultConverter().convert(han, Style.TONE3, False, 'ignore',
                                      True) == [['bei3'], ['jing1']]
    assert A().convert(han, Style.TONE3, False, 'ignore', True) == [['zhao1'],
                                                                    ['yang2']]
