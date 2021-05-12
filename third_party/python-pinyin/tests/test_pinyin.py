#!/usr/bin/env python3

import pytest

from pypinyin import (pinyin, slug, lazy_pinyin, load_single_dict,
                      load_phrases_dict, NORMAL, TONE, TONE2, TONE3, INITIALS,
                      FIRST_LETTER, FINALS, FINALS_TONE, FINALS_TONE2,
                      FINALS_TONE3, BOPOMOFO, BOPOMOFO_FIRST, CYRILLIC,
                      CYRILLIC_FIRST, Style)
from pypinyin.constants import SUPPORT_UCS4
from pypinyin.seg import simpleseg


def test_pinyin_initials():
    """包含声明和韵母的词语"""
    hans = '中心'
    # 默认风格，带声调
    assert pinyin(hans) == [['zh\u014dng'], ['x\u012bn']]
    assert pinyin(hans, strict=False) == [['zh\u014dng'], ['x\u012bn']]
    # 普通风格，不带声调
    assert pinyin(hans, NORMAL) == [['zhong'], ['xin']]
    assert pinyin(hans, NORMAL, strict=False) == [['zhong'], ['xin']]
    # 声调风格，拼音声调在韵母第一个字母上
    assert pinyin(hans, TONE) == [['zh\u014dng'], ['x\u012bn']]
    assert pinyin(hans, TONE, strict=False) == [['zh\u014dng'], ['x\u012bn']]
    # 声调风格2，即拼音声调在各个声母之后，用数字 [1-4] 进行表示
    assert pinyin(hans, TONE2) == [['zho1ng'], ['xi1n']]
    assert pinyin(hans, TONE2, strict=False) == [['zho1ng'], ['xi1n']]
    # 声调风格3，即拼音声调在各个拼音之后，用数字 [1-4] 进行表示
    assert pinyin(hans, TONE3) == [['zhong1'], ['xin1']]
    assert pinyin(hans, TONE3, strict=False) == [['zhong1'], ['xin1']]
    # 声母风格，只返回各个拼音的声母部分
    assert pinyin(hans, INITIALS) == [['zh'], ['x']]
    assert pinyin(hans, INITIALS, strict=False) == [['zh'], ['x']]
    # 首字母风格，只返回拼音的首字母部分
    assert pinyin(hans, FIRST_LETTER) == [['z'], ['x']]
    assert pinyin(hans, FIRST_LETTER, strict=False) == [['z'], ['x']]
    # 注音风格，带声调
    assert pinyin(hans, BOPOMOFO) == [['ㄓㄨㄥ'], ['ㄒㄧㄣ']]
    assert pinyin(hans, BOPOMOFO, strict=False) == [['ㄓㄨㄥ'], ['ㄒㄧㄣ']]
    # 注音风格，首字母
    assert pinyin(hans, BOPOMOFO_FIRST) == [['ㄓ'], ['ㄒ']]
    assert pinyin(hans, BOPOMOFO_FIRST, strict=False) == [['ㄓ'], ['ㄒ']]
    # test CYRILLIC style
    assert pinyin(hans, CYRILLIC) == [['чжун1'], ['синь1']]
    assert pinyin(hans, CYRILLIC, strict=False) == [['чжун1'], ['синь1']]
    # CYRILLIC_FIRST style return only first letters
    assert pinyin(hans, CYRILLIC_FIRST) == [['ч'], ['с']]
    assert pinyin(hans, CYRILLIC_FIRST, strict=False) == [['ч'], ['с']]
    # 启用多音字模式
    assert pinyin(hans, heteronym=True) == [['zh\u014dng', 'zh\xf2ng'],
                                            ['x\u012bn']]
    assert pinyin(hans, heteronym=True, strict=False) == \
        [['zh\u014dng', 'zh\xf2ng'], ['x\u012bn']]
    # 韵母风格1，只返回各个拼音的韵母部分，不带声调
    assert pinyin(hans, style=FINALS) == [['ong'], ['in']]
    assert pinyin(hans, style=FINALS, strict=False) == [['ong'], ['in']]
    # 韵母风格2，带声调，声调在韵母第一个字母上
    assert pinyin(hans, style=FINALS_TONE) == [['\u014dng'], ['\u012bn']]
    assert pinyin(hans, style=FINALS_TONE, strict=False) == \
        [['\u014dng'], ['\u012bn']]
    # 韵母风格2，带声调，声调在各个声母之后，用数字 [1-4] 进行表示
    assert pinyin(hans, style=FINALS_TONE2) == [['o1ng'], ['i1n']]
    assert pinyin(hans, style=FINALS_TONE2, strict=False) == \
        [['o1ng'], ['i1n']]
    # 韵母风格3，带声调，声调在各个拼音之后，用数字 [1-4] 进行表示
    assert pinyin(hans, style=FINALS_TONE3) == [['ong1'], ['in1']]
    assert pinyin(hans, style=FINALS_TONE3, strict=False) == \
        [['ong1'], ['in1']]


def test_pinyin_finals():
    """只包含韵母的词语"""
    hans = '嗷嗷'
    assert pinyin(hans) == [['\xe1o'], ['\xe1o']]
    assert pinyin(hans + 'abc') == [['\xe1o'], ['\xe1o'], ['abc']]
    assert pinyin(hans, NORMAL) == [['ao'], ['ao']]
    assert pinyin(hans, TONE) == [['\xe1o'], ['\xe1o']]
    assert pinyin(hans, TONE2) == [['a2o'], ['a2o']]
    assert pinyin(hans, TONE3) == [['ao2'], ['ao2']]
    assert pinyin(hans, INITIALS) == [[''], ['']]
    assert pinyin(hans, FIRST_LETTER) == [['a'], ['a']]
    assert pinyin(hans, BOPOMOFO) == [['ㄠˊ'], ['ㄠˊ']]
    assert pinyin(hans, BOPOMOFO_FIRST) == [['ㄠ'], ['ㄠ']]
    assert pinyin(hans, CYRILLIC) == [['ао2'], ['ао2']]
    assert pinyin(hans, CYRILLIC_FIRST) == [['а'], ['а']]
    assert pinyin(hans, heteronym=True) == [['\xe1o'], ['\xe1o']]
    assert pinyin('啊', heteronym=True) == \
        [['a', 'ā', 'á', 'ǎ', 'à', 'è']]
    assert pinyin(hans, style=FINALS) == [['ao'], ['ao']]
    assert pinyin(hans, style=FINALS_TONE) == [['\xe1o'], ['\xe1o']]
    assert pinyin(hans, style=FINALS_TONE2) == [['a2o'], ['a2o']]
    assert pinyin(hans, style=FINALS_TONE3) == [['ao2'], ['ao2']]


def test_slug():
    hans = '中心'
    assert slug(hans) == 'zhong-xin'
    assert slug(hans, heteronym=True) == 'zhong-xin'


def test_zh_and_en():
    """中英文混合的情况"""
    # 中英文
    hans = '中心'
    assert pinyin(hans + 'abc') == [['zh\u014dng'], ['x\u012bn'], ['abc']]
    # 中英文混合的固定词组
    assert pinyin('黄山B股', style=TONE2) == \
        [['hua2ng'], ['sha1n'], ['B'], ['gu3']]
    assert pinyin('A股', style=TONE2) == [['A'], ['gu3']]
    assert pinyin('阿Q', style=TONE2) == [['a1'], ['Q']]
    assert pinyin('B超', style=TONE2) == [['B'], ['cha1o']]
    assert pinyin('AB超C', style=TONE2) == [['AB'], ['cha1o'], ['C']]
    assert pinyin('AB阿C', style=TONE2) == [['AB'], ['a1'], ['C']]
    assert pinyin('维生素C', style=TONE2) == \
        [['we2i'], ['she1ng'], ['su4'], ['C']]


def test_others():
    # 空字符串
    assert pinyin('') == []
    # 单个汉字
    assert pinyin('營') == [['y\xedng']]
    # 中国 人
    assert pinyin('中国人') == [['zh\u014dng'], ['gu\xf3'], ['r\xe9n']]
    # 日文
    assert pinyin('の') == [['\u306e']]
    # 没有读音的汉字，还不存在的汉字
    assert pinyin('\u9fff') == [['\u9fff']]


def test_lazy_pinyin():
    assert lazy_pinyin('中国人') == ['zhong', 'guo', 'ren']
    assert lazy_pinyin('中心') == ['zhong', 'xin']
    assert lazy_pinyin('中心', style=TONE) == ['zh\u014dng', 'x\u012bn']
    assert lazy_pinyin('中心', style=INITIALS) == ['zh', 'x']
    assert lazy_pinyin('中心', style=BOPOMOFO) == ['ㄓㄨㄥ', 'ㄒㄧㄣ']
    assert lazy_pinyin('中心', style=CYRILLIC) == ['чжун1', 'синь1']


def test_seg():
    hans = '音乐'
    hans_seg = list(simpleseg(hans))
    assert pinyin(hans_seg, style=TONE2) == [['yi1n'], ['yue4']]
    # 中英文混合的固定词组
    assert pinyin('黄山B股', style=TONE2) == \
        [['hua2ng'], ['sha1n'], ['B'], ['gu3']]
    assert pinyin('A股', style=TONE2) == [['A'], ['gu3']]
    assert pinyin('阿Q', style=TONE2) == [['a1'], ['Q']]
    assert pinyin('B超', style=TONE2) == [['B'], ['cha1o']]
    assert pinyin('AB超C', style=TONE2) == [['AB'], ['cha1o'], ['C']]
    assert pinyin('AB阿C', style=TONE2) == [['AB'], ['a1'], ['C']]
    assert pinyin('维生素C', style=TONE2) == \
        [['we2i'], ['she1ng'], ['su4'], ['C']]


def test_custom_pinyin_dict():
    hans = '桔'
    try:
        assert lazy_pinyin(hans, style=TONE2) == ['ju2']
    except AssertionError:
        pass
    load_single_dict({ord('桔'): 'jú,jié'})
    assert lazy_pinyin(hans, style=TONE2) == ['ju2']


def test_custom_pinyin_dict2():
    hans = ['同行']
    try:
        assert lazy_pinyin(hans, style=TONE2) == ['to2ng', 'ha2ng']
    except AssertionError:
        pass
    load_phrases_dict({'同行': [['tóng'], ['xíng']]})
    assert lazy_pinyin(hans, style=TONE2) == ['to2ng', 'xi2ng']


def test_custom_pinyin_dict_tone2():
    load_single_dict({ord('桔'): 'ce4,si4'}, style='tone2')
    assert lazy_pinyin('桔', style=TONE2) == ['ce4']
    assert pinyin('桔') == [['cè']]


def test_custom_pinyin_dict2_tone2():
    load_phrases_dict({'同行': [['to4ng'], ['ku1']]}, style='tone2')
    assert lazy_pinyin(['同行'], style=TONE2) == ['to4ng', 'ku1']
    assert pinyin('同行') == [['tòng'], ['kū']]


# yapf: disable
def test_errors():
    hans = (
        ('啊', {'style': TONE2}, [['a']]),
        ('啊a', {'style': TONE2}, [['a'], ['a']]),
        # 非中文字符，没有拼音
        ('⺁', {'style': TONE2}, [['\u2e81']]),
        ('⺁', {'style': TONE2, 'errors': 'ignore'}, []),
        ('⺁', {'style': TONE2, 'errors': 'replace'}, [['2e81']]),
        ('⺁⺁', {'style': TONE2, 'errors': 'replace'}, [['2e812e81']]),
        ('⺁⺁', {'style': TONE2, 'errors': lambda x: ['a' for _ in x]},
         [['a'], ['a']]),
        ('⺁⺁', {'style': TONE2, 'errors': lambda x: [['a', 'b'], ['b', 'c']]},
         [['a'], ['b']]),
        ('⺁⺁', {'style': TONE2, 'heteronym': True,
                'errors': lambda x: [['a', 'b'], ['b', 'c']]},
         [['a', 'b'], ['b', 'c']]),
        # 中文字符，没有拼音
        ('鿅', {'style': TONE2}, [['\u9fc5']]),
        ('鿅', {'style': TONE2, 'errors': 'ignore'}, []),
        ('鿅', {'style': TONE2, 'errors': '233'}, []),
        ('鿅', {'style': TONE2, 'errors': 'replace'}, [['9fc5']]),
        ('鿅', {'style': TONE2, 'errors': lambda x: ['a']}, [['a']]),
        ('鿅', {'style': TONE2, 'errors': lambda x: None}, []),
        ('鿅鿅', {'style': TONE2, 'errors': lambda x: ['a' for _ in x]},
         [['a'], ['a']]),
        ('鿅鿅', {'style': TONE2, 'errors': lambda x: [['a', 'b']]},
         [['a'], ['a']]),
        ('鿅鿅', {'style': TONE2, 'heteronym': True,
                'errors': lambda x: [['a', 'b']]},
         [['a', 'b'], ['a', 'b']]),
    )
    for han in hans:
        assert pinyin(han[0], **han[1]) == han[2]


def test_errors_callable():
    def foobar(chars):
        return 'a' * len(chars)

    class Foobar(object):
        def __call__(self, chars):
            return 'a' * len(chars)

    n = 5
    assert pinyin('あ' * n, errors=foobar) == [['a' * n]]
    assert pinyin('あ' * n, errors=Foobar()) == [['a' * n]]


def test_simple_seg():
    data = {
        '北京abcc': 'be3i ji1ng abcc',
        '你好にほんごРусский язык': 'ni3 ha3o にほんごРусский язык',
    }
    for h, p in data.items():
        assert slug([h], style=TONE2, separator=' ') == p

    hans = '你好にほんごРусский язык'
    ret = 'ni3 ha3o'
    assert slug(hans, style=TONE2, separator=' ', errors=lambda x: None) == ret


data_for_update = [
    # 便宜的发音
    [
        ['便宜'], {'style': TONE2}, ['pia2n', 'yi2']
    ],
    [
        ['便宜从事'], {'style': TONE2}, ['bia4n', 'yi2', 'co2ng', 'shi4']
    ],
    [
        ['便宜施行'], {'style': TONE2}, ['bia4n', 'yi2', 'shi1', 'xi2ng']
    ],
    [
        ['便宜货'], {'style': TONE2}, ['pia2n', 'yi2', 'huo4']
    ],
    [
        ['贪便宜'], {'style': TONE2}, ['ta1n', 'pia2n', 'yi2']
    ],
    [
        ['讨便宜'], {'style': TONE2}, ['ta3o', 'pia2n', 'yi2']
    ],
    [
        ['小便宜'], {'style': TONE2}, ['xia3o', 'pia2n', 'yi2']
    ],
    [
        ['占便宜'], {'style': TONE2}, ['zha4n', 'pia2n', 'yi2']
    ],
    #
    [
        '\u3400', {'style': TONE2}, ['qiu1'],  # CJK 扩展 A:[3400-4DBF]
    ],
    [
        '\u4E00', {'style': TONE2}, ['yi1'],   # CJK 基本:[4E00-9FFF]
    ],
    # [
    #     '\uFA29', {'style': TONE2}, ['da3o'],  # CJK 兼容:[F900-FAFF]
    # ],
    # 误把 yu 放到声母列表了
    ['鱼', {'style': TONE2}, ['yu2']],
    ['鱼', {'style': FINALS}, ['v']],
    ['鱼', {'style': BOPOMOFO}, ['ㄩˊ']],
    ['鱼', {'style': CYRILLIC}, ['юй']],
    ['雨', {'style': TONE2}, ['yu3']],
    ['雨', {'style': FINALS}, ['v']],
    ['雨', {'style': BOPOMOFO}, ['ㄩˇ']],
    ['雨', {'style': CYRILLIC}, ['юй']],
    ['元', {'style': TONE2}, ['yua2n']],
    ['元', {'style': FINALS}, ['van']],
    ['元', {'style': BOPOMOFO}, ['ㄩㄢˊ']],
    ['元', {'style': CYRILLIC}, ['юань2']],
    # y, w 也不是拼音, yu的韵母是v, yi的韵母是i, wu的韵母是u
    ['呀', {'style': INITIALS}, ['']],
    ['呀', {'style': TONE2}, ['ya']],
    ['呀', {'style': FINALS}, ['ia']],
    ['呀', {'style': BOPOMOFO}, ['ㄧㄚ˙']],
    ['呀', {'style': CYRILLIC}, ['я']],
    ['无', {'style': INITIALS}, ['']],
    ['无', {'style': TONE2}, ['wu2']],
    ['无', {'style': FINALS}, ['u']],
    ['无', {'style': FINALS_TONE}, ['ú']],
    ['无', {'style': BOPOMOFO}, ['ㄨˊ']],
    ['无', {'style': CYRILLIC}, ['у2']],
    ['衣', {'style': TONE2}, ['yi1']],
    ['衣', {'style': FINALS}, ['i']],
    ['衣', {'style': BOPOMOFO}, ['ㄧ']],
    ['衣', {'style': CYRILLIC}, ['и1']],
    ['万', {'style': TONE2}, ['wa4n']],
    ['万', {'style': FINALS}, ['uan']],
    ['万', {'style': BOPOMOFO}, ['ㄨㄢˋ']],
    ['万', {'style': CYRILLIC}, ['вань4']],
    # ju, qu, xu 的韵母应该是 v
    ['具', {'style': FINALS_TONE}, ['ǜ']],
    ['具', {'style': FINALS_TONE2}, ['v4']],
    ['具', {'style': FINALS}, ['v']],
    ['具', {'style': BOPOMOFO}, ['ㄐㄩˋ']],
    ['具', {'style': CYRILLIC}, ['цзюй4']],
    ['取', {'style': FINALS_TONE}, ['ǚ']],
    ['取', {'style': FINALS_TONE2}, ['v3']],
    ['取', {'style': FINALS}, ['v']],
    ['取', {'style': BOPOMOFO}, ['ㄑㄩˇ']],
    ['取', {'style': CYRILLIC}, ['цюй3']],
    ['徐', {'style': FINALS_TONE}, ['ǘ']],
    ['徐', {'style': FINALS_TONE2}, ['v2']],
    ['徐', {'style': FINALS}, ['v']],
    ['徐', {'style': BOPOMOFO}, ['ㄒㄩˊ']],
    ['徐', {'style': CYRILLIC}, ['сюй2']],
    # ń
    ['嗯', {'style': NORMAL}, ['n']],
    ['嗯', {'style': TONE}, ['ń']],
    ['嗯', {'style': TONE2}, ['n2']],
    ['嗯', {'style': INITIALS}, ['']],
    ['嗯', {'style': FIRST_LETTER}, ['n']],
    ['嗯', {'style': FINALS}, ['n']],
    ['嗯', {'style': FINALS_TONE}, ['ń']],
    ['嗯', {'style': FINALS_TONE2}, ['n2']],
    ['嗯', {'style': BOPOMOFO}, ['ㄣˊ']],
    ['嗯', {'style': CYRILLIC}, ['н2']],
    # ḿ  \u1e3f  U+1E3F
    ['呣', {'style': NORMAL}, ['m']],
    ['呣', {'style': TONE}, ['ḿ']],
    ['呣', {'style': TONE2}, ['m2']],
    ['呣', {'style': INITIALS}, ['']],
    ['呣', {'style': FIRST_LETTER}, ['m']],
    ['呣', {'style': FINALS}, ['m']],
    ['呣', {'style': FINALS_TONE}, ['ḿ']],
    ['呣', {'style': FINALS_TONE2}, ['m2']],
    ['呣', {'style': BOPOMOFO}, ['ㄇㄨˊ']],
    ['呣', {'style': CYRILLIC}, ['м2']],
    # 41
    ['彷徨', {}, ['pang', 'huang']],
    ['彷徨', {'style': CYRILLIC}, ['пан2', 'хуан2']],
    # 注音
    ['打量', {'style': BOPOMOFO}, ['ㄉㄚˇ', 'ㄌㄧㄤˋ']],
    ['黄山b股', {'style': BOPOMOFO}, ['ㄏㄨㄤˊ', 'ㄕㄢ', 'b', 'ㄍㄨˇ']],
    ['打量', {'style': CYRILLIC}, ['да3', 'лян4']],
    ['黄山b股', {'style': CYRILLIC}, ['хуан2', 'шань1', 'b', 'гу3']],
    # 50
    ['打量', {'style': TONE2}, ['da3', 'lia4ng']],
    ['打量', {'style': TONE3}, ['da3', 'liang4']],
    ['侵略', {'style': TONE2}, ['qi1n', 'lve4']],
    ['侵略', {'style': TONE3}, ['qin1', 'lve4']],
    ['侵略', {'style': FINALS_TONE2}, ['i1n', 've4']],
    ['侵略', {'style': FINALS_TONE3}, ['in1', 've4']],
    ['侵略', {'style': BOPOMOFO}, ['ㄑㄧㄣ', 'ㄌㄩㄝˋ']],
    ['侵略', {'style': CYRILLIC}, ['цинь1', 'люэ4']],
    ['〇', {'style': TONE}, ['líng']],
    # 二次分词
    [['你要', '重新考虑OK'], {'style': TONE}, [
        'nǐ', 'yào', 'chóng', 'xīn', 'kǎo', 'lǜ', 'OK']],
]


@pytest.mark.parametrize('hans, kwargs, result', data_for_update)
def test_update(hans, kwargs, result):
    assert lazy_pinyin(hans, **kwargs) == result


@pytest.mark.skipif(not SUPPORT_UCS4, reason='dont support ucs4')
@pytest.mark.parametrize(
    'han, result', [
        ['\U00020000', ['he']],      # CJK 扩展 B:[20000-2A6DF]
        ['\U0002A79D', ['duo']],      # CJK 扩展 C:[2A700-2B73F]
        # ['\U0002B740', ['wu']],      # CJK 扩展 D:[2B740-2B81D]
        # ['\U0002F80A', ['seng']],    # CJK 兼容扩展:[2F800-2FA1F]
    ]
)
def test_support_ucs4(han, result):
    assert lazy_pinyin(han) == result


@pytest.mark.skipif(SUPPORT_UCS4, reason='support ucs4')
@pytest.mark.parametrize(
    'han', [
        '\U00020000',      # CJK 扩展 B:[20000-2A6DF]
        '\U0002A79D',      # CJK 扩展 C:[2A700-2B73F]
        # '\U0002B740',      # CJK 扩展 D:[2B740-2B81D]
        # '\U0002F80A',      # CJK 兼容扩展:[2F800-2FA1F]
    ]
)
def test_dont_support_ucs4(han):
    assert pinyin(han) == [[han]]


def test_36():
    hans = '两年前七斤喝醉了酒'
    pys = ['liang', 'nian', 'qian', 'qi', 'jin', 'he', 'zui', 'le', 'jiu']
    assert lazy_pinyin(hans) == pys


def test_with_unknown_style():
    assert lazy_pinyin('中国') == ['zhong', 'guo']
    assert lazy_pinyin('中国', style='unknown') == ['zhōng', 'guó']
    assert pinyin('中国') == [['zhōng'], ['guó']]
    assert pinyin('中国', style='unknown') == [['zhōng'], ['guó']]


@pytest.mark.parametrize('kwargs,result', [
    [{}, [['zh\u014dng', 'zh\xf2ng'], ['x\u012bn']]],
    [dict(strict=False), [['zh\u014dng', 'zh\xf2ng'], ['x\u012bn']]],
    [dict(style=NORMAL), [['zhong'], ['xin']]],
    [dict(style=NORMAL, strict=False), [['zhong'], ['xin']]],
    [dict(style=TONE), [['zh\u014dng', 'zh\xf2ng'], ['x\u012bn']]],
    [dict(style=TONE, strict=False), [
        ['zh\u014dng', 'zh\xf2ng'], ['x\u012bn']]],
    [dict(style=TONE2), [['zho1ng', 'zho4ng'], ['xi1n']]],
    [dict(style=TONE2, strict=False), [['zho1ng', 'zho4ng'], ['xi1n']]],
    [dict(style=TONE3), [['zhong1', 'zhong4'], ['xin1']]],
    [dict(style=TONE3, strict=False), [['zhong1', 'zhong4'], ['xin1']]],
    [dict(style=INITIALS), [['zh'], ['x']]],
    [dict(style=INITIALS, strict=False), [['zh'], ['x']]],
    [dict(style=FIRST_LETTER), [['z'], ['x']]],
    [dict(style=FIRST_LETTER, strict=False), [['z'], ['x']]],
    [dict(style=FINALS), [['ong'], ['in']]],
    [dict(style=FINALS, strict=False), [['ong'], ['in']]],
    [dict(style=FINALS_TONE), [['\u014dng', '\xf2ng'], ['\u012bn']]],
    [dict(style=FINALS_TONE, strict=False),  [
        ['\u014dng', '\xf2ng'], ['\u012bn']]],
    [dict(style=FINALS_TONE2), [['o1ng', 'o4ng'], ['i1n']]],
    [dict(style=FINALS_TONE2, strict=False),  [['o1ng', 'o4ng'], ['i1n']]],
    [dict(style=FINALS_TONE3), [['ong1', 'ong4'], ['in1']]],
    [dict(style=FINALS_TONE3, strict=False), [['ong1', 'ong4'], ['in1']]],
])
def test_heteronym_and_style(kwargs, result):
    hans = '中心'
    kwargs['heteronym'] = True
    assert pinyin(hans, **kwargs) == result


@pytest.mark.parametrize('kwargs,result', [
    [{}, [['zhāo'], ['yáng']]],
    [dict(heteronym=True), [['zhāo', 'cháo'], ['yáng']]],
    [dict(strict=False), [['zhāo'], ['yáng']]],
    [dict(strict=False, heteronym=True), [['zhāo', 'cháo'], ['yáng']]],
    [dict(style=NORMAL), [['zhao'], ['yang']]],
    [dict(style=NORMAL, heteronym=True), [['zhao', 'chao'], ['yang']]],
    [dict(style=NORMAL, strict=False), [['zhao'], ['yang']]],
    [dict(style=NORMAL, strict=False, heteronym=True), [['zhao', 'chao'],
                                                        ['yang']]],
    [dict(style=TONE), [['zhāo'], ['yáng']]],
    [dict(style=TONE, heteronym=True), [['zhāo', 'cháo'], ['yáng']]],
    [dict(style=TONE, strict=False), [['zhāo'], ['yáng']]],
    [dict(style=TONE, strict=False, heteronym=True), [['zhāo', 'cháo'],
                                                      ['yáng']]],
    [dict(style=TONE2), [['zha1o'], ['ya2ng']]],
    [dict(style=TONE2, heteronym=True), [['zha1o', 'cha2o'], ['ya2ng']]],
    [dict(style=TONE2, strict=False), [['zha1o'], ['ya2ng']]],
    [dict(style=TONE2, strict=False, heteronym=True), [['zha1o', 'cha2o'],
                                                       ['ya2ng']]],
    [dict(style=TONE3), [['zhao1'], ['yang2']]],
    [dict(style=TONE3, heteronym=True), [['zhao1', 'chao2'], ['yang2']]],
    [dict(style=TONE3, strict=False), [['zhao1'], ['yang2']]],
    [dict(style=TONE3, strict=False, heteronym=True), [['zhao1', 'chao2'],
                                                       ['yang2']]],
    [dict(style=INITIALS), [['zh'], ['']]],
    [dict(style=INITIALS, heteronym=True), [['zh', 'ch'], ['']]],
    [dict(style=INITIALS, strict=False), [['zh'], ['y']]],
    [dict(style=INITIALS, strict=False, heteronym=True), [['zh', 'ch'],
                                                          ['y']]],
    [dict(style=FIRST_LETTER), [['z'], ['y']]],
    [dict(style=FIRST_LETTER, heteronym=True), [['z', 'c'], ['y']]],
    [dict(style=FIRST_LETTER, strict=False), [['z'], ['y']]],
    [dict(style=FIRST_LETTER, strict=False, heteronym=True), [['z', 'c'],
                                                              ['y']]],
    [dict(style=FINALS), [['ao'], ['iang']]],
    [dict(style=FINALS, heteronym=True), [['ao'], ['iang']]],
    [dict(style=FINALS, strict=False), [['ao'], ['ang']]],
    [dict(style=FINALS, strict=False, heteronym=True), [['ao'], ['ang']]],
    [dict(style=FINALS_TONE), [['āo'], ['iáng']]],
    [dict(style=FINALS_TONE, heteronym=True), [['āo', 'áo'], ['iáng']]],
    [dict(style=FINALS_TONE, strict=False),  [['āo'], ['áng']]],
    [dict(style=FINALS_TONE, strict=False, heteronym=True),  [['āo', 'áo'],
                                                              ['áng']]],
    [dict(style=FINALS_TONE2), [['a1o'], ['ia2ng']]],
    [dict(style=FINALS_TONE2, heteronym=True), [['a1o', 'a2o'], ['ia2ng']]],
    [dict(style=FINALS_TONE2, strict=False),  [['a1o'], ['a2ng']]],
    [dict(style=FINALS_TONE2, strict=False, heteronym=True),  [['a1o', 'a2o'],
                                                               ['a2ng']]],
    [dict(style=FINALS_TONE3), [['ao1'], ['iang2']]],
    [dict(style=FINALS_TONE3, heteronym=True), [['ao1', 'ao2'], ['iang2']]],
    [dict(style=FINALS_TONE3, strict=False), [['ao1'], ['ang2']]],
    [dict(style=FINALS_TONE3, strict=False, heteronym=True), [['ao1', 'ao2'],
                                                              ['ang2']]],
])
def test_heteronym_and_style_phrase(kwargs, result):
    hans = '朝阳'
    assert pinyin(hans, **kwargs) == result


def test_m4():
    # U+5463: ḿ,móu,m̀  # 呣
    han = '呣'
    assert pinyin(han) == [['ḿ']]
    assert pinyin(han, heteronym=True) == [['ḿ', 'm̀', 'móu']]
    assert pinyin(
        han, heteronym=True, style=NORMAL) == [['m', 'mou']]
    assert pinyin(
        han, heteronym=True, style=TONE) == [['ḿ', 'm̀', 'móu']]
    assert pinyin(
        han, heteronym=True, style=TONE2) == [['m2', 'm4', 'mo2u']]
    assert pinyin(
        han, heteronym=True, style=TONE3) == [['m2', 'm4', 'mou2']]
    assert pinyin(
        han, heteronym=True, style=INITIALS) == [['', 'm']]  # TODO: fix ''
    assert pinyin(
        han, heteronym=True, style=FIRST_LETTER) == [['m']]
    assert pinyin(
        han, heteronym=True, style=FINALS) == [['m', 'ou']]
    assert pinyin(
        han, heteronym=True, style=FINALS_TONE) == [['ḿ', 'm̀', 'óu']]
    assert pinyin(
        han, heteronym=True, style=FINALS_TONE2) == [['m2', 'm4', 'o2u']]
    assert pinyin(
        han, heteronym=True, style=FINALS_TONE3) == [['m2', 'm4', 'ou2']]


@pytest.mark.parametrize('han,style,expect', [
    ['呣', Style.TONE, ['ḿ', 'm̀']],
    ['呣', Style.TONE2, ['m2', 'm4']],
    ['嘸', Style.TONE, ['m̄', 'ḿ']],
    ['嘸', Style.TONE2, ['m1', 'm2']],
    ['誒', Style.TONE, ['ê̄', 'ế', 'ê̌', 'ề']],
    ['誒', Style.TONE2, ['ê1', 'ê2', 'ê3', 'ê4']],
])
def test_m_e(han, style, expect):
    result = pinyin(han, style=style, heteronym=True)
    assert len(result) == 1
    assert (set(result[0]) & set(expect)) == set(expect)


if __name__ == '__main__':
    import pytest
    pytest.cmdline.main()
