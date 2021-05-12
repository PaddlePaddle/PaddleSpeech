#!/usr/bin/env python3

from pypinyin import pinyin, Style, lazy_pinyin, slug
from pypinyin.seg.simpleseg import simple_seg


def test_import_all():
    pinyin('啦啦啦')  # noqa
    pinyin('啦啦啦', Style.TONE2)  # noqa
    lazy_pinyin('啦啦啦')  # noqa
    slug('啦啦啦')  # noqa


def test_simple_seg():
    assert simple_seg('啦啦') == ['啦啦']
    assert simple_seg('啦啦abc') == ['啦啦', 'abc']
    assert simple_seg('&##啦啦abc') == ['&##', '啦啦', 'abc']
    assert simple_seg('&#哦#啦啦abc') == ['&#', '哦', '#', '啦啦', 'abc']
    assert simple_seg('哦ほ#') == ['哦', 'ほ#']
    assert simple_seg(['啦啦']) == ['啦啦']
    assert simple_seg(['啦啦', 'abc']) == ['啦啦', 'abc']
    assert simple_seg('哦ほ#哪') == ['哦', 'ほ#', '哪']
    assert simple_seg('哦ほ#哪#') == ['哦', 'ほ#', '哪', '#']
    assert simple_seg('你好啊 --') == ['你好啊', ' --']
    assert simple_seg('啊 -- ') == ['啊', ' -- ']
    assert simple_seg('你好啊 -- 那') == ['你好啊', ' -- ', '那']
    assert simple_seg('啊 -- 你好那 ') == ['啊', ' -- ', '你好那', ' ']
    assert simple_seg('a 你好啊 -- 那 ') == ['a ', '你好啊', ' -- ', '那', ' ']
    assert simple_seg('a啊 -- 你好那 ') == ['a', '啊', ' -- ', '你好那', ' ']


def test_issue_205():
    assert pinyin('金融行业', Style.FIRST_LETTER)[2] == ['h']
    assert pinyin('军工行业', Style.FIRST_LETTER)[2] == ['h']
    assert pinyin('浦发银行', Style.FIRST_LETTER)[3] == ['h']
    assert pinyin('交通银行', Style.FIRST_LETTER)[3] == ['h']


if __name__ == '__main__':
    import pytest
    pytest.cmdline.main()
