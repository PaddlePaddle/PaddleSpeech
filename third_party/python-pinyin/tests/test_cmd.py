#!/usr/bin/env python3

from pypinyin.runner import get_parser


def test_default():
    options = get_parser().parse_args(['你好'])
    assert options.func == 'pinyin'
    assert options.style == 'zh4ao'
    assert options.separator == '-'
    assert not options.heteronym
    assert options.hans == '你好'
    assert options.errors == 'default'


def test_custom():
    options = get_parser().parse_args([
        '--func', 'slug', '--style', 'zhao', '--separator', ' ', '--errors',
        'ignore', '--heteronym', '你好啊'
    ])
    assert options.func == 'slug'
    assert options.style == 'zhao'
    assert options.separator == ' '
    assert options.errors == 'ignore'
    assert options.heteronym
    assert options.hans == '你好啊'


if __name__ == '__main__':
    import pytest
    pytest.cmdline.main()
