#!/usr/bin/env python3

import copy
try:
    from importlib import reload
except ImportError:
    pass
import os
import sys

import pytest


@pytest.fixture(scope='function')
def cleanup():
    _clean()
    try:
        yield
    finally:
        _clean()


def _clean():
    for module in copy.copy(sys.modules):
        if module.startswith('pypinyin'):
            sys.modules.pop(module, None)


def test_env(cleanup):
    os.environ['PYPINYIN_NO_PHRASES'] = 'true'
    import pypinyin.core  # noqa

    assert pypinyin.core.PHRASES_DICT == {}
    assert pypinyin.core.seg('北京') == ['北京']


def test_no_copy(cleanup):
    """ 禁用copy操作的测试 """
    import pypinyin.core  # noqa

    assert pypinyin.core.PINYIN_DICT is not pypinyin.pinyin_dict.pinyin_dict

    os.environ['PYPINYIN_NO_DICT_COPY'] = 'true'
    reload(pypinyin.constants)
    assert pypinyin.constants.PINYIN_DICT is pypinyin.pinyin_dict.pinyin_dict
