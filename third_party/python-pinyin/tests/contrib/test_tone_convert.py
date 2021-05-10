from pytest import mark

from pypinyin.contrib.tone_convert import (
    tone_to_normal,
    tone_to_tone2,
    tone2_to_tone,
    tone_to_tone3,
    tone3_to_tone,
    tone2_to_normal,
    tone2_to_tone3,
    tone3_to_tone2,
    tone3_to_normal,
    to_normal,
    to_tone,
    to_tone2,
    to_tone3, )


@mark.parametrize('pinyin,result', [
    ['zhōng', 'zhong'],
    ['ān', 'an'],
    ['yuè', 'yue'],
    ['er', 'er'],
    ['nǚ', 'nv'],
    ['ā', 'a'],
    ['a', 'a'],
])
def test_tone_to_normal(pinyin, result):
    assert tone_to_normal(pinyin) == result

    assert to_normal(pinyin) == result
    assert to_normal(result) == result


@mark.parametrize('pinyin,v_to_u,result', [
    ['nǚ', False, 'nv'],
    ['nǚ', True, 'nü'],
])
def test_tone_to_normal_with_v_to_u(pinyin, v_to_u, result):
    assert tone_to_normal(pinyin, v_to_u=v_to_u) == result
    assert to_normal(pinyin, v_to_u=v_to_u) == result


@mark.parametrize('pinyin,result', [
    ['zhōng', 'zho1ng'],
    ['ān', 'a1n'],
    ['yuè', 'yue4'],
    ['er', 'er'],
    ['nǚ', 'nv3'],
    ['ā', 'a1'],
    ['a', 'a'],
    ['shang', 'shang'],
])
def test_tone_tone2(pinyin, result):
    assert tone_to_tone2(pinyin) == result
    assert to_tone2(pinyin) == result

    assert tone2_to_tone(result) == pinyin

    assert to_tone(result) == pinyin
    assert to_tone(pinyin) == pinyin
    assert to_tone2(result) == result


@mark.parametrize('pinyin,neutral_tone_with_5,result', [
    ['shang', False, 'shang'],
    ['shang', True, 'sha5ng'],
])
def test_tone_tone2_with_neutral_tone_with_5(pinyin, neutral_tone_with_5,
                                             result):
    assert tone_to_tone2(
        pinyin, neutral_tone_with_5=neutral_tone_with_5) == result
    assert to_tone2(pinyin, neutral_tone_with_5=neutral_tone_with_5) == result

    assert tone2_to_tone(result) == pinyin
    assert to_tone(result) == pinyin


@mark.parametrize('pinyin,v_to_u,result', [
    ['nǚ', False, 'nv3'],
    ['nǚ', True, 'nü3'],
])
def test_tone_tone2_with_v_to_u(pinyin, v_to_u, result):
    assert tone_to_tone2(pinyin, v_to_u=v_to_u) == result
    assert to_tone2(pinyin, v_to_u=v_to_u) == result

    assert tone2_to_tone(result) == pinyin
    assert to_tone(result) == pinyin


@mark.parametrize('pinyin,result', [
    ['zhōng', 'zhong1'],
    ['ān', 'an1'],
    ['yuè', 'yue4'],
    ['er', 'er'],
    ['nǚ', 'nv3'],
    ['ā', 'a1'],
    ['a', 'a'],
    ['shang', 'shang'],
])
def test_tone_tone3(pinyin, result):
    assert tone_to_tone3(pinyin) == result
    assert to_tone3(pinyin) == result

    assert tone3_to_tone(result) == pinyin

    assert to_tone(result) == pinyin
    assert to_tone(pinyin) == pinyin
    assert to_tone3(result) == result


@mark.parametrize('pinyin,neutral_tone_with_5,result', [
    ['shang', False, 'shang'],
    ['shang', True, 'shang5'],
])
def test_tone_tone3_with_neutral_tone_with_5(pinyin, neutral_tone_with_5,
                                             result):
    assert tone_to_tone3(
        pinyin, neutral_tone_with_5=neutral_tone_with_5) == result
    assert to_tone3(pinyin, neutral_tone_with_5=neutral_tone_with_5) == result

    assert tone3_to_tone(result) == pinyin
    assert to_tone(result) == pinyin


@mark.parametrize('pinyin,v_to_u,result', [
    ['nǚ', False, 'nv3'],
    ['nǚ', True, 'nü3'],
])
def test_tone_tone3_with_v_to_u(pinyin, v_to_u, result):
    assert tone_to_tone3(pinyin, v_to_u=v_to_u) == result
    assert to_tone3(pinyin, v_to_u=v_to_u) == result

    assert tone3_to_tone(result) == pinyin
    assert to_tone(result) == pinyin


@mark.parametrize('pinyin,result', [
    ['zho1ng', 'zhong1'],
    ['a1n', 'an1'],
    ['yue4', 'yue4'],
    ['er', 'er'],
    ['nv3', 'nv3'],
    ['nü3', 'nü3'],
    ['a1', 'a1'],
    ['a', 'a'],
    ['shang', 'shang'],
    ['sha5ng', 'shang5'],
])
def test_tone2_tone3(pinyin, result):
    assert tone2_to_tone3(pinyin) == result
    assert to_tone3(pinyin) == result

    assert tone3_to_tone2(result) == pinyin
    assert to_tone2(result) == pinyin
    assert to_tone2(pinyin) == pinyin


@mark.parametrize('pinyin,result', [
    ['zho1ng', 'zhong'],
    ['a1n', 'an'],
    ['yue4', 'yue'],
    ['er', 'er'],
    ['nv3', 'nv'],
    ['nü3', 'nü'],
    ['a1', 'a'],
    ['a', 'a'],
    ['shang', 'shang'],
    ['sha5ng', 'shang'],
])
def test_tone2_to_normal(pinyin, result):
    assert tone2_to_normal(pinyin) == result

    assert to_normal(pinyin) == result
    assert to_normal(result) == result


@mark.parametrize('pinyin,v_to_u,result', [
    ['nv3', False, 'nv'],
    ['nv3', True, 'nü'],
    ['nü3', False, 'nü'],
    ['nü3', True, 'nü'],
])
def test_tone2_to_normal_with_v_to_u(pinyin, v_to_u, result):
    assert tone2_to_normal(pinyin, v_to_u=v_to_u) == result

    assert to_normal(pinyin, v_to_u=v_to_u) == result
    assert to_normal(result) == result


@mark.parametrize('pinyin,result', [
    ['zhong1', 'zhong'],
    ['an1', 'an'],
    ['yue4', 'yue'],
    ['er', 'er'],
    ['nv3', 'nv'],
    ['nü3', 'nü'],
    ['a1', 'a'],
    ['a', 'a'],
    ['shang', 'shang'],
    ['shang5', 'shang'],
])
def test_tone3_to_normal(pinyin, result):
    assert tone3_to_normal(pinyin) == result
    assert to_normal(pinyin) == result


@mark.parametrize('pinyin,v_to_u,result', [
    ['nv3', False, 'nv'],
    ['nv3', True, 'nü'],
    ['nü3', False, 'nü'],
    ['nü3', True, 'nü'],
])
def test_tone3_to_normal_with_v_to_u(pinyin, v_to_u, result):
    assert tone3_to_normal(pinyin, v_to_u=v_to_u) == result
    assert to_normal(pinyin, v_to_u=v_to_u) == result
