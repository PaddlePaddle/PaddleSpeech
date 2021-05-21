#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/18
"""
"""


def test_phkit():
    from phkit import text2phoneme, text2sequence, symbol_chinese
    from phkit import chinese_sequence_to_text, chinese_text_to_sequence
    text = "汉字转音素，TTS：《Text to speech》。"
    target_ph = ['h', 'an', '4', '-', 'z', 'iy', '4', '-', 'zh', 'uan', '3', '-', 'ii', 'in', '1', '-', 's', 'u', '4',
                 '-', ',', '-',
                 'Tt', 'Tt', 'Ss', '-', ':', '-', '(', '-', 'T', 'E', 'X', 'T', '-', '#', '-', 'T', 'O', '-', '#', '-',
                 'S', 'P', 'E', 'E', 'C', 'H', '-', ')', '-', '.', '-', '~', '_']

    result = text2phoneme(text)
    assert result == target_ph

    target_seq = [11, 32, 74, 2, 28, 51, 74, 2, 29, 59, 73, 2, 12, 46, 71, 2, 22, 56, 74, 2, 131, 2, 95, 95, 94, 2, 133,
                  2, 136, 2, 121,
                  106, 125, 121, 2, 135, 2, 121, 116, 2, 135, 2, 120, 117, 106, 106, 104, 109, 2, 137, 2, 130, 2, 1, 0]

    result = text2sequence(text)
    assert result == target_seq

    result = chinese_text_to_sequence(text)
    assert result == target_seq

    target_ph = ' '.join(target_ph)
    result = chinese_sequence_to_text(result)
    assert result == target_ph

    assert len(symbol_chinese) == 145

    text = "岂有此理"
    target = ['q', 'i', '2', '-', 'ii', 'iu', '3', '-', 'c', 'iy', '2', '-', 'l', 'i', '3', '-', '~', '_']
    result = text2phoneme(text)
    assert result == target

    text = "我的儿子玩会儿"
    target = ['uu', 'uo', '3', '-', 'd', 'e', '5', '-', 'ee', 'er', '2', '-', 'z', 'iy', '5', '-', 'uu', 'uan', '2',
              '-', 'h', 'ui', '4', '-', 'ee', 'er', '5', '-', '~', '_']
    result = text2phoneme(text)
    assert result == target


if __name__ == "__main__":
    print(__file__)
    test_phkit()
