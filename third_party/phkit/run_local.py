#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2019/12/1
"""
local
"""
import logging

logging.basicConfig(level=logging.INFO)


def run_text2phoneme():
    from phkit.chinese.sequence import text2phoneme, text2sequence
    text = "汉字转音素，TTS：《Text to speech》。"
    # text = "岂有此理"
    # text = "我的儿子玩会儿"
    out = text2phoneme(text)
    print(out)
    # ['h', 'an', '4', '-', 'z', 'iy', '4', '-', 'zh', 'uan', '3', '-', 'ii', 'in', '1', '-', 's', 'u', '4', '-', ',',
    # 'Tt', 'Tt', 'Ss', ':', '(', 'T', 'E', 'X', 'T', '#', 'T', 'O', '#', 'S', 'P', 'E', 'E', 'C', 'H', ')', '.', '-',
    #  '~', '_']
    out = text2sequence(text)
    print(out)
    # [11, 32, 76, 2, 28, 51, 76, 2, 29, 59, 75, 2, 12, 46, 73, 2, 22, 56, 76, 2, 133, 97, 97, 96, 135, 138, 123, 108,
    # 127, 123, 137, 123, 118, 137, 122, 119, 108, 108, 106, 111, 139, 132, 2, 1, 0]


def run_english():
    from phkit.english import text_to_sequence, sequence_to_text
    from phkit.english.cmudict import CMUDict
    text = "text to speech"
    cmupath = 'phkit/english/cmu_dictionary'
    cmudict = CMUDict(cmupath)
    seq = text_to_sequence(text, cleaner_names=["english_cleaners"], dictionary=cmudict)
    print(seq)
    txt = sequence_to_text(seq)
    print(txt)


if __name__ == "__main__":
    print(__file__)
    run_text2phoneme()
    run_english()
