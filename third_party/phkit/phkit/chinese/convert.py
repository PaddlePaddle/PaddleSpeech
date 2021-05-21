#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/17
"""
#### convert
文本转换。

全角半角转换，简体繁体转换。
"""
from hanziconv import hanziconv

hc = hanziconv.HanziConv()

# 繁体转简体
fan2jian = hc.toSimplified

# 简体转繁体
jian2fan = hc.toTraditional

# 半角转全角映射表
ban2quan_dict = {i: i + 65248 for i in range(33, 127)}
ban2quan_dict.update({32: 12288})

# 全角转半角映射表
quan2ban_dict = {v: k for k, v in ban2quan_dict.items()}


def ban2quan(text: str):
    """
    半角转全角
    :param text:
    :return:
    """
    return text.translate(ban2quan_dict)


def quan2ban(text: str):
    """
    全角转半角
    :param text:
    :return:
    """
    return text.translate(quan2ban_dict)


if __name__ == "__main__":
    assert ban2quan("aA1 ,:$。、") == "ａＡ１　，：＄。、"
    assert quan2ban("ａＡ１　，：＄。、") == "aA1 ,:$。、"
    assert jian2fan("中国语言") == "中國語言"
    assert jian2fan("中國語言") == "中国语言"
