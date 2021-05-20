#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/16
"""
#### number
数字读法。

按数值大小读，一个一个数字读。
"""
import re

_number_cn = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
_number_level = ['千', '百', '十', '万', '千', '百', '十', '亿', '千', '百', '十', '万', '千', '百', '十', '个']
_zero = _number_cn[0]
_ten_re = re.compile(r'^一十')
_grade_level = {'万', '亿', '个'}
_number_group_re = re.compile(r"([0-9]+)")


def say_digit(num: str):
    outs = []
    for zi in num:
        outs.append(_number_cn[int(zi)])
    return ''.join(outs)


def say_number(num: str):
    x = str(int(num))
    if x == '0':
        return _number_cn[0]
    elif len(x) > 16:
        return num
    length = len(x)
    outs = []
    for num, zi in enumerate(x):
        a = _number_cn[int(zi)]
        b = _number_level[len(_number_level) - length + num]
        if a != _zero:
            outs.append(a)
            outs.append(b)
        else:
            if b in _grade_level:
                if outs[-1] != _zero:
                    outs.append(b)
                else:
                    outs[-1] = b
            else:
                if outs[-1] != _zero:
                    outs.append(a)
    out = ''.join(outs[:-1])
    out = _ten_re.sub(r'十', out)
    return out


def say_decimal(num: str):
    z, x = num.split('.')
    z_cn = say_number(z)
    x_cn = say_digit(x)
    return z_cn + '点' + x_cn


def convert_number(text):
    parts = _number_group_re.split(text)
    outs = []
    for elem in parts:
        if elem.isdigit():
            if len(elem) <= 9:
                outs.append(say_number(elem))
            else:
                outs.append(say_digit(elem))
        else:
            outs.append(elem)
    return ''.join(outs)

if __name__ == "__main__":
    print(__file__)
    assert say_number("1234567890123456") == "一千二百三十四万五千六百七十八亿九千零一十二万三千四百五十六"
    assert say_digit("123456") == "一二三四五六"
    assert say_decimal("3.14") == "三点一四"
    assert convert_number("hello314.1592and2718281828") == "hello三百一十四.一千五百九十二and二七一八二八一八二八"
