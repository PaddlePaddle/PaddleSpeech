"""
Rules to verbalize numbers into Chinese characters.
https://zh.wikipedia.org/wiki/中文数字#現代中文
"""

import re
from typing import List
from collections import OrderedDict

DIGITS = {str(i): tran for i, tran in enumerate('零一二三四五六七八九')}
UNITS = OrderedDict({
    1: '十',
    2: '百',
    3: '千',
    4: '万',
    8: '亿',
})

# 分数表达式
RE_FRAC = re.compile(r'(-?)(\d+)/(\d+)')
def replace_frac(match: re.Match) -> str:
    sign = match.group(1)
    nominator = match.group(2)
    denominator = match.group(3)
    sign: str = "负" if sign else ""
    nominator: str = num2str(nominator)
    denominator: str = num2str(denominator)
    result = f"{sign}{denominator}分之{nominator}"
    return result
    

# 百分数表达式
RE_PERCENTAGE = re.compile(r'(-?)(\d+(\.\d+)?)%')
def replace_percentage(match: re.Match) -> str:
    sign = match.group(1)
    percent = match.group(2)
    sign: str = "负" if sign else ""
    percent: str = num2str(percent)
    result = f"{sign}百分之{percent}"
    return result

# 整数表达式
# 带负号或者不带负号的整数 12, -10
RE_INTEGER = re.compile(
    r'(-?)'
    r'(\d+)'
)

# 编号-无符号整形
# 00078
RE_DEFAULT_NUM = re.compile(r'\d{4}\d*')
def replace_default_num(match: re.Match):
    number = match.group(0)
    return verbalize_digit(number)

# 数字表达式
# 1. 整数: -10, 10;
# 2. 浮点数: 10.2, -0.3
# 3. 不带符号和整数部分的纯浮点数: .22, .38   
RE_NUMBER = re.compile(
    r'(-?)((\d+)(\.\d+)?)'
    r'|(\.(\d+))'
)
def replace_number(match: re.Match) -> str:
    sign = match.group(1)
    number = match.group(2)
    pure_decimal = match.group(5)
    if pure_decimal:
        result = num2str(pure_decimal)
    else:
        sign: str = "负" if sign else ""
        number: str = num2str(number)
        result = f"{sign}{number}"
    return result

# 范围表达式
# 12-23, 12~23
RE_RANGE = re.compile(
    r'(\d+)[-~](\d+)'
)
def replace_range(match: re.Match) -> str:
    first, second = match.group(1), match.group(2)
    first: str = num2str(first)
    second: str = num2str(second)
    result = f"{first}到{second}"
    return result


def _get_value(value_string: str, use_zero: bool=True) -> List[str]:
    stripped = value_string.lstrip('0')
    if len(stripped) == 0:
        return []
    elif len(stripped) == 1:
        if use_zero and len(stripped) < len(value_string):
            return [DIGITS['0'], DIGITS[stripped]]
        else:
            return [DIGITS[stripped]]
    else:
        largest_unit = next(power for power in reversed(UNITS.keys()) if power < len(stripped))
        first_part = value_string[:-largest_unit]
        second_part = value_string[-largest_unit:]
        return _get_value(first_part) + [UNITS[largest_unit]] + _get_value(second_part)

def verbalize_cardinal(value_string: str) -> str:
    if not value_string:
        return ''
    
    # 000 -> '零' , 0 -> '零'
    value_string = value_string.lstrip('0')
    if len(value_string) == 0:
        return DIGITS['0']
    
    result_symbols = _get_value(value_string)
    # verbalized number starting with '一十*' is abbreviated as `十*`
    if len(result_symbols) >= 2 and result_symbols[0] == DIGITS['1'] and result_symbols[1] == UNITS[1]:
        result_symbols = result_symbols[1:]
    return ''.join(result_symbols)

def verbalize_digit(value_string: str, alt_one=False) -> str:
    result_symbols = [DIGITS[digit] for digit in value_string]
    result = ''.join(result_symbols)
    if alt_one:
        result.replace("一", "幺")
    return result

def num2str(value_string: str) -> str:
    integer_decimal = value_string.split('.')
    if len(integer_decimal) == 1:
        integer = integer_decimal[0]
        decimal = ''
    elif len(integer_decimal) == 2:
        integer, decimal = integer_decimal
    else:
        raise ValueError(f"The value string: '${value_string}' has more than one point in it.")
    
    result = verbalize_cardinal(integer)

    decimal = decimal.rstrip('0')
    if decimal:
        # '.22' is verbalized as '点二二'
        # '3.20' is verbalized as '三点二
        result += '点' + verbalize_digit(decimal)
    return result












