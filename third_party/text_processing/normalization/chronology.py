import re
from .num import verbalize_cardinal, verbalize_digit, num2str, DIGITS

def _time_num2str(num_string: str) -> str:
    """A special case for verbalizing number in time."""
    result = num2str(num_string.lstrip('0'))
    if num_string.startswith('0'):
        result = DIGITS['0'] + result
    return result

# 时刻表达式
RE_TIME = re.compile(
    r'([0-1]?[0-9]|2[0-3])'
    r':([0-5][0-9])'
    r'(:([0-5][0-9]))?'
)
def replace_time(match: re.Match) -> str:
    hour = match.group(1)
    minute = match.group(2)
    second = match.group(4)
    
    result = f"{num2str(hour)}点"
    if minute.lstrip('0'):
        result += f"{_time_num2str(minute)}分"
    if second and second.lstrip('0'):
        result += f"{_time_num2str(second)}秒"
    return result


RE_DATE = re.compile(
    r'(\d{4}|\d{2})年'
    r'((0?[1-9]|1[0-2])月)?'
    r'(((0?[1-9])|((1|2)[0-9])|30|31)([日号]))?'
)
def replace_date(match: re.Match) -> str:
    year = match.group(1)
    month = match.group(3)
    day = match.group(5)
    result = ""
    if year:
        result += f"{verbalize_digit(year)}年"
    if month:
        result += f"{verbalize_cardinal(month)}月"
    if day:
        result += f"{verbalize_cardinal(day)}{match.group(9)}"
    return result

# 用 / 或者 - 分隔的 YY/MM/DD 或者 YY-MM-DD 日期
RE_DATE2 = re.compile(
    r'(\d{4})([- /.])(0[1-9]|1[012])\2(0[1-9]|[12][0-9]|3[01])'
)
def replace_date2(match: re.Match) -> str:
    year = match.group(1)
    month = match.group(3)
    day = match.group(4)
    result = ""
    if year:
        result += f"{verbalize_digit(year)}年"
    if month:
        result += f"{verbalize_cardinal(month)}月"
    if day:
        result += f"{verbalize_cardinal(day)}日"
    return result