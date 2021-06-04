import re
from .num import verbalize_digit


# 规范化固话/手机号码
# 手机
# http://www.jihaoba.com/news/show/13680
# 移动：139、138、137、136、135、134、159、158、157、150、151、152、188、187、182、183、184、178、198
# 联通：130、131、132、156、155、186、185、176
# 电信：133、153、189、180、181、177
RE_MOBILE_PHONE= re.compile(
    r"(?<!\d)((\+?86 ?)?1([38]\d|5[0-35-9]|7[678]|9[89])\d{8})(?!\d)")
RE_TELEPHONE = re.compile(
    r"(?<!\d)((0(10|2[1-3]|[3-9]\d{2})-?)?[1-9]\d{6,7})(?!\d)")


def phone2str(phone_string: str, mobile=True) -> str:
    if mobile:
        sp_parts = phone_string.strip('+').split()
        result = ''.join(
            [verbalize_digit(part, alt_one=True) for part in sp_parts])
        return result
    else:
        sil_parts = phone_string.split('-')
        result = ''.join(
            [verbalize_digit(part, alt_one=True) for part in sil_parts])
        return result


def replace_phone(match: re.Match) -> str:
    return phone2str(match.group(0))
