"""Style.INITIALS 风格"""
from typing import Any
from typing import Text

from pypinyin.constants import Style
from pypinyin.style import register
from pypinyin.style._utils import get_initials


@register(Style.INITIALS)
def convert(pinyin: Text, **kwargs: Any) -> Text:
    strict = kwargs.get('strict')
    return get_initials(pinyin, strict)
