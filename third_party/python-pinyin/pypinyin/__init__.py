"""汉字拼音转换工具."""
from pypinyin.constants import BOPOMOFO
from pypinyin.constants import BOPOMOFO_FIRST
from pypinyin.constants import CYRILLIC
from pypinyin.constants import CYRILLIC_FIRST
from pypinyin.constants import FINALS
from pypinyin.constants import FINALS_TONE
from pypinyin.constants import FINALS_TONE2
from pypinyin.constants import FINALS_TONE3
from pypinyin.constants import FIRST_LETTER
from pypinyin.constants import INITIALS
from pypinyin.constants import NORMAL
from pypinyin.constants import Style
from pypinyin.constants import STYLE_BOPOMOFO
from pypinyin.constants import STYLE_BOPOMOFO_FIRST
from pypinyin.constants import STYLE_CYRILLIC
from pypinyin.constants import STYLE_CYRILLIC_FIRST
from pypinyin.constants import STYLE_FINALS
from pypinyin.constants import STYLE_FINALS_TONE
from pypinyin.constants import STYLE_FINALS_TONE2
from pypinyin.constants import STYLE_FINALS_TONE3
from pypinyin.constants import STYLE_FIRST_LETTER
from pypinyin.constants import STYLE_INITIALS
from pypinyin.constants import STYLE_NORMAL
from pypinyin.constants import STYLE_TONE
from pypinyin.constants import STYLE_TONE2
from pypinyin.constants import STYLE_TONE3
from pypinyin.constants import TONE
from pypinyin.constants import TONE2
from pypinyin.constants import TONE3
from pypinyin.core import lazy_pinyin
from pypinyin.core import load_phrases_dict
from pypinyin.core import load_single_dict
from pypinyin.core import pinyin
from pypinyin.core import slug

__all__ = [
    'pinyin', 'lazy_pinyin', 'slug', 'load_single_dict', 'load_phrases_dict',
    'Style', 'STYLE_NORMAL', 'NORMAL', 'STYLE_TONE', 'TONE', 'STYLE_TONE2',
    'TONE2', 'STYLE_TONE3', 'TONE3', 'STYLE_INITIALS', 'INITIALS',
    'STYLE_FINALS', 'FINALS', 'STYLE_FINALS_TONE', 'FINALS_TONE',
    'STYLE_FINALS_TONE2', 'FINALS_TONE2', 'STYLE_FINALS_TONE3', 'FINALS_TONE3',
    'STYLE_FIRST_LETTER', 'FIRST_LETTER', 'STYLE_BOPOMOFO', 'BOPOMOFO',
    'STYLE_BOPOMOFO_FIRST', 'BOPOMOFO_FIRST', 'STYLE_CYRILLIC', 'CYRILLIC',
    'STYLE_CYRILLIC_FIRST', 'CYRILLIC_FIRST'
]

__title__ = 'pypinyin'
__version__ = '0.41.0'
__license__ = 'MIT'
__author__ = 'Hui Zhang'
__copyright__ = 'Copyright (c) 2021 Hui Zhang'
