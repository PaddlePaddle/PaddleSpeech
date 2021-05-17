"""Provides CC-CEDICT character constants."""

from . import simplified
from . import traditional
from . import all

#: A string containing all Simplified characters according to CC-CEDICT.
simp = simplified = simplified.CHARACTERS

#: A string containing all Traditional characters according to CC-CEDICT.
trad = traditional = traditional.CHARACTERS

#: A string containing all Chinese characters found in CC-CEDICT.
all = all.CHARACTERS
