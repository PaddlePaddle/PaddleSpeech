
"""Constants for processing Pinyin strings."""
from string import whitespace

_a = 'a\u0101\u00E0\u00E1\u01CE'
_e = 'e\u0113\u00E9\u011B\u00E8'
_i = 'i\u012B\u00ED\u01D0\u00EC'
_o = 'o\u014D\u00F3\u01D2\u00F2'
_u = 'u\u016B\u00FA\u01D4\u00F9'
_v = 'v\u00FC\u01D6\u01D8\u01DA\u01DC'

_lowercase_vowels = _a + _e + _i + _o + _u + _v
_uppercase_vowels = _lowercase_vowels.upper()
_lowercase_consonants = 'bpmfdtnlgkhjqxzcsrwy'
_uppercase_consonants = _lowercase_consonants.upper()

#: A string containing every Pinyin vowel (lowercase and uppercase).
vowels = _lowercase_vowels + _uppercase_vowels

#: A string containing every Pinyin consonant (lowercase and uppercase).
consonants = _lowercase_consonants + _uppercase_consonants

#: A string containing every lowercase Pinyin character.
lowercase = _lowercase_consonants + _lowercase_vowels

#: A string containing every uppercase Pinyin character.
uppercase = _uppercase_consonants + _uppercase_vowels

#: A string containing all Pinyin marks that have special meaning:
#: middle dot and numbers for tones, colon for easily writing \u00FC ('u:'),
#: hyphen for connecting syllables within words, and apostrophe for
#: separating a syllable beginning with a vowel from the previous syllable
#: in its word. All of these marks can be used within a valid Pinyin word.
marks = "·012345:-'"

#: A string containing valid punctuation marks that are not stops.
non_stops = """"#$%&'()*+,-/:;<=>@[\]^_`{|}~"""

#: A string containing valid stop punctuation marks.
stops = '.!?'

#: A string containing all punctuation marks.
punctuation = non_stops + stops

#: A string containing all printable Pinyin characters, marks, punctuation,
#: and whitespace.
printable = vowels + consonants + marks[:-3] + whitespace + punctuation

_a_vowels = {'a': _a, 'e': _e, 'i': _i, 'o': _o, 'u': _u, 'v': _v}
_n_vowels = {'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u', 'v': 'v\u00FC'}


def _build_syl(vowels, tone_numbers=False):
    """Builds a Pinyin syllable re pattern.

    Syllables can be preceded by a middle dot (tone mark). Syllables that end
    in a consonant are only valid if they aren't followed directly by a vowel
    with no apostrophe in between.

    The rough approach used to validate a Pinyin syllable is:
        1. Get the longest valid syllable.
        2. If it ends in a consonant make sure it's not followed directly by a
            vowel (hyphens and apostrophes don't count).
        3. If the above didn't match, repeat for the next longest valid match.

    Lookahead assertions are used to ensure that hyphens and apostrophes are
    only considered valid if used correctly. This helps to weed out non-Pinyin
    strings.

    """
    # This is the end-of-syllable-consonant lookahead assertion.
    consonant_end = '(?![{a}{e}{i}{o}{u}{v}]|u:)'.format(
        a=_a, e=_e, i=_i, o=_o, u=_u, v=_v
    )
    _vowels = vowels.copy()
    for v, s in _vowels.items():
        if len(s) > 1:
            _vowels[v] = '[{}]'.format(s)
    return (
        '(?:\u00B7|\u2027)?'
        '(?:'
        '(?:(?:[zcs]h|[gkh])u%(a)sng%(consonant_end)s)|'
        '(?:[jqx]i%(o)sng%(consonant_end)s)|'
        '(?:[nljqx]i%(a)sng%(consonant_end)s)|'
        '(?:(?:[zcs]h?|[dtnlgkhrjqxy])u%(a)sn%(consonant_end)s)|'
        '(?:(?:[zcs]h|[gkh])u%(a)si)|'
        '(?:(?:[zc]h?|[rdtnlgkhsy])%(o)sng%(consonant_end)s)|'
        '(?:(?:[zcs]h?|[rbpmfdtnlgkhw])?%(e)sng%(consonant_end)s)|'
        '(?:(?:[zcs]h?|[rbpmfdtnlgkhwy])?%(a)sng%(consonant_end)s)|'
        '(?:[bpmdtnljqxy]%(i)sng%(consonant_end)s)|'
        '(?:[bpmdtnljqx]i%(a)sn%(consonant_end)s)|'
        '(?:[bpmdtnljqx]i%(a)so)|'
        '(?:[nl](?:v|u:|\u00FC)%(e)s)|'
        '(?:[nl](?:%(v)s|u:))|'
        '(?:[jqxy]u%(e)s)|'
        '(?:[bpmnljqxy]%(i)sn%(consonant_end)s)|'
        '(?:[mdnljqx]i%(u)s)|'
        '(?:[bpmdtnljqx]i%(e)s)|'
        '(?:[dljqx]i%(a)s)|'
        '(?:(?:[zcs]h?|[rdtnlgkhxqjy])%(u)sn%(consonant_end)s)|'
        '(?:(?:[zcs]h?|[rdtgkh])u%(i)s)|'
        '(?:(?:[zcs]h?|[rdtnlgkh])u%(o)s)|'
        '(?:(?:[zcs]h|[rgkh])u%(a)s)|'
        '(?:(?:[zcs]h?|[rbpmfdngkhw])?%(e)sn%(consonant_end)s)|'
        '(?:(?:[zcs]h?|[rbpmfdtnlgkhwy])?%(a)sn%(consonant_end)s)|'
        '(?:(?:[zcs]h?|[rpmfdtnlgkhy])?%(o)su)|'
        '(?:(?:[zcs]h?|[rbpmdtnlgkhy])?%(a)so)|'
        '(?:(?:[zs]h|[bpmfdtnlgkhwz])?%(e)si)|'
        '(?:(?:[zcs]h?|[bpmdtnlgkhw])?%(a)si)|'
        '(?:(?:[zcs]h?|[rjqxybpmdtnl])%(i)s)|'
        '(?:(?:[zcs]h?|[rwbpmfdtnlgkhjqxwy])%(u)s)|'
        '(?:%(e)s(?:r%(consonant_end)s)?)|'
        '(?:(?:[zcs]h?|[rmdtnlgkhy])%(e)s)|'
        '(?:[bpmfwyl]?%(o)s)|'
        '(?:(?:[zcs]h|[bpmfdtnlgkhzcswy])?%(a)s)|'
        '(?:r%(consonant_end)s)'
        ')' + ('[0-5]?' if tone_numbers else '')
    ) % {
        'consonant_end': consonant_end, 'a': _vowels['a'], 'e': _vowels['e'],
        'i': _vowels['i'], 'o': _vowels['o'], 'u': _vowels['u'],
        'v': _vowels['v']
    }


def _build_word(syl, vowels):
    """Builds a Pinyin word re pattern from a Pinyin syllable re pattern.

    A word is defined as a series of consecutive valid Pinyin syllables
    with optional hyphens and apostrophes interspersed. Hyphens must be
    followed immediately by another valid Pinyin syllable. Apostrophes must be
    followed by another valid Pinyin syllable that starts with an 'a', 'e', or
    'o'.

    """
    return "(?:{syl}(?:-(?={syl})|'(?=[{a}{e}{o}])(?={syl}))?)+".format(
        syl=syl, a=vowels['a'], e=vowels['e'], o=vowels['o'])


def _build_sentence(word):
    """Builds a Pinyin sentence re pattern from a Pinyin word re pattern.

    A sentence is defined as a series of valid Pinyin words, punctuation
    (non-stops), and spaces followed by a single stop and zero or more
    container-closing punctuation marks (e.g. apostrophe and brackets).

    """
    return (
        "(?:{word}|[{non_stops}]|(?<![{stops} ]) )+"
        "[{stops}]['\"\]\}}\)]*"
    ).format(word=word, non_stops=non_stops.replace('-', '\-'),
             stops=stops)


#: A regular expression pattern for a valid accented Pinyin syllable.
a_syl = acc_syl = accented_syllable = _build_syl(_a_vowels, tone_numbers=False)

#: A regular expression pattern for a valid numbered Pinyin syllable.
n_syl = num_syl = numbered_syllable = _build_syl(_n_vowels, tone_numbers=True)

#: A regular expression pattern for a valid Pinyin syllable.
syl = syllable = _build_syl(_a_vowels, tone_numbers=True)


#: A regular expression pattern for a valid accented Pinyin word.
a_word = acc_word = accented_word = _build_word(a_syl, _a_vowels)

#: A regular expression pattern for a valid numbered Pinyin word.
n_word = num_word = numbered_word = _build_word(n_syl, _n_vowels)

#: A regular expression pattern for a valid Pinyin word.
word = _build_word(syl, _a_vowels)


#: A regular expression pattern for a valid accented Pinyin sentence.
a_sent = acc_sent = accented_sentence = _build_sentence(a_word)

#: A regular expression pattern for a valid numbered Pinyin sentence.
n_sent = num_sent = numbered_sentence = _build_sentence(n_word)

#: A regular expression pattern for a valid Pinyin sentence.
sent = sentence = _build_sentence(word)
