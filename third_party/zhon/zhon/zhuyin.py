
"""Constants for working with Zhuyin (Bopomofo)."""

#: A string containing all Zhuyin characters.
characters = (
    'ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙ'
    'ㄚㄛㄝㄜㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩㄭ'
)

#: A string containing all Zhuyin tone marks.
marks = (
    '\u02C7'  # Caron
    '\u02CA'  # Modifier letter accute accent
    '\u02CB'  # Modifier letter grave accent
    '\u02D9'  # Dot above
)

#: A regular expression pattern for a Zhuyin syllable.
syl = syllable = (
    '(?:'
    '[ㄇㄉㄊㄋㄌㄍㄎㄏㄓㄔㄕㄖㄗㄘㄙ]?ㄜ|'
    '[ㄅㄆㄇㄉㄊㄋㄌㄍㄎㄏㄓㄔㄕㄗㄘㄙㄧ]?ㄞ|'
    '[ㄅㄆㄇㄈㄉㄋㄌㄍㄏㄓㄕㄗ]?ㄟ|'
    '[ㄅㄆㄇㄈㄋㄍㄎㄏㄓㄔㄕㄖㄗㄘㄙ]?ㄣ|'
    '[ㄉㄌㄐㄑㄒ]?ㄧㄚ|'
    '[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄓㄔㄕㄗㄘㄙ]?ㄚ|'
    '[ㄅㄆㄇㄉㄊㄋㄌㄍㄎㄏㄓㄔㄕㄖㄗㄘㄙ]?ㄠ|'
    '[ㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄓㄔㄕㄖㄗㄘㄙ]?ㄡ|'
    '[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄓㄔㄕㄖㄗㄘㄙ]?ㄢ|'
    '[ㄇㄉㄋㄌㄐㄑㄒ]?ㄧㄡ|'
    '[ㄅㄆㄇㄋㄌㄐㄑㄒ]?ㄧㄣ|'
    '[ㄐㄑㄒ]?ㄩ[ㄢㄥ]|'
    '[ㄌㄐㄑㄒ]?ㄩㄣ|'
    '[ㄋㄌㄐㄑㄒ]?(?:ㄩㄝ?|ㄧㄤ)|'
    '[ㄅㄆㄇㄈㄌㄧ]?ㄛ|'
    '[ㄅㄆㄇㄉㄊㄋㄌㄐㄑㄒ]?ㄧ[ㄝㄠㄢㄥ]?|'
    '[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄓㄔㄕㄖㄗㄘㄙ]?[ㄤㄥ]|'
    '[ㄍㄎㄏㄓㄔㄕ]?ㄨ[ㄚㄞㄤ]|'
    '[ㄉㄊㄋㄌㄍㄎㄏㄓㄔㄕㄖㄗㄘㄙ]?ㄨㄛ|'
    '[ㄉㄊㄍㄎㄏㄓㄔㄕㄖㄗㄘㄙ]?ㄨㄟ|'
    '[ㄉㄊㄋㄌㄍㄎㄏㄓㄔㄕㄖㄗㄘㄙ]?ㄨㄢ|'
    '[ㄉㄊㄌㄍㄎㄏㄓㄔㄕㄖㄗㄘㄙ]?ㄨㄣ|'
    '[ㄉㄊㄋㄌㄍㄎㄏㄓㄔㄖㄗㄘㄙ]?ㄨㄥ|'
    '[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄓㄔㄕㄖㄗㄘㄙ]?ㄨ|'
    '[ㄓㄔㄕㄖㄗㄘㄙㄝㄦㄧ]'
    ')[{marks}]?'
).format(marks=marks)
