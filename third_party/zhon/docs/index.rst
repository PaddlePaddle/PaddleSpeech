.. Zhon documentation master file, created by
   sphinx-quickstart on Tue Jan 28 22:18:02 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Zhon
====

Introduction
------------

Zhon is a Python library that provides constants commonly used in Chinese text
processing:

* CJK characters and radicals
* Chinese punctuation marks
* Chinese sentence regular expression pattern
* Pinyin vowels, consonants, lowercase, uppercase, and punctuation
* Pinyin syllable, word, and sentence regular expression patterns
* Zhuyin characters and marks
* Zhuyin syllable regular expression pattern
* CC-CEDICT characters

Installation
------------

Zhon supports Python 2.7 and 3. Install using pip:

.. code:: bash

    $ pip install zhon

If you want to download the latest source code, check out `Zhon's GitHub
repository <https://github.com/tsroten/zhon>`_.

Be sure to `report any bugs <https://github.com/tsroten/zhon/issues>`_ you find.
Thanks!

.. module:: zhon

Using Zhon
----------

Zhon contains four modules that export helpful Chinese constants:

* :py:mod:`zhon.hanzi`
* :py:mod:`zhon.pinyin`
* :py:mod:`zhon.zhuyin`
* :py:mod:`zhon.cedict`

Zhon's constants are formatted in one of three ways:

* Characters listed individually. These can be used with membership tests
  or used to build regular expression patterns. For example, ``'aeiou'``.
* Character code ranges. These are used to build regular expression patterns.
  For example, ``'u\0041-\u005A\u0061-\u007A'``.
* Regular expression pattern. These are regular expression patterns
  that can be used with the regular expression library directly. For
  example, ``'[u\0020-\u007E]+'``.

Using the constants listed below is simple. For constants that list the
characters individually, you can perform membership tests or use them in
regular expressions:

.. code:: python

    >>> '车' in zhon.cedict.traditional
    False

    >>> # This regular expression finds all characters that aren't considered
    ... # traditional according to CC-CEDICT
    ... re.findall('[^{}]'.format(zhon.cedict.traditional), '我买了一辆车')
    ['买', '辆', '车']

For constants that contain character code ranges, you'll want to build a
regular expression:

.. code:: python

    >>> re.findall('[{}]'.format(zhon.hanzi.punctuation), '我买了一辆车。')
    ['。']

For constants that are regular expression patterns, you can use them directly
with the regular expression library, without formatting them:

.. code:: python

    >>> re.findall(zhon.hanzi.sentence, '我买了一辆车。妈妈做的菜，很好吃！')
    ['我买了一辆车。', '妈妈做的菜，很好吃！']

.. module:: zhon.hanzi

``zhon.hanzi``
~~~~~~~~~~~~~~

These constants can be used when working directly with Chinese characters.

These constants can be used in a variety of ways, but they can't directly
distinguish between Chinese, Japanese, and Korean characters/words.
Chapter 12 of The Unicode Standard
(`PDF <http://www.unicode.org/versions/Unicode6.2.0/ch12.pdf>`_)
has some useful information about this:

    There is some concern that unifying the Han characters may lead to confusion because they are sometimes used differently by the various East Asian languages. Computationally, Han character unification presents no more difficulty than employing a single Latin character set that is used to write languages as different as English and French. Programmers do not expect the characters "c", "h", "a", and "t" alone to tell us whether chat is a French word for cat or an English word meaning “informal talk.” Likewise, we depend on context to identify the American hood (of a car) with the British bonnet. Few computer users are confused by the fact that ASCII can also be used to represent such words as the Welsh word ynghyd, which are strange looking to English eyes. Although it would be convenient to identify words by language for programs such as spell-checkers, it is neither practical nor productive to encode a separate Latin character set for every language that uses it.

.. py:data:: characters
    cjk_ideographs

    Character codes and code ranges for pertinent CJK ideograph Unicode characters. This includes:

    * `CJK Unified Ideographs <http://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)>`_
    * `CJK Unified Ideographs Extension A <http://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_A>`_
    * `CJK Unified Ideographs Extension B <http://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_B>`_
    * `CJK Unified Ideographs Extension C <http://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_C>`_
    * `CJK Unified Ideographs Extension D <http://en.wikipedia.org/wiki/CJK_Unified_Ideographs_Extension_D>`_
    * `CJK Compatibility Ideographs <http://en.wikipedia.org/wiki/CJK_Compatibility_Ideographs>`_
    * `CJK Compatibility Ideographs Supplement <http://en.wikipedia.org/wiki/CJK_Compatibility_Ideographs_Supplement>`_
    * Ideographic number zero

    Some of the characters in this constant will not be Chinese characters,
    but this is a convienient way to approach the issue. If you'd rather have
    an enormous string of Chinese characters from a Chinese dictionary, check
    out :py:data:`zhon.cedict`.

.. py:data:: radicals

    Character code ranges for the `Kangxi Radicals <http://en.wikipedia.org/wiki/Kangxi_radical#Unicode>`_
    and `CJK Radicals Supplement <http://en.wikipedia.org/wiki/CJK_Radicals_Supplement>`_
    Unicode blocks.

.. py:data:: punctuation

    This is the concatenation of :py:data:`zhon.hanzi.non_stops` and
    :py:data:`zhon.hanzi.stops`.

.. py:data:: non_stops

    The string ``'＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏'``.
    This contains Chinese punctuation marks, excluding punctuation marks that
    function as stops.

.. py:data:: stops

    The string ``'！？｡。'``. These punctuation marks function as stops.

.. py:data:: sent
    sentence

    A regular expression pattern for a Chinese sentence. A sentence is defined
    as a series of CJK characters (as defined by
    :py:data:`zhon.hanzi.characters`) and non-stop punctuation marks followed
    by a stop and zero or more container-closing punctuation marks (e.g.
    apostrophe and brackets).

    .. code:: python

        >>> re.findall(zhon.hanzi.sentence, '我买了一辆车。')
        ['我买了一辆车。']

.. module:: zhon.pinyin

``zhon.pinyin``
~~~~~~~~~~~~~~~

These constants can be used when working with Pinyin.

.. py:data:: vowels

    The string ``'aeiouvüāēīōūǖáéíóúǘǎěǐǒǔǚàèìòùǜAEIOUVÜĀĒĪŌŪǕÁÉÍÓÚǗǍĚǏǑǓǙÀÈÌÒÙǛ'``. This contains every Pinyin vowel (lowercase and uppercase).

.. py:data:: consonants

    The string ``'bpmfdtnlgkhjqxzcsrwyBPMFDTNLGKHJQXZCSRWY'``. This
    contains every Pinyin consonant (lowercase and uppercase).

.. py:data:: lowercase

    The string ``'bpmfdtnlgkhjqxzcsrwyaeiouvüāēīōūǖáéíóúǘǎěǐǒǔǚàèìòùǜ'``. This contains every lowercase Pinyin vowel and consonant.

.. py:data:: uppercase

    The string ``'BPMFDTNLGKHJQXZCSRWYAEIOUVÜĀĒĪŌŪǕÁÉÍÓÚǗǍĚǏǑǓǙÀÈÌÒÙǛ'``.
    This contains every uppercase vowel and consonant.

.. py:data:: marks

    The string ``"·012345:-'"``. This contains all Pinyin marks that have
    special meaning: a middle dot and numbers for indicating tone, a colon for
    easily writing ü ('u:'), a hyphen for connecting syllables within words,
    and an apostrophe for separating a syllable beginning with a vowel from
    the previous syllable in its word. All of these marks can be used within a
    valid Pinyin word.

.. py:data:: punctuation

    The concatenation of :py:data:`zhon.pinyin.non_stops` and
    :py:data:`zhon.pinyin.stops`.

.. py:data:: non_stops

    The string ``'"#$%&\'()*+,-/:;<=>@[\]^_`{|}~"'``. This contains every
    ASCII punctuation mark that doesn't function as a stop.

.. py:data:: stops

    The string ``'.!?'``. This contains every ASCII punctuation mark that
    functions as a stop.

.. py:data:: printable

    The concatenation of :py:data:`zhon.pinyin.vowels`,
    :py:data:`zhon.pinyin.consonants`, :py:data:`zhon.pinyin.marks`,
    :py:data:`zhon.pinyin.punctuation`, and :py:data:`string.whitespace`. This
    is essentially a Pinyin whitelist for complete Pinyin sentences -- it's
    every possible valid character a Pinyin string can use assuming all
    non-Chinese words that might be included (like proper nouns) use ASCII.

Validating and splitting Pinyin isn't as simple as checking that only
valid characters exist or matching maximum-length valid syllables.
The regular expression library's lookahead features are used in this
module's regular expression patterns to ensure that only valid Pinyin
syllables are matched. The approach used to segment a string into valid
Pinyin syllables is roughly:

1. Match the longest possible valid syllable.
2. If that match is followed directly by a vowel, drop that match and try
   again with the next longest possible valid syllable.

Additionally, lookahead assertions are used to ensure that hyphens and
apostrophes are only accepted when they are used correctly. This helps to
weed out non-Pinyin strings.

.. py:data:: syl
    syllable

    A regular expression pattern for a valid Pinyin syllable (accented or
    numbered). Compile with :py:data:`re.IGNORECASE` (:py:data:`re.I`) to
    accept uppercase letters as well.

    .. code:: python

        >>> re.findall(zhon.pinyin.syllable, 'Shū zài zhuōzi shàngmian. Shu1 zai4 zhuo1zi5 shang4mian5.', re.IGNORECASE)
        ['Shū', 'zài', 'zhuō', 'zi', 'shàng', 'mian', 'Shu1', 'zai4', 'zhuo1', 'zi5', 'shang4', 'mian5']

.. py:data:: a_syl
    acc_syl
    accented_syllable

    A regular expression for a valid accented Pinyin syllable. Compile with
    :py:data:`re.IGNORECASE` (:py:data:`re.I`) to accept uppercase letters as
    well.

    .. code:: python

        >>> re.findall(zhon.pinyin.acc_syl, 'Shū zài zhuōzi shàngmian.', re.IGNORECASE)
        ['Shū', 'zài', 'zhuō', 'zi', 'shàng', 'mian']


.. py:data:: n_syl
    num_syl
    numbered_syllable

    A regular expression for a valid numbered Pinyin syllable. Compile with
    :py:data:`re.IGNORECASE` (:py:data:`re.I`) to accept uppercase letters as
    well.

    .. code:: python

        >>> re.findall(zhon.pinyin.num_syl, 'Shu1 zai4 zhuo1zi5 shang4mian5.', re.IGNORECASE)
        ['Shu1', 'zai4', 'zhuo1', 'zi5', 'shang4', 'mian5']

.. py:data:: word

    A regular expression pattern for a valid Pinyin word (accented or
    numbered). Compile with :py:data:`re.IGNORECASE` (:py:data:`re.I`) to
    accept uppercase letters as well.

    .. code:: python

        >>> re.findall(zhon.pinyin.word, 'Shū zài zhuōzi shàngmian. Shu1 zai4 zhuo1zi5 shang4mian5.', re.IGNORECASE)
        ['Shū', 'zài', 'zhuōzi', 'shàngmian', 'Shu1', 'zai4', 'zhuo1zi5', 'shang4mian5'

.. py:data:: a_word
    acc_word
    accented_word

    A regular expression for a valid accented Pinyin word. Compile with
    :py:data:`re.IGNORECASE` (:py:data:`re.I`) to accept uppercase letters as
    well.

    .. code:: python

        >>> re.findall(zhon.pinyin.acc_word, 'Shū zài zhuōzi shàngmian.', re.IGNORECASE)
        ['Shū', 'zài', 'zhuōzi', 'shàngmian']


.. py:data:: n_word
    num_word
    numbered_word

    A regular expression for a valid numbered Pinyin word. Compile with
    :py:data:`re.IGNORECASE` (:py:data:`re.I`) to accept uppercase letters as
    well.

    .. code:: python

        >>> re.findall(zhon.pinyin.num_word, 'Shu1 zai4 zhuo1zi5 shang4mian5.', re.IGNORECASE)
        ['Shu1', 'zai4', 'zhuo1zi5', 'shang4mian5']

.. py:data:: sent
    sentence

    A regular expression pattern for a valid Pinyin sentence (accented or
    numbered). Compile with :py:data:`re.IGNORECASE` (:py:data:`re.I`) to
    accept uppercase letters as well.

    .. code:: python

        >>> re.findall(zhon.pinyin.sentence, 'Shū zài zhuōzi shàngmian. Shu1 zai4 zhuo1zi5 shang4mian5.', re.IGNORECASE)
        ['Shū zài zhuōzi shàngmian.', 'Shu1 zai4 zhuo1zi5 shang4mian5.']

.. py:data:: a_sent
    acc_sent
    accented_sentence

    A regular expression for a valid accented Pinyin sentence. Compile with
    :py:data:`re.IGNORECASE` (:py:data:`re.I`) to accept uppercase letters as
    well.


    .. code:: python

        >>> re.findall(zhon.pinyin.acc_sent, 'Shū zài zhuōzi shàngmian.', re.IGNORECASE)
        ['Shū zài zhuōzi shàngmian.']


.. py:data:: n_sent
    num_sent
    numbered_sentence

    A regular expression for a valid numbered Pinyin sentence. Compile with
    :py:data:`re.IGNORECASE` (:py:data:`re.I`) to accept uppercase letters as
    well.


    .. code:: python

        >>> re.findall(zhon.pinyin.num_sent, 'Shu1 zai4 zhuo1zi5 shang4mian5.', re.IGNORECASE)
        ['Shu1 zai4 zhuo1zi5 shang4mian5.']

.. module:: zhon.zhuyin

``zhon.zhuyin``
~~~~~~~~~~~~~~~

These constants can be used when working with Zhuyin (Bopomofo).

.. py:data:: characters

    The string ``'ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄝㄜㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧ'``.
    This contains all Zhuyin characters as defined by the `Bomopofo Unicode
    block <http://en.wikipedia.org/wiki/Bopomofo_(Unicode_block)>`_. It does
    not include the
    `Bomopofo Extended block <http://en.wikipedia.org/wiki/Bopomofo_Extended_(Unicode_block)>`_
    that defines characters used in non-standard dialects or minority
    languages.

.. py:data:: marks

    The string ``'ˇˊˋ˙'``. This contains the Zhuyin tone marks.

.. py:data:: syl
    syllable

    A regular expression pattern for a valid Zhuyin syllable.

    .. code:: python

        >>> re.findall(zhon.zhuyin.syllable, 'ㄓㄨˋ ㄧㄣ ㄈㄨˊ ㄏㄠˋ')
        ['ㄓㄨˋ', 'ㄧㄣ', 'ㄈㄨˊ', 'ㄏㄠˋ']

.. module:: zhon.cedict

``zhon.cedict``
~~~~~~~~~~~~~~~

These constants are built from the `CC-CEDICT dictionary
<http://cc-cedict.org/wiki/>`_.
They aren't guaranteed to contain every possible Chinese character. They only
provide characters that exist in the CC-CEDICT dictionary.

.. py:data:: all

    A string containing all Chinese characters found in `CC-CEDICT
    <http://cc-cedict.org/wiki/>`_.

.. py:data:: trad
    traditional

    A string containing characters considered by `CC-CEDICT
    <http://cc-cedict.org/wiki/>`_ to be Traditional Chinese characters.
    Some of these characters are also present in
    :py:data:`zhon.cedict.simplified` because many characters were left
    untouched by the simplification process.

.. py:data:: simp
    simplified

    A string containing characters considered by `CC-CEDICT
    <http://cc-cedict.org/wiki/>`_ to be Simplified Chinese characters.
    Some of these characters are also present in
    :py:data:`zhon.cedict.traditional` because many characters were left
    untouched by the simplification process.
