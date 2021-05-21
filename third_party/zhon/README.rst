====
Zhon
====

.. image:: https://badge.fury.io/py/zhon.png
    :target: http://badge.fury.io/py/zhon

.. image:: https://travis-ci.org/tsroten/zhon.png?branch=develop
        :target: https://travis-ci.org/tsroten/zhon

Zhon is a Python library that provides constants commonly used in Chinese text
processing.

* Documentation: http://zhon.rtfd.org
* GitHub: https://github.com/tsroten/zhon
* Support: https://github.com/tsroten/zhon/issues
* Free software: `MIT license <http://opensource.org/licenses/MIT>`_

About
-----

Zhon's constants can be used in Chinese text processing, for example:

* Find CJK characters in a string:

  .. code:: python

    >>> re.findall('[{}]'.format(zhon.hanzi.characters), 'I broke a plate: 我打破了一个盘子.')
    ['我', '打', '破', '了', '一', '个', '盘', '子']

* Validate Pinyin syllables, words, or sentences:

  .. code:: python

    >>> re.findall(zhon.pinyin.syllable, 'Yuànzi lǐ tíngzhe yí liàng chē.', re.I)
    ['Yuàn', 'zi', 'lǐ', 'tíng', 'zhe', 'yí', 'liàng', 'chē']

    >>> re.findall(zhon.pinyin.word, 'Yuànzi lǐ tíngzhe yí liàng chē.', re.I)
    ['Yuànzi', 'lǐ', 'tíngzhe', 'yí', 'liàng', 'chē']

    >>> re.findall(zhon.pinyin.sentence, 'Yuànzi lǐ tíngzhe yí liàng chē.', re.I)
    ['Yuànzi lǐ tíngzhe yí liàng chē.']

Features
--------

+ Includes commonly-used constants:
    - CJK characters and radicals
    - Chinese punctuation marks
    - Chinese sentence regular expression pattern
    - Pinyin vowels, consonants, lowercase, uppercase, and punctuation
    - Pinyin syllable, word, and sentence regular expression patterns
    - Zhuyin characters and marks
    - Zhuyin syllable regular expression pattern
    - CC-CEDICT characters
+ Runs on Python 2.7 and 3

Getting Started
---------------

* `Install Zhon <http://zhon.readthedocs.org/en/latest/#installation>`_
* Read `Zhon's introduction <http://zhon.readthedocs.org/en/latest/#using-zhon>`_
* Learn from the `API documentation <http://zhon.readthedocs.org/en/latest/#zhon-hanzi>`_
