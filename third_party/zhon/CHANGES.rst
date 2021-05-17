Changes
=======

v0.1.0 (2013-05-05)
-------------------

* Initial release

v0.1.1 (2013-05-05)
-------------------

* Adds zhon.cedict package to setup.py

v0.2.0 (2013-05-07)
-------------------

* Allows for mapping between simplified and traditional.
* Adds logging to build_string().
* Adds constants for numbered Pinyin and accented Pinyin.

v0.2.1 (2013-05-07)
-------------------

* Fixes typo in README.rst.

v.1.0.0 (2014-01-25)
--------------------

* Complete rewrite that refactors code, renames constants, and improves Pinyin
  support.

v.1.1.0 (2014-01-28)
--------------------

* Adds ``zhon.pinyin.punctuation`` constant.
* Adds ``zhon.pinyin.accented_syllable``, ``zhon.pinyin.accented_word``, and
  ``zhon.pinyin.accented_sentence`` constants.
* Adds ``zhon.pinyin.numbered_syllable``, ``zhon.pinyin.numbered_word``, and
  ``zhon.pinyin.numbered_sentence`` constants.
* Fixes some README.rst typos.
* Clarifies information regarding Traditional and Simplified character
  constants in README.rst.
* Adds constant short names to README.rst.

v.1.1.1 (2014-01-29)
--------------------

* Adds documentation.
* Adds ``zhon.cedict.all`` constant.
* Removes duplicate code ranges from ``zhon.hanzi.characters``.
* Makes ``zhon.hanzi.non_stops`` a string containing all non-stops instead of
  a string containing code ranges.
* Removes duplicate letters in ``zhon.pinyin.consonants``.
* Refactors Pinyin vowels/consonant code.
* Removes the Latin alpha from ``zhon.pinyin.vowels``. Fixes #16.
* Adds ``cjk_ideographs`` alias for ``zhon.hanzi.characters``.
* Fixes various typos.
* Removes numbers from Pinyin word constants. Fixes #15.
* Adds lowercase and uppercase constants to ``zhon.pinyin``.
* Fixes a bug with ``zhon.pinyin.sentence``.
* Adds ``sent`` alias for ``zhon.pinyin.sentence``.

v.1.1.2 (2014-01-31)
--------------------

* Fixes bug with ``zhon.cedict.all``.

v.1.1.3 (2014-02-12)
--------------------

* Adds Ideographic number zero to ``zhon.hanzi.characters``. Fixes #17.
* Fixes r-suffix bug. Fixes #18.

v.1.1.4 (2015-01-25)
--------------------

* Removes duplicate module declarations in documentation.
* Moves tests inside zhon package.
* Adds travis config file.
* Adds Python 3.4 tests to travis and tox.
* Fixes flake8 warnings.
* Adds distutil fallback import statment to setup.py.
* Adds missing hanzi punctuation. Fixes #19.

v.1.1.5 (2016-05-23)
--------------------

* Add missing Zhuyin characters. Fixes #23.
