# Copyright 2014 Bernard Yue
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

__doc__ = """
Hanzi Converter 繁簡轉換器 | 繁简转换器
This module provides functions converting chinese text between simplified and
traditional characters.  It returns unicode represnetation of the text.
Class HanziConv is the main entry point of the module, you can import the
class by doing:
    >>> from hanziconv import HanziConv
"""

import os
from zhon import cedict

class HanziConv():
    """This class supports hanzi (漢字) convention between simplified and
    traditional format"""
    __traditional_charmap = cedict.traditional
    __simplified_charmap = cedict.simplified

    @classmethod
    def __convert(cls, text, toTraditional=True):
        """Convert `text` to Traditional characters if `toTraditional` is
        True, else convert to simplified characters
        :param text:           data to convert
        :param toTraditional:  True -- convert to traditional text
                               False -- covert to simplified text
        :returns:              converted 'text`
        """
        if isinstance(text, bytes):
            text = text.decode('utf-8')

        fromMap = cls.__simplified_charmap
        toMap = cls.__traditional_charmap
        if not toTraditional:
            fromMap = cls.__traditional_charmap
            toMap = cls.__simplified_charmap

        final = []
        for c in text:
            index = fromMap.find(c)
            if index != -1:
                final.append(toMap[index])
            else:
                final.append(c)
        return ''.join(final)

    @classmethod
    def toSimplified(cls, text):
        """Convert `text` to simplified character string.  Assuming text is
        traditional character string
        :param text:  text to convert
        :returns:     converted UTF-8 characters
        >>> from hanziconv import HanziConv
        >>> print(HanziConv.toSimplified('繁簡轉換器'))
        繁简转换器
        """
        return cls.__convert(text, toTraditional=False)

    @classmethod
    def toTraditional(cls, text):
        """Convert `text` to traditional character string.  Assuming text is
        simplified character string
        :param text:  text to convert
        :returns:     converted UTF-8 characters
        >>> from hanziconv import HanziConv
        >>> print(HanziConv.toTraditional('繁简转换器'))
        繁簡轉換器
        """
        return cls.__convert(text, toTraditional=True)

    @classmethod
    def same(cls, text1, text2):
        """Return True if text1 and text2 meant literally the same, False
        otherwise
        :param text1: string to compare to ``text2``
        :param text2: string to compare to ``text1``
        :returns:     **True**  -- ``text1`` and ``text2`` are the same in meaning,
                      **False** -- otherwise
        >>> from hanziconv import HanziConv
        >>> print(HanziConv.same('繁简转换器', '繁簡轉換器'))
        True
        """
        t1 = cls.toSimplified(text1)
        t2 = cls.toSimplified(text2)
        return t1 == t2