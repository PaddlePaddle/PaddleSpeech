
"""Tests for the zhon.hanzi module."""

import re
import unittest

from zhon import hanzi


class TestCharacters(unittest.TestCase):

    def test_all_chinese(self):
        c_re = re.compile('[^{}]'.format(hanzi.characters))
        t = '你我都很她它隹廿'
        self.assertEqual(c_re.search(t), None)

    def test_chinese_and_punc(self):
        c_re = re.compile('[^{}]'.format(hanzi.characters))
        t = '你我都很她它隹廿。，！'
        self.assertNotEqual(c_re.search(t), None)


class TestRadicals(unittest.TestCase):

    def test_only_radicals(self):
        r_re = re.compile('[^{}]'.format(hanzi.radicals))
        t = '\u2F00\u2F31\u2FBA\u2E98\u2EF3\u2ECF'
        self.assertEqual(r_re.search(t), None)

    def test_chinese_equivalents(self):
        r_re = re.compile('[^{}]'.format(hanzi.radicals))
        t = '\u4E00\u5E7F\u516B\u5165'
        self.assertNotEqual(r_re.search(t), None)


class TestPunctuation(unittest.TestCase):

    def test_split_on_punctuation(self):
        p_re = re.compile('[{}]'.format(hanzi.punctuation))
        t = '你好你好好好哈哈，米饭很好吃；哈哈！'
        self.assertEqual(len(p_re.split(t)), 4)

    def test_issue_19(self):
        self.assertTrue('《' in hanzi.punctuation)
        self.assertTrue('·' in hanzi.punctuation)
        self.assertTrue('〈' in hanzi.punctuation)
        self.assertTrue('〉' in hanzi.punctuation)
        self.assertTrue('﹑' in hanzi.punctuation)
        self.assertTrue('﹔' in hanzi.punctuation)
