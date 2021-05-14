
"""Tests for the zhon.cedict module."""

import re
import unittest
from zhon import cedict


class TestSimplified(unittest.TestCase):

    simplified_text = '有人丢失了一把斧子怎么找也没有找到'

    def test_re_complement_search(self):
        re_complement = re.compile('[^{}]'.format(cedict.simplified))
        self.assertEqual(re_complement.search(self.simplified_text), None)


class TestTraditional(unittest.TestCase):

    simplified_text = '有人丢失了一把斧子怎么找也没有找到'

    def test_re_complement_search(self):
        re_complement = re.compile('[^{}]'.format(cedict.traditional))
        self.assertNotEqual(re_complement.search(self.simplified_text), None)


class TestAll(unittest.TestCase):

    all_text = '车車'

    def test_re_complement_search(self):
        re_complement = re.compile('[^{}]'.format(cedict.all))
        self.assertEqual(re_complement.search(self.all_text), None)
