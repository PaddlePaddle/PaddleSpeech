# -*- coding: utf-8 -*-
"""Test error rate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import error_rate


class TestParse(unittest.TestCase):
    def test_wer_1(self):
        ref = 'i UM the PHONE IS i LEFT THE portable PHONE UPSTAIRS last night'
        hyp = 'i GOT IT TO the FULLEST i LOVE TO portable FROM OF STORES last night'
        word_error_rate = error_rate.wer(ref, hyp)
        self.assertTrue(abs(word_error_rate - 0.769230769231) < 1e-6)

    def test_wer_2(self):
        ref = 'i UM the PHONE IS i LEFT THE portable PHONE UPSTAIRS last night'
        word_error_rate = error_rate.wer(ref, ref)
        self.assertEqual(word_error_rate, 0.0)

    def test_wer_3(self):
        ref = ' '
        hyp = 'Hypothesis sentence'
        try:
            word_error_rate = error_rate.wer(ref, hyp)
        except Exception as e:
            self.assertTrue(isinstance(e, ValueError))

    def test_cer_1(self):
        ref = 'werewolf'
        hyp = 'weae  wolf'
        char_error_rate = error_rate.cer(ref, hyp)
        self.assertTrue(abs(char_error_rate - 0.25) < 1e-6)

    def test_cer_2(self):
        ref = 'werewolf'
        char_error_rate = error_rate.cer(ref, ref)
        self.assertEqual(char_error_rate, 0.0)

    def test_cer_3(self):
        ref = u'我是中国人'
        hyp = u'我是 美洲人'
        char_error_rate = error_rate.cer(ref, hyp)
        self.assertTrue(abs(char_error_rate - 0.6) < 1e-6)

    def test_cer_4(self):
        ref = u'我是中国人'
        char_error_rate = error_rate.cer(ref, ref)
        self.assertFalse(char_error_rate, 0.0)

    def test_cer_5(self):
        ref = ''
        hyp = 'Hypothesis'
        try:
            char_error_rate = error_rate.cer(ref, hyp)
        except Exception as e:
            self.assertTrue(isinstance(e, ValueError))


if __name__ == '__main__':
    unittest.main()
