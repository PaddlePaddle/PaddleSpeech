# -*- coding: utf-8 -*-
"""Test error rate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from utils import error_rate


class TestParse(unittest.TestCase):
    def test_wer_1(self):
        ref = 'i UM the PHONE IS i LEFT THE portable PHONE UPSTAIRS last night'
        hyp = 'i GOT IT TO the FULLEST i LOVE TO portable FROM OF STORES last '\
                'night'
        word_error_rate = error_rate.wer(ref, hyp)
        self.assertTrue(abs(word_error_rate - 0.769230769231) < 1e-6)

    def test_wer_2(self):
        ref = 'as any in england i would say said gamewell proudly that is '\
                'in his day'
        hyp = 'as any in england i would say said came well proudly that is '\
                'in his day'
        word_error_rate = error_rate.wer(ref, hyp)
        self.assertTrue(abs(word_error_rate - 0.1333333) < 1e-6)

    def test_wer_3(self):
        ref = 'the lieutenant governor lilburn w boggs afterward governor '\
                'was a pronounced mormon hater and throughout the period of '\
                'the troubles he manifested sympathy with the persecutors'
        hyp = 'the lieutenant governor little bit how bags afterward '\
                'governor was a pronounced warman hater and throughout the '\
                'period of th troubles he manifests sympathy with the '\
                'persecutors'
        word_error_rate = error_rate.wer(ref, hyp)
        self.assertTrue(abs(word_error_rate - 0.2692307692) < 1e-6)

    def test_wer_4(self):
        ref = 'the wood flamed up splendidly under the large brewing copper '\
                'and it sighed so deeply'
        hyp = 'the wood flame do splendidly under the large brewing copper '\
                'and its side so deeply'
        word_error_rate = error_rate.wer(ref, hyp)
        self.assertTrue(abs(word_error_rate - 0.2666666667) < 1e-6)

    def test_wer_5(self):
        ref = 'all the morning they trudged up the mountain path and at noon '\
                'unc and ojo sat on a fallen tree trunk and ate the last of '\
                'the bread which the old munchkin had placed in his pocket'
        hyp = 'all the morning they trudged up the mountain path and at noon '\
                'unc in ojo sat on a fallen tree trunk and ate the last of '\
                'the bread which the old munchkin had placed in his pocket'
        word_error_rate = error_rate.wer(ref, hyp)
        self.assertTrue(abs(word_error_rate - 0.027027027) < 1e-6)

    def test_wer_6(self):
        ref = 'i UM the PHONE IS i LEFT THE portable PHONE UPSTAIRS last night'
        word_error_rate = error_rate.wer(ref, ref)
        self.assertEqual(word_error_rate, 0.0)

    def test_wer_7(self):
        ref = ' '
        hyp = 'Hypothesis sentence'
        with self.assertRaises(ValueError):
            word_error_rate = error_rate.wer(ref, hyp)

    def test_cer_1(self):
        ref = 'werewolf'
        hyp = 'weae  wolf'
        char_error_rate = error_rate.cer(ref, hyp)
        self.assertTrue(abs(char_error_rate - 0.25) < 1e-6)

    def test_cer_2(self):
        ref = 'werewolf'
        hyp = 'weae  wolf'
        char_error_rate = error_rate.cer(ref, hyp, remove_space=True)
        self.assertTrue(abs(char_error_rate - 0.125) < 1e-6)

    def test_cer_3(self):
        ref = 'were wolf'
        hyp = 'were  wolf'
        char_error_rate = error_rate.cer(ref, hyp)
        self.assertTrue(abs(char_error_rate - 0.0) < 1e-6)

    def test_cer_4(self):
        ref = 'werewolf'
        char_error_rate = error_rate.cer(ref, ref)
        self.assertEqual(char_error_rate, 0.0)

    def test_cer_5(self):
        ref = u'我是中国人'
        hyp = u'我是 美洲人'
        char_error_rate = error_rate.cer(ref, hyp)
        self.assertTrue(abs(char_error_rate - 0.6) < 1e-6)

    def test_cer_6(self):
        ref = u'我 是 中 国 人'
        hyp = u'我 是 美 洲 人'
        char_error_rate = error_rate.cer(ref, hyp, remove_space=True)
        self.assertTrue(abs(char_error_rate - 0.4) < 1e-6)

    def test_cer_7(self):
        ref = u'我是中国人'
        char_error_rate = error_rate.cer(ref, ref)
        self.assertFalse(char_error_rate, 0.0)

    def test_cer_8(self):
        ref = ''
        hyp = 'Hypothesis'
        with self.assertRaises(ValueError):
            char_error_rate = error_rate.cer(ref, hyp)


if __name__ == '__main__':
    unittest.main()
