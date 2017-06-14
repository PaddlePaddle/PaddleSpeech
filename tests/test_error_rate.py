# -*- coding: utf-8 -*-
import unittest
import sys
sys.path.append('..')
import error_rate


class TestParse(unittest.TestCase):
    def test_wer(self):
        ref = 'i UM the PHONE IS i LEFT THE portable PHONE UPSTAIRS last night'
        hyp = 'i GOT IT TO the FULLEST i LOVE TO portable FROM OF STORES last night'
        word_error_rate = error_rate.wer(ref, hyp)
        self.assertTrue(abs(word_error_rate - 0.769230769231) < 1e-6)

    def test_cer_en(self):
        ref = 'werewolf'
        hyp = 'weae  wolf'
        char_error_rate = error_rate.cer(ref, hyp)
        self.assertTrue(abs(char_error_rate - 0.25) < 1e-6)

    def test_cer_zh(self):
        ref = u'我是中国人'
        hyp = u'我是 美洲人'
        char_error_rate = error_rate.cer(ref, hyp)
        self.assertTrue(abs(char_error_rate - 0.6) < 1e-6)


if __name__ == '__main__':
    unittest.main()
