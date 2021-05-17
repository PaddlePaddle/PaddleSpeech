
"""Tests for the zhon.pinyin module."""

import random
import re
import unittest

from zhon import pinyin


NUM_WORDS = 50   # Number of random words to test
WORD_LENGTH = 4    # Length of random words (number of syllables)
NUM_SENT = 10     # Number of random sentences to test
SENT_LENGTH = 5   # Length of random sentences (number of words)

VALID_SYLS = (  # 411 total syllables, including 'r'
    'ba', 'pa', 'ma', 'fa', 'da', 'ta', 'na', 'la', 'ga', 'ka', 'ha', 'za',
    'ca', 'sa', 'zha', 'cha', 'sha', 'a', 'bo', 'po', 'mo', 'fo', 'yo', 'lo',
    'o', 'me', 'de', 'te', 'ne', 'le', 'ge', 'ke', 'he', 'ze', 'ce', 'se',
    'zhe', 'che', 'she', 're', 'e', 'bai', 'pai', 'mai', 'dai', 'tai',
    'nai', 'lai', 'gai', 'kai', 'hai', 'zai', 'cai', 'sai', 'zhai', 'chai',
    'shai', 'ai', 'bei', 'pei', 'mei', 'fei', 'dei', 'tei', 'nei', 'lei',
    'gei', 'kei', 'hei', 'zei', 'zhei', 'shei', 'ei', 'bao', 'pao', 'mao',
    'dao', 'tao', 'nao', 'lao', 'gao', 'kao', 'hao', 'zao', 'cao', 'sao',
    'zhao', 'chao', 'shao', 'rao', 'ao', 'pou', 'mou', 'fou', 'dou', 'tou',
    'nou', 'lou', 'gou', 'kou', 'hou', 'zou', 'cou', 'sou', 'zhou', 'chou',
    'shou', 'rou', 'ou', 'ban', 'pan', 'man', 'fan', 'dan', 'tan', 'nan',
    'lan', 'gan', 'kan', 'han', 'zan', 'can', 'san', 'zhan', 'chan',
    'shan', 'ran', 'an', 'bang', 'pang', 'mang', 'fang', 'dang', 'tang',
    'nang', 'lang', 'gang', 'kang', 'hang', 'zang', 'cang', 'sang',
    'zhang', 'chang', 'shang', 'rang', 'ang', 'ben', 'pen', 'men', 'fen',
    'den', 'nen', 'gen', 'ken', 'hen', 'zen', 'cen', 'sen', 'zhen', 'chen',
    'shen', 'ren', 'en', 'beng', 'peng', 'meng', 'feng', 'deng', 'teng',
    'neng', 'leng', 'geng', 'keng', 'heng', 'zeng', 'ceng', 'seng',
    'zheng', 'cheng', 'sheng', 'reng', 'eng', 'dong', 'tong', 'nong',
    'long', 'gong', 'kong', 'hong', 'zong', 'cong', 'song', 'zhong',
    'chong', 'rong', 'bu', 'pu', 'mu', 'fu', 'du', 'tu', 'nu', 'lu',
    'gu', 'ku', 'hu', 'zu', 'cu', 'su', 'zhu', 'chu', 'shu', 'ru', 'wu',
    'gua', 'kua', 'hua', 'zhua', 'chua', 'shua', 'rua', 'wa', 'duo', 'tuo',
    'nuo', 'luo', 'guo', 'kuo', 'huo', 'zuo', 'cuo', 'suo', 'zhuo', 'chuo',
    'shuo', 'ruo', 'wo', 'guai', 'kuai', 'huai', 'zhuai', 'chuai', 'shuai',
    'wai', 'dui', 'tui', 'gui', 'kui', 'hui', 'zui', 'cui', 'sui', 'zhui',
    'chui', 'shui', 'rui', 'wei', 'duan', 'tuan', 'nuan', 'luan', 'guan',
    'kuan', 'huan', 'zuan', 'cuan', 'suan', 'zhuan', 'chuan', 'shuan',
    'ruan', 'wan', 'guang', 'kuang', 'huang', 'zhuang', 'chuang', 'shuang',
    'wang', 'dun', 'tun', 'nun', 'lun', 'gun', 'kun', 'hun', 'zun', 'cun',
    'sun', 'zhun', 'chun', 'shun', 'run', 'wen', 'weng', 'bi', 'pi', 'mi',
    'di', 'ti', 'ni', 'li', 'zi', 'ci', 'si', 'zhi', 'chi', 'shi', 'ri',
    'ji', 'qi', 'xi', 'yi', 'dia', 'lia', 'jia', 'qia', 'xia', 'ya', 'bie',
    'pie', 'mie', 'die', 'tie', 'nie', 'lie', 'jie', 'qie', 'xie', 'ye',
    'biao', 'piao', 'miao', 'diao', 'tiao', 'niao', 'liao', 'jiao', 'qiao',
    'xiao', 'yao', 'miu', 'diu', 'niu', 'liu', 'jiu', 'qiu', 'xiu', 'you',
    'bian', 'pian', 'mian', 'dian', 'tian', 'nian', 'lian', 'jian', 'qian',
    'xian', 'yan', 'niang', 'liang', 'jiang', 'qiang', 'xiang', 'yang',
    'bin', 'pin', 'min', 'nin', 'lin', 'jin', 'qin', 'xin', 'yin', 'bing',
    'ping', 'ming', 'ding', 'ting', 'ning', 'ling', 'jing', 'qing', 'xing',
    'ying', 'jiong', 'qiong', 'xiong', 'yong', 'nü', 'lü', 'ju', 'qu',
    'xu', 'yu', 'nüe', 'lüe', 'jue', 'que', 'xue', 'yue', 'juan', 'quan',
    'xuan', 'yuan', 'jun', 'qun', 'xun', 'yun', 'er', 'r'
)

SYL = re.compile(pinyin.syllable)
A_SYL = re.compile(pinyin.a_syl)
N_SYL = re.compile(pinyin.n_syl)
WORD = re.compile(pinyin.word)
N_WORD = re.compile(pinyin.n_word)
A_WORD = re.compile(pinyin.a_word)
SENT = re.compile(pinyin.sentence)
N_SENT = re.compile(pinyin.n_sent)
A_SENT = re.compile(pinyin.a_sent)


VOWELS = 'aeiou\u00FC'
VOWEL_MAP = {
    'a1': '\u0101', 'a2': '\xe1', 'a3': '\u01ce', 'a4': '\xe0', 'a5': 'a',
    'e1': '\u0113', 'e2': '\xe9', 'e3': '\u011b', 'e4': '\xe8', 'e5': 'e',
    'i1': '\u012b', 'i2': '\xed', 'i3': '\u01d0', 'i4': '\xec', 'i5': 'i',
    'o1': '\u014d', 'o2': '\xf3', 'o3': '\u01d2', 'o4': '\xf2', 'o5': 'o',
    'u1': '\u016b', 'u2': '\xfa', 'u3': '\u01d4', 'u4': '\xf9', 'u5': 'u',
    '\u00fc1': '\u01d6', '\u00fc2': '\u01d8', '\u00fc3': '\u01da',
    '\u00fc4': '\u01dc', '\u00fc5': '\u00fc'
}


def _num_vowel_to_acc(vowel, tone):
    """Convert a numbered vowel to an accented vowel."""
    try:
        return VOWEL_MAP[vowel + str(tone)]
    except IndexError:
        raise ValueError("Vowel must be one of '{}' and tone must be an int"
                         "1-5.".format(VOWELS))


def num_syl_to_acc(syllable):
    """Convert a numbered pinyin syllable to an accented pinyin syllable.

    Implements the following algorithm:
        1. If the syllable has an 'a' or 'e', put the tone over that vowel.
        2. If the syllable has 'ou', place the tone over the 'o'.
        3. Otherwise, put the tone on the last vowel.

    """
    if syllable.startswith('r') and len(syllable) <= 2:
        return 'r'  # Special case for 'r' syllable.
    if re.search('[{}]'.format(VOWELS), syllable) is None:
        return syllable
    syl, tone = syllable[:-1], syllable[-1]
    if tone not in '12345':
        # We did not find a tone number. Abort conversion.
        return syl
    syl = re.sub('u:|v', '\u00fc', syl)
    if 'a' in syl:
        return syl.replace('a', _num_vowel_to_acc('a', tone))
    elif 'e' in syl:
        return syl.replace('e', _num_vowel_to_acc('e', tone))
    elif 'ou' in syl:
        return syl.replace('o', _num_vowel_to_acc('o', tone))
    last_vowel = syl[max(map(syl.rfind, VOWELS))]  # Find last vowel index.
    return syl.replace(last_vowel, _num_vowel_to_acc(last_vowel, tone))


class TestPinyinSyllables(unittest.TestCase):

    maxDiff = None

    def test_number_syllables(self):
        vs = list(VALID_SYLS)
        _vs = []
        for n in range(0, len(vs)):
            vs[n] = vs[n] + str(random.randint(1, 5))
            _vs.append(vs[n])
            if _vs[n][0] in 'aeo':
                _vs[n] = "'{}".format(_vs[n])
        s = ''.join(_vs)
        self.assertEqual(SYL.findall(s), vs)
        self.assertEqual(N_SYL.findall(s), vs)

    def test_accent_syllables(self):
        vs = list(VALID_SYLS)
        _vs = []
        for n in range(0, len(vs)):
            syl = vs[n]
            vs[n] = num_syl_to_acc(vs[n] + str(random.randint(1, 5)))
            _vs.append(vs[n])
            if syl[0] in 'aeo':
                _vs[n] = "'{}".format(_vs[n])
        s = ''.join(_vs)
        self.assertEqual(SYL.findall(s), vs)
        self.assertEqual(A_SYL.findall(s), vs)


def create_word(accented=False):
    if accented:
        tone = lambda: str(random.randint(1, 5))
        vs = [num_syl_to_acc(s + tone()) for s in VALID_SYLS]
    else:
        vs = [s + str(random.randint(1, 5)) for s in VALID_SYLS]
    word = vs[random.randint(0, len(vs) - 1)]
    for n in range(1, WORD_LENGTH):
        num = random.randint(0, len(vs) - 1)
        word += ['-', ''][random.randint(0, 1)]
        if VALID_SYLS[num][0] in 'aeo' and word[-1] != '-':
            word += "'"
        word += vs[num]
    return word


class TestPinyinWords(unittest.TestCase):

    def test_number_words(self):
        for n in range(0, NUM_WORDS):
            word = create_word()
            self.assertEqual(WORD.match(word).group(0), word)
            self.assertEqual(N_WORD.match(word).group(0), word)

    def test_accent_words(self):
        for n in range(0, NUM_WORDS):
            word = create_word(accented=True)
            self.assertEqual(WORD.match(word).group(0), word)
            self.assertEqual(A_WORD.match(word).group(0), word)


def create_sentence(accented=False):
    _sent = []
    for n in range(0, SENT_LENGTH):
        _sent.append(create_word(accented=accented))
    sentence = [_sent.pop(0)]
    sentence.extend([random.choice([' ', ', ', '; ']) + w for w in _sent])
    return ''.join(sentence) + '.'


class TestPinyinSentences(unittest.TestCase):

    def test_number_sentences(self):
        for n in range(0, NUM_SENT):
            sentence = create_sentence()
            self.assertEqual(SENT.match(sentence).group(0), sentence)
            self.assertEqual(N_SENT.match(sentence).group(0), sentence)

    def test_accent_sentences(self):
        for n in range(0, NUM_SENT):
            sentence = create_sentence(accented=True)
            self.assertEqual(SENT.match(sentence).group(0), sentence)
            self.assertEqual(A_SENT.match(sentence).group(0), sentence)
