# -*- coding: utf-8 -*-

import os
import io
import re
import codecs
from pypinyin.phonetic_symbol import phonetic_symbol
from pypinyin.pinyin_dict import pinyin_dict
from pypinyin.style.tone import ToneConverter

ROOT = os.path.dirname(os.path.realpath(__file__))


tone_converter = ToneConverter()
tone3_2_tone_dict = {}
for k, v in pinyin_dict.items():
    parts = v.split(',')
    for part in parts:
        part = part.strip()
        if part:
            tone3 = tone_converter.to_tone3(part).strip().lower()
            if tone3:
                tone3_2_tone_dict[tone3] = part


def tone3_to_tone1(tone3):
    tone3 = tone3.strip().lower()
    # 儿化
    if tone3 == 'r5':
        return 'er'
    # 轻声
    if '5' in tone3:
        new = tone3.replace('5', '')
        if new:
            return new
    # 律
    if 'u:' in tone3:
        tone3 = tone3.replace('u:', 'v')

    return tone3_2_tone_dict[tone3]


if __name__ == '__main__':
    LINE_PARTS_RE = re.compile(
        r'(?P<zht>\w+)\s+(?P<zhs>\w+)\s+\[(?P<py>.+?)\]')
    LETTER_DIGIT_RE = re.compile(r'[a-zA-Z0-9]')
    cnt = 0
    with codecs.open(os.path.join(ROOT, 'cc_cedict.txt'), 'w', 'utf-8-sig') as fpw:
        with codecs.open(os.path.join(ROOT, 'cedict_ts.u8'), 'r', 'utf-8-sig') as fpr:
            for line in fpr:
                line_stripped = line.strip()
                if not line or line_stripped[0] == '#' or line_stripped[0] == '%':
                    continue
                # print(line_stripped)
                parts = LINE_PARTS_RE.match(line_stripped)
                if not parts:
                    continue
                zhs = parts.group('zhs')
                py = parts.group('py').split()
                try:
                    tone1 = [tone3_to_tone1(i) for i in py]
                except Exception as e:
                    print(e)
                    #input()
                    continue
                #print(zhs, py, tone1)
                if LETTER_DIGIT_RE.search(zhs):
                    continue
                if len(zhs) < 2:
                    continue
                fpw.write(f'{zhs}: {" ".join(tone1)}\n')
                cnt += 1
                if cnt % 10000 == 0:
                    print(f'{cnt} lines processed...')
