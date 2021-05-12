# -*- coding: utf-8 -*-
"""补充 8105 中汉字的拼音数据"""
from collections import namedtuple
import re
import sys

from pyquery import PyQuery
import requests

re_pinyin = re.compile(r'拼音：(?P<pinyin>\S+) ')
re_code = re.compile(r'统一码\w?：(?P<code>\S+) ')
re_alternate = re.compile(r'异体字：\s+?(?P<alternate>\S+)')
HanziInfo = namedtuple('HanziInfo', 'pinyin code alternate')


def fetch_html(url, params):
    response = requests.get(url, params=params)
    return response.content


def fetch_info(hanzi):
    url = 'http://www.guoxuedashi.com/zidian/so.php'
    params = {
        'sokeyzi': hanzi,
        'kz': 1,
        'submit': '',
    }
    html = fetch_html(url, params)
    pq = PyQuery(html)
    pq = PyQuery(pq('table.zui td')[1])
    text = pq('tr').text()
    text_alternate = pq(html)('.info_txt2')('em').text()

    pinyin = ''
    pinyin_match = re_pinyin.search(text)
    if pinyin_match is not None:
        pinyin = pinyin_match.group('pinyin')
    code = re_code.search(text).group('code')
    alternate = ''
    alternate_match = re_alternate.search(text_alternate)
    if alternate_match is not None:
        alternate = alternate_match.group('alternate')

    return HanziInfo(pinyin, code, alternate)


def parse_hanzi(hanzi):
    info = fetch_info(hanzi)
    if (not info.pinyin) and info.alternate:
        alternate = fetch_info(info.alternate)
    else:
        alternate = ''
    return HanziInfo(info.pinyin, info.code, alternate)


def main(lines):
    for line in lines:
        if line.startswith('# U+') and '<-' in line:
            # # U+xxx ... -> U+xxx
            code = line.split(':')[0].strip('# ')
            # U+xxx -> xxx
            code = code[2:]
            info = parse_hanzi(code)
            pinyin = info.pinyin
            extra = ''
            if (not pinyin) and info.alternate:
                alternate = info.alternate
                pinyin = alternate.pinyin
                extra = '  => U+{0}'.format(alternate.code)
                if ',' in pinyin:
                    first_pinyin, extra_pinyin = pinyin.split(',', 1)
                    pinyin = first_pinyin
                    extra += '  ?-> ' + extra_pinyin
            if pinyin:
                line = line.strip()
                # # U+xxx -> U+xxx
                line = line[2:]
                line = line.replace('<-', pinyin)
                if extra:
                    line += extra
        yield line.strip()

if __name__ == '__main__':
    args = sys.argv[1:]
    input_file = args[0]
    with open(input_file) as fp:
        for line in main(fp):
            print(line)
