# -*- coding: utf-8 -*-
import sys
import codecs


def parse(lines):
    """
    :yield: hanzi, others
    """
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue

        hanzi, others = line.split(':', 1)
        yield hanzi.strip(), others.strip()


def merge(pinyin_d_list):
    """
    :rtype: dict
    """
    final_d = {}
    for overwrite_d in pinyin_d_list:
        final_d.update(overwrite_d)
    return final_d


def sort(pinyin_d):
    """
    :rtype: list
    """
    return sorted(pinyin_d.items(), key=lambda x: x[0])


def output(pinyin_s):
    print('# version: 0.10.5')
    print('# source: https://github.com/mozillazg/phrase-pinyin-data')
    for hanzi, pinyin in pinyin_s:
        hanzi = hanzi.split('_')[0]
        print('{hanzi}: {pinyin}'.format(hanzi=hanzi, pinyin=pinyin))


def main(files):
    pinyin_d_list = []
    for name in files:
        with codecs.open(name, 'r', 'utf-8-sig') as fp:
            d = {}
            for h, p in parse(fp):
                d.setdefault(h, p)
            pinyin_d_list.append(d)

    pinyin_d = merge(pinyin_d_list)
    output(sort(pinyin_d))


if __name__ == '__main__':
    main(sys.argv[1:])
