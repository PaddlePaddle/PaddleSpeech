# -*- coding: utf-8 -*-
"""生成初始的 kMandarin_8105.txt"""
from merge_unihan import parse_pinyins, code_to_hanzi


def parse_china_x():
    with open('tools/china-8105-06062014.txt') as fp:
        for line in fp:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            yield line.split()[0]


def parse_zdic():
    with open('zdic.txt') as fp:
        return parse_pinyins(fp)


def parse_kmandain():
    with open('pinyin.txt') as fp:
        return parse_pinyins(fp)


def diff(kmandarin, zdic, commons):
    for key in commons:
        hanzi = code_to_hanzi(key)
        if key in kmandarin:
            value = kmandarin[key][0]
            if key in zdic and value != zdic[key][0]:
                yield '{0}: {1}  # {2} -> {3}'.format(
                    key, value, hanzi, zdic[key][0]
                )
            else:
                yield '{0}: {1}  # {2}'.format(key, value, hanzi)
        elif key in zdic:
            value = zdic[key][0]
            yield '{0}: {1}  # {2}'.format(key, value, hanzi)
        else:
            yield '# {0}: {1}  # {2}'.format(key, '<-', hanzi)

if __name__ == '__main__':
    zdic = parse_zdic()
    kmandarin = parse_kmandain()
    commons = parse_china_x()
    lst = diff(kmandarin, zdic, commons)
    for x in lst:
        print(x)
