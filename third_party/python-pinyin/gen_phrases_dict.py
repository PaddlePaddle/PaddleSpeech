import sys


def remove_dup_items(lst):
    new_lst = []
    for item in lst:
        if item not in new_lst:
            new_lst.append(item)
    return new_lst


def parse(fp):
    phrases_dict = {}
    for line in in_fp.readlines():
        line = line.strip()
        if line.startswith('#') or not line:
            continue

        # 中国: zhōng guó
        data = line.split('#')[0]
        hanzi, pinyin = data.strip().split(':')
        hanzi = hanzi.strip()
        # [[zhōng], [guó]]
        pinyin_list = [[s] for s in pinyin.split()]

        if hanzi not in phrases_dict:
            phrases_dict[hanzi] = pinyin_list
        else:
            for index, value in enumerate(phrases_dict[hanzi]):
                value.extend(pinyin_list[index])
                phrases_dict[hanzi][index] = remove_dup_items(value)

    return phrases_dict


def main(in_fp, out_fp):
    out_fp.write('''# Warning: Auto-generated file, don't edit.
phrases_dict = {
''')

    hanzi_pairs = sorted(parse(in_fp).items(), key=lambda x: x[0])
    for hanzi, pinyin_list in hanzi_pairs:
        #     中国: [[zhōng], [guó]]
        new_line = "    '{hanzi}': {pinyin_list},\n".format(
            hanzi=hanzi.strip(), pinyin_list=pinyin_list)
        out_fp.write(new_line)

    out_fp.write('}\n')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('python gen_phrases_dict.py INPUT OUTPUT')
        sys.exit(1)

    in_f = sys.argv[1]
    out_f = sys.argv[2]

    with open(in_f) as in_fp, open(out_f, 'w') as out_fp:
        main(in_fp, out_fp)
