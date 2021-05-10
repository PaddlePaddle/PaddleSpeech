env = {}
with open('./pypinyin/pinyin_dict.py') as fp:
    exec(fp.read(), env, env)
with open('./pypinyin/phrases_dict_large.py') as fp:
    exec(fp.read(), env, env)


def get_pinyins_via_pinyin_dict(phrases):
    pinyins = []
    for han in phrases:
        pinyin = env['pinyin_dict'][ord(han)].split(',')[0]
        pinyins.append([pinyin])

    return pinyins


def save(new_dict, output_file):
    with open(output_file, 'w') as out_fp:
        out_fp.write('''# Warning: Auto-generated file, don't edit.
phrases_dict = {
''')

        hanzi_pairs = sorted(new_dict.items(), key=lambda x: x[0])
        for hanzi, pinyin_list in hanzi_pairs:
            #     中国: [[zhōng], [guó]]
            new_line = "    '{hanzi}': {pinyin_list},\n".format(
                hanzi=hanzi.strip(), pinyin_list=pinyin_list)
            out_fp.write(new_line)

        out_fp.write('}\n')


def double_check():
    import pypinyin

    missing_dict = {}
    for phrases, pinyins in env['phrases_dict'].items():
        if pypinyin.pinyin(phrases, heteronym=True) != pinyins:
            missing_dict[phrases] = pinyins

    return missing_dict


def tidy():
    new_dict = {}
    for phrases, pinyins in env['phrases_dict'].items():
        pinyins_via_pinyin_dict = get_pinyins_via_pinyin_dict(phrases)
        if pinyins != pinyins_via_pinyin_dict:
            new_dict[phrases] = pinyins

    return new_dict


def main():
    output = 'pypinyin/phrases_dict.py'
    new_dict = tidy()
    save(new_dict, output)

    missing_dict = double_check()
    new_dict.update(missing_dict)
    save(new_dict, output)


if __name__ == '__main__':
    main()
