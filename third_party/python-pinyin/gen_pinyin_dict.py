import sys


def main(in_fp, out_fp):
    out_fp.write('''# Warning: Auto-generated file, don't edit.
pinyin_dict = {
''')

    for line in in_fp.readlines():
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        else:
            # line is U+4E2D: zhōng,zhòng  # 中
            # raw_line U+4E2D: zhōng,zhòng
            raw_line = line.split('#')[0].strip()
            # 0x4E2D: zhōng,zhòng
            new_line = raw_line.replace('U+', '0x')
            # 0x4E2D: 'zhōng,zhòng
            new_line = new_line.replace(': ', ": '")
            #     0x4E2D: 'zhōng,zhòng'\n
            new_line = "    {new_line}',\n".format(new_line=new_line)
            out_fp.write(new_line)

    out_fp.write('}\n')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('python gen_pinyin_dict.py INPUT OUTPUT')
        sys.exit(1)

    in_f = sys.argv[1]
    out_f = sys.argv[2]

    with open(in_f) as in_fp, open(out_f, 'w') as out_fp:
        main(in_fp, out_fp)
