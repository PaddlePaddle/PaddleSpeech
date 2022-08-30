import argparse
import os


def process_sentence(line):
    if line == '': return ''
    res = line[0]
    for i in range(1, len(line)):
        res += (' ' + line[i])
    return res


if __name__ == "__main__":
    paser = argparse.ArgumentParser(description="Input filename")
    paser.add_argument('-input_file')
    paser.add_argument('-output_file')
    sentence_cnt = 0
    args = paser.parse_args()
    with open(args.input_file, 'r') as f:
        with open(args.output_file, 'w') as write_f:
            while True:
                line = f.readline()
                if line:
                    sentence_cnt += 1
                    write_f.write(process_sentence(line))
                else:
                    break
    print('preprocess over')
    print('total sentences number:', sentence_cnt)
