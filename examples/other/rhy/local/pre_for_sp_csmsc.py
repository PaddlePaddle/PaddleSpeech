import argparse
import os
import re

replace_ = {"#1": "%", "#2": "`", "#3": "~", "#4": "$"}


def replace_rhy_with_punc(line):
    # r'[：、，；。？！,.:;"?!”’《》【】<=>{}()（）#&@“”^_|…\\]%*$', '', line)     #参考checkcheck_oov.py,
    line = re.sub(r'^$\*%', '', line)
    for r in replace_.keys():
        if r in line:
            line = line.replace(r, replace_[r])
    return line


def pre_and_write(data, file):
    with open(file, 'w') as rf:
        for d in data:
            d = d.split('\t')[1].strip()
            d = replace_rhy_with_punc(d)
            d = ' '.join(d) + ' \n'
            rf.write(d)


def main():
    parser = argparse.ArgumentParser(
        description="Train a Rhy prediction model.")
    parser.add_argument("--data", type=str, default="label_train-set.txt")
    parser.add_argument(
        "--processed_path", type=str, default="../data/rhy_predict")
    args = parser.parse_args()
    print(args.data, args.processed_path)
    os.makedirs(args.processed_path, exist_ok=True)

    with open(args.data) as rf:
        rf = rf.readlines()
    text = rf[0::2]
    len_ = len(text)
    lens = [int(len_ * 0.9), int(len_ * 0.05), int(len_ * 0.05)]
    files = ['train.txt', 'test.txt', 'dev.txt']

    i = 0
    for l_, file in zip(lens, files):
        file = os.path.join(args.processed_path, file)
        pre_and_write(text[i:i + l_], file)
        i = i + l_


if __name__ == "__main__":
    main()
