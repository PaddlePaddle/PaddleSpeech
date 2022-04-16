#!/usr/bin/env python3
import argparse

def main(args):
    with open(args.text, 'r') as fin, open(args.lexicon, 'w') as fout:
        for line in fin:
            line = line.strip()
            if args.has_key:
                utt, text = line.split(maxsplit=1)
                words = text.split()
            else:
                words = line.split()
            
            for word in words:
                val = " ".join(list(word))
                fout.write(f"{word}\t{val}\n")
                fout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='text(line:utt1 中国 人) to lexicon（line:中国 中 国).')
    parser.add_argument(
        '--has_key',
        default=True,
        help='text path, with utt or not')
    parser.add_argument(
        '--text',
        required=True,
        help='text path. line: utt1 中国 人 or 中国 人')
    parser.add_argument(
        '--lexicon',
        required=True,
        help='lexicon path. line:中国 中 国')
    args = parser.parse_args()
    print(args)

    main(args)
