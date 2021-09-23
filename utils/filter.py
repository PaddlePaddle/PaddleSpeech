#!/usr/bin/env python3
# Apache 2.0
import argparse
import codecs
import sys

is_python2 = sys.version_info[0] == 2


def get_parser():
    parser = argparse.ArgumentParser(
        description="filter words in a text file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument(
        "--exclude",
        "-v",
        dest="exclude",
        action="store_true",
        help="exclude filter words", )
    parser.add_argument("filt", type=str, help="filter list")
    parser.add_argument("infile", type=str, help="input file")
    return parser


def main(args):
    args = get_parser().parse_args(args)
    filter_file(args.infile, args.filt, args.exclude)


def filter_file(infile, filt, exclude):
    vocab = set()
    with codecs.open(filt, "r", encoding="utf-8") as vocabfile:
        for line in vocabfile:
            vocab.add(line.strip())

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout
                                           if is_python2 else sys.stdout.buffer)
    with codecs.open(infile, "r", encoding="utf-8") as textfile:
        for line in textfile:
            if exclude:
                print(" ".join(
                    map(
                        lambda word: word if word not in vocab else "",
                        line.strip().split(), )))
            else:
                print(" ".join(
                    map(
                        lambda word: word if word in vocab else "<UNK>",
                        line.strip().split(), )))


if __name__ == "__main__":
    main(sys.argv[1:])
