#!/usr/bin/env python3
"""remove longshort data from manifest"""
import argparse
import logging

import jsonlines

from paddlespeech.s2t.utils.cli_utils import get_commandline_args

# manifest after format
# josnline like this
# {
#   "input": [{"name": "input1", "shape": (100, 83), "feat": "xxx.ark:123"}],
#   "output": [{"name":"target1", "shape": (40, 5002), "text": "a b c de"}],
#   "utt2spk": "111-2222",
#   "utt": "111-2222-333"
# }


def get_parser():
    parser = argparse.ArgumentParser(
        description="remove longshort data from format manifest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument(
        "--verbose", "-V", default=0, type=int, help="Verbose option")
    parser.add_argument(
        "--iaxis",
        default=0,
        type=int,
        help="multi inputs index, 0 is the first")
    parser.add_argument(
        "--oaxis",
        default=0,
        type=int,
        help="multi outputs index, 0 is the first")
    parser.add_argument("--maxframes", default=2000, type=int, help="maxframes")
    parser.add_argument("--minframes", default=10, type=int, help="minframes")
    parser.add_argument("--maxchars", default=200, type=int, help="max tokens")
    parser.add_argument("--minchars", default=0, type=int, help="min tokens")
    parser.add_argument(
        "--stride_ms", default=10, type=int, help="stride in ms unit.")
    parser.add_argument(
        "rspecifier",
        type=str,
        help="jsonl format manifest. e.g. manifest.jsonl")
    parser.add_argument(
        "wspecifier_or_wxfilename",
        type=str,
        help="Write specifier. e.g. manifest.jsonl")
    return parser


def filter_input(args, line):
    tmp = line['input'][args.iaxis]
    if args.sound:
        # second to frame
        nframe = tmp['shape'][0] * 1000 / args.stride_ms
    else:
        nframe = tmp['shape'][0]

    if nframe < args.minframes or nframe > args.maxframes:
        return True
    else:
        return False


def filter_output(args, line):
    nchars = len(line['output'][args.iaxis]['text'])
    if nchars < args.minchars or nchars > args.maxchars:
        return True
    else:
        return False


def main():
    args = get_parser().parse_args()

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    with jsonlines.open(args.rspecifier, 'r') as reader:
        lines = list(reader)
    logging.info(f"Example: {len(lines)}")
    feat = lines[0]['input'][args.iaxis]['feat']
    args.soud = False
    if feat.split('.')[-1] not in 'ark, scp':
        args.sound = True

    count = 0
    filter = 0
    with jsonlines.open(args.wspecifier_or_wxfilename, 'w') as writer:
        for line in lines:
            if filter_input(args, line) or filter_output(args, line):
                filter += 1
                continue
            writer.write(line)
            count += 1
    logging.info(f"Example after filter: {count}\{filter}")


if __name__ == '__main__':
    main()
