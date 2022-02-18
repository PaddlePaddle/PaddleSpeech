#!/usr/bin/env python
import argparse
import json


def main(args):
    with open(args.json_file, 'r') as fin:
        data_json = json.load(fin)

    with open(args.manifest_file, 'w') as fout:
        for key, value in data_json['utts'].items():
            value['utt'] = key
            fout.write(json.dumps(value, ensure_ascii=False))
            fout.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--json-file', type=str, default=None, help="espnet data json file.")
    parser.add_argument(
        '--manifest-file',
        type=str,
        default='manifest.train',
        help='manifest data json line file.')
    args = parser.parse_args()
    main(args)
