#!/usr/bin/env python
import argparse
import json


def main(args):
    with open(args.json_file, 'r') as fin:
        data_json = json.load(fin)

    # manifest format:
    # {"input": [
    #       {"feat": "dev/deltafalse/feats.1.ark:842920", "name": "input1", "shape": [349, 83]}
    #  ],
    #  "output": [
    #       {"name": "target1", "shape": [12, 5002], "text": "NO APOLLO", "token": "▁NO ▁A PO LL O", "tokenid": "3144 482 352 269 317"}
    #  ],
    #  "utt2spk": "116-288045",
    #  "utt": "116-288045-0019"}
    with open(args.manifest_file, 'w') as fout:
        for key, value in data_json['utts'].items():
            value['utt'] = key
            fout.write(json.dumps(value, ensure_ascii=False))
            fout.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json-file',
                        type=str,
                        default=None,
                        help="espnet data json file.")
    parser.add_argument('--manifest-file',
                        type=str,
                        default='maniefst.train',
                        help='manifest data json line file.')
    args = parser.parse_args()
    main(args)
