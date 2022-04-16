#!/usr/bin/env python3
"""Manifest file to key-value files."""
import argparse
import functools
from pathlib import Path

import jsonlines

from utils.utility import add_arguments
from utils.utility import print_arguments


def main(args):
    print_arguments(args, globals())

    count = 0

    outdir = Path(args.output_path)
    wav_scp = outdir / 'wav.scp'
    dur_scp = outdir / 'duration'
    text_scp = outdir / 'text'

    with jsonlines.open(args.manifest_path, 'r') as reader:
        manifest_jsons = list(reader)

    with wav_scp.open('w') as fwav, dur_scp.open('w') as fdur, text_scp.open(
            'w') as ftxt:
        for line_json in manifest_jsons:
            # utt:str
            # utt2spk:str
            # input: [{name:str, shape:[dur_in_sec, feat_dim], feat:str, filetype:str}, ]
            # output: [{name:str, shape:[tokenlen, vocab_dim], text:str, token:str, tokenid:str}, ] 
            utt = line_json['utt']
            utt2spk = line_json['utt2spk']

            # input
            assert (len(line_json['input']) == 1), "only support one input now"
            input_json = line_json['input'][0]
            feat = input_json['feat']
            feat_shape = input_json['shape']
            file_type = input_json['filetype']

            file_ext = Path(feat).suffix  # .wav
            dur = feat_shape[0]
            feat_dim = feat_shape[1]

            if file_ext == '.wav':
                fwav.write(f"{utt} {feat}\n")
            fdur.write(f"{utt} {dur}\n")

            # output
            assert (
                len(line_json['output']) == 1), "only support one output now"
            output_json = line_json['output'][0]
            text = output_json['text']
            if 'token' in output_json:
                tokens = output_json['token']
                tokenids = output_json['tokenid']
                token_shape = output_json['shape']
                token_len = token_shape[0]
                vocab_dim = token_shape[1]
            ftxt.write(f"{utt} {text}\n")

            count += 1

    print(f"Examples number: {count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    # yapf: disable
    add_arg('manifest_path',    str,
            'data/librispeech/manifest.train',
            "Filepath of manifest to compute normalizer's mean and stddev.")
    add_arg('output_path',    str,
            'data/train',
            "dir path to dump wav.scp/duaration/text files.")
    # yapf: disable
    args = parser.parse_args()

    main(args)
