#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#           2018 Xuankai Chang (Shanghai Jiao Tong University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import argparse
import logging
import sys

import jsonlines
from utility import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description="convert a json to a transcription file with a token dictionary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
    parser.add_argument("json", type=str, help="jsonlines files")
    parser.add_argument("dict", type=str, help="dict, not used.")
    parser.add_argument(
        "--num-spkrs", type=int, default=1, help="number of speakers")
    parser.add_argument(
        "--refs", type=str, nargs="+", help="ref for all speakers")
    parser.add_argument(
        "--hyps", type=str, nargs="+", help="hyp for all outputs")
    return parser


def main(args):
    args = get_parser().parse_args(args)
    convert(args.json, args.dict, args.refs, args.hyps, args.num_spkrs)


def convert(jsonf, dic, refs, hyps, num_spkrs=1):
    n_ref = len(refs)
    n_hyp = len(hyps)
    assert n_ref == n_hyp
    assert n_ref == num_spkrs

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    logging.info("reading %s", jsonf)
    with jsonlines.open(jsonf, "r") as f:
        j = [item for item in f]

    logging.info("reading %s", dic)
    with open(dic, "r") as f:
        dictionary = f.readlines()
    char_list = [entry.split(" ")[0] for entry in dictionary]
    char_list.insert(0, "<blank>")
    char_list.append("<eos>")

    for ns in range(num_spkrs):
        hyp_file = open(hyps[ns], "w")
        ref_file = open(refs[ns], "w")

        for x in j:
            # recognition hypothesis
            if num_spkrs == 1:
                #seq = [char_list[int(i)] for i in x['hyps_tokenid'][0]]
                seq = x['hyps'][0]
            else:
                seq = [char_list[int(i)] for i in x['hyps_tokenid'][ns]]
            # In the recognition hypothesis,
            # the <eos> symbol is usually attached in the last part of the sentence
            # and it is removed below.
            #hyp_file.write(" ".join(seq).replace("<eos>", ""))
            hyp_file.write(seq.replace("<eos>", ""))
            # spk-uttid
            hyp_file.write(" (" + x["utt"] + ")\n")

            # reference
            if num_spkrs == 1:
                seq = x["refs"][0]
            else:
                seq = x['refs'][ns]
            # Unlike the recognition hypothesis,
            # the reference is directly generated from a token without dictionary
            # to avoid to include <unk> symbols in the reference to make scoring normal.
            # The detailed discussion can be found at
            # https://github.com/espnet/espnet/issues/993
            # ref_file.write(
            #     seq + " (" + j["utts"][x]["utt2spk"].replace("-", "_") + "-" + x + ")\n"
            # )
            ref_file.write(seq + " (" + x['utt'] + ")\n")

        hyp_file.close()
        ref_file.close()


if __name__ == "__main__":
    main(sys.argv[1:])
