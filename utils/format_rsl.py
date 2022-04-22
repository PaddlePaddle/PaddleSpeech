# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse

import jsonlines


def trans_hyp(origin_hyp, trans_hyp=None, trans_hyp_sclite=None):
    """
    Args:
        origin_hyp: The input json file which contains the model output
        trans_hyp: The output file for caculate CER/WER
        trans_hyp_sclite: The output file for caculate CER/WER using sclite
    """
    input_dict = {}

    with open(origin_hyp, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            input_dict[item["utt"]] = item["hyps"][0]
    if trans_hyp is not None:
        with open(trans_hyp, "w+", encoding="utf8") as f:
            for key in input_dict.keys():
                f.write(key + " " + input_dict[key] + "\n")
    if trans_hyp_sclite is not None:
        with open(trans_hyp_sclite, "w+") as f:
            for key in input_dict.keys():
                line = input_dict[key] + "(" + key + ".wav" + ")" + "\n"
                f.write(line)


def trans_ref(origin_ref, trans_ref=None, trans_ref_sclite=None):
    """
    Args:
        origin_hyp: The input json file which contains the model output
        trans_hyp: The output file for caculate CER/WER
        trans_hyp_sclite: The output file for caculate CER/WER using sclite
    """
    input_dict = {}

    with open(origin_ref, "r", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            input_dict[item["utt"]] = item["text"]
    if trans_ref is not None:
        with open(trans_ref, "w", encoding="utf8") as f:
            for key in input_dict.keys():
                f.write(key + " " + input_dict[key] + "\n")

    if trans_ref_sclite is not None:
        with open(trans_ref_sclite, "w") as f:
            for key in input_dict.keys():
                line = input_dict[key] + "(" + key + ".wav" + ")" + "\n"
                f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='format hyp file for compute CER/WER', add_help=True)
    parser.add_argument(
        '--origin_hyp', type=str, default=None, help='origin hyp file')
    parser.add_argument(
        '--trans_hyp',
        type=str,
        default=None,
        help='hyp file for caculating CER/WER')
    parser.add_argument(
        '--trans_hyp_sclite',
        type=str,
        default=None,
        help='hyp file for caculating CER/WER by sclite')

    parser.add_argument(
        '--origin_ref', type=str, default=None, help='origin ref file')
    parser.add_argument(
        '--trans_ref',
        type=str,
        default=None,
        help='ref file for caculating CER/WER')
    parser.add_argument(
        '--trans_ref_sclite',
        type=str,
        default=None,
        help='ref file for caculating CER/WER by sclite')
    parser_args = parser.parse_args()

    if parser_args.origin_hyp is not None:
        trans_hyp(
            origin_hyp=parser_args.origin_hyp,
            trans_hyp=parser_args.trans_hyp,
            trans_hyp_sclite=parser_args.trans_hyp_sclite, )

    if parser_args.origin_ref is not None:
        trans_ref(
            origin_ref=parser_args.origin_ref,
            trans_ref=parser_args.trans_ref,
            trans_ref_sclite=parser_args.trans_ref_sclite, )
