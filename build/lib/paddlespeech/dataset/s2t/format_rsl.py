# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
"""
format ref/hyp file for `utt text` format to compute CER/WER/MER.

norm:
BAC009S0764W0196 明确了发展目标和重点任务
BAC009S0764W0186 实现我国房地产市场的平稳运行


sclite:
加大对结构机械化环境和收集谈控机制力度(BAC009S0906W0240.wav)
河南省新乡市丰秋县刘光镇政府东五零左右(BAC009S0770W0441.wav)
"""
import argparse

import jsonlines

from paddlespeech.utils.argparse import print_arguments


def transform_hyp(origin, trans, trans_sclite):
    """
    Args:
        origin: The input json file which contains the model output
        trans: The output file for caculate CER/WER
        trans_sclite: The output file for caculate CER/WER using sclite
    """
    input_dict = {}

    with open(origin, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            input_dict[item["utt"]] = item["hyps"][0]

    if trans:
        with open(trans, "w+", encoding="utf8") as f:
            for key in input_dict.keys():
                f.write(key + " " + input_dict[key] + "\n")
        print(f"transform_hyp output: {trans}")

    if trans_sclite:
        with open(trans_sclite, "w+") as f:
            for key in input_dict.keys():
                line = input_dict[key] + "(" + key + ".wav" + ")" + "\n"
                f.write(line)
        print(f"transform_hyp output: {trans_sclite}")


def transform_ref(origin, trans, trans_sclite):
    """
    Args:
        origin: The input json file which contains the model output
        trans: The output file for caculate CER/WER
        trans_sclite: The output file for caculate CER/WER using sclite
    """
    input_dict = {}

    with open(origin, "r", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            input_dict[item["utt"]] = item["text"]

    if trans:
        with open(trans, "w", encoding="utf8") as f:
            for key in input_dict.keys():
                f.write(key + " " + input_dict[key] + "\n")
        print(f"transform_hyp output: {trans}")

    if trans_sclite:
        with open(trans_sclite, "w") as f:
            for key in input_dict.keys():
                line = input_dict[key] + "(" + key + ".wav" + ")" + "\n"
                f.write(line)
        print(f"transform_hyp output: {trans_sclite}")


def define_argparse():
    parser = argparse.ArgumentParser(
        prog='format ref/hyp file for compute CER/WER', add_help=True)
    parser.add_argument(
        '--origin_hyp', type=str, default="", help='origin hyp file')
    parser.add_argument(
        '--trans_hyp',
        type=str,
        default="",
        help='hyp file for caculating CER/WER')
    parser.add_argument(
        '--trans_hyp_sclite',
        type=str,
        default="",
        help='hyp file for caculating CER/WER by sclite')

    parser.add_argument(
        '--origin_ref', type=str, default="", help='origin ref file')
    parser.add_argument(
        '--trans_ref',
        type=str,
        default="",
        help='ref file for caculating CER/WER')
    parser.add_argument(
        '--trans_ref_sclite',
        type=str,
        default="",
        help='ref file for caculating CER/WER by sclite')
    parser_args = parser.parse_args()
    return parser_args


def format_result(origin_hyp="",
                  trans_hyp="",
                  trans_hyp_sclite="",
                  origin_ref="",
                  trans_ref="",
                  trans_ref_sclite=""):

    if origin_hyp:
        transform_hyp(
            origin=origin_hyp, trans=trans_hyp, trans_sclite=trans_hyp_sclite)

    if origin_ref:
        transform_ref(
            origin=origin_ref, trans=trans_ref, trans_sclite=trans_ref_sclite)


def main():
    args = define_argparse()
    print_arguments(args, globals())

    format_result(**vars(args))


if __name__ == "__main__":
    main()
