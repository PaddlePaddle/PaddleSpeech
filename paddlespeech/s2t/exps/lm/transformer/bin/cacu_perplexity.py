# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import sys

import configargparse


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="The parser for caculating the perplexity of transformer language model ",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter, )

    parser.add_argument(
        "--rnnlm", type=str, default=None, help="RNNLM model file to read")

    parser.add_argument(
        "--rnnlm-conf",
        type=str,
        default=None,
        help="RNNLM model config file to read")

    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="vocab path to for token2id")

    parser.add_argument(
        "--bpeprefix",
        type=str,
        default=None,
        help="The path of bpeprefix for loading")

    parser.add_argument(
        "--text_path",
        type=str,
        default=None,
        help="The path of text file for testing ")

    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpu to use, 0 for using cpu instead")

    parser.add_argument(
        "--dtype",
        choices=("float16", "float32", "float64"),
        default="float32",
        help="Float precision (only available in --api v2)", )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="The output directory to store the sentence PPL")

    return parser


def main(args):
    parser = get_parser()
    args = parser.parse_args(args)
    from paddlespeech.s2t.exps.lm.transformer.lm_cacu_perplexity import run_get_perplexity
    run_get_perplexity(args)


if __name__ == "__main__":
    main(sys.argv[1:])
