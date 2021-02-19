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
"""Inferer for DeepSpeech2 model."""

import io
import logging
import argparse
import functools

from paddle import distributed as dist

from utils.utility import print_arguments
from training.cli import default_argument_parser

from model_utils.config import get_cfg_defaults
from model_utils.model import DeepSpeech2Tester as Tester
from utils.error_rate import char_errors, word_errors


def main_sp(config, args):
    exp = Tester(config, args)
    exp.setup()
    exp.run_test()


def main(config, args):
    main_sp(config, args)


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    print_arguments(args)

    # https://yaml.org/type/float.html
    config = get_cfg_defaults()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config)
    if args.dump_config:
        with open(args.dump_config, 'w') as f:
            print(config, file=f)

    main(config, args)
