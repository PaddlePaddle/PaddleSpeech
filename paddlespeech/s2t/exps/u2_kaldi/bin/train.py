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
"""Trainer for U2 model."""
import cProfile
import os

from yacs.config import CfgNode

from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.utils.argparse import print_arguments

model_train_alias = {
    "u2": "paddlespeech.s2t.exps.u2.model:U2Trainer",
    "u2_kaldi": "paddlespeech.s2t.exps.u2_kaldi.model:U2Trainer",
}


def main_sp(config, args):
    class_obj = dynamic_import(args.model_name, model_train_alias)
    exp = class_obj(config, args)
    exp.setup()
    exp.run()


def main(config, args):
    main_sp(config, args)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        '--model-name',
        type=str,
        default='u2_kaldi',
        help='model name, e.g: deepspeech2, u2, u2_kaldi, u2_st')
    args = parser.parse_args()
    print_arguments(args, globals())

    config = CfgNode()
    config.set_new_allowed(True)
    config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config)
    if args.dump_config:
        with open(args.dump_config, 'w') as f:
            print(config, file=f)

    # Setting for profiling
    pr = cProfile.Profile()
    pr.runcall(main, config, args)
    pr.dump_stats(os.path.join(args.output, 'train.profile'))
