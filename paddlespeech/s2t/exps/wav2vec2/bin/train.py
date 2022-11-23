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
"""Trainer for wav2vec2.0 model."""
import cProfile
import os

from yacs.config import CfgNode

from paddlespeech.s2t.exps.wav2vec2.model import Wav2Vec2ASRTrainer as Trainer
from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.s2t.utils.utility import print_arguments


def main_sp(config, args):
    exp = Trainer(config, args)
    exp.setup()
    exp.run()


def main(config, args):
    main_sp(config, args)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        '--resume', type=str, default="", nargs="?", help='resume ckpt path.')
    args = parser.parse_args()
    print_arguments(args, globals())
    # https://yaml.org/type/float.html
    config = CfgNode(new_allowed=True)
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    if args.dump_config:
        with open(args.dump_config, 'w') as f:
            print(config, file=f)

    # Setting for profiling
    pr = cProfile.Profile()
    pr.runcall(main, config, args)
    pr.dump_stats(os.path.join(args.output, 'train.profile'))
