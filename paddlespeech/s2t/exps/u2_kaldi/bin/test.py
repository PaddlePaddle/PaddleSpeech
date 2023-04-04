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
"""Evaluation for U2 model."""
import cProfile

from yacs.config import CfgNode

from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.s2t.utils.utility import print_arguments

model_test_alias = {
    "u2": "paddlespeech.s2t.exps.u2.model:U2Tester",
    "u2_kaldi": "paddlespeech.s2t.exps.u2_kaldi.model:U2Tester",
}


def main_sp(config, args):
    class_obj = dynamic_import(args.model_name, model_test_alias)
    exp = class_obj(config, args)
    with exp.eval():
        exp.setup()
        if args.run_mode == 'test':
            exp.run_test()
        elif args.run_mode == 'export':
            exp.run_export()
        elif args.run_mode == 'align':
            exp.run_align()


def main(config, args):
    main_sp(config, args)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        '--model-name',
        type=str,
        default='u2_kaldi',
        help='model name, e.g: deepspeech2, u2, u2_kaldi, u2_st')
    parser.add_argument('--run-mode',
                        type=str,
                        default='test',
                        help='run mode, e.g. test, align, export')
    parser.add_argument('--dict-path',
                        type=str,
                        default=None,
                        help='dict path.')
    # save asr result to
    parser.add_argument("--result-file",
                        type=str,
                        help="path of save the asr result")
    # save jit model to
    parser.add_argument("--export-path",
                        type=str,
                        help="path of the jit model to save")
    args = parser.parse_args()
    print_arguments(args, globals())

    config = CfgNode()
    config.set_new_allowed(True)
    config.merge_from_file(args.config)
    if args.decode_cfg:
        decode_confs = CfgNode(new_allowed=True)
        decode_confs.merge_from_file(args.decode_cfg)
        config.decode = decode_confs
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
    pr.dump_stats('test.profile')
