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
"""Export for U2 model."""
from yacs.config import CfgNode

from paddlespeech.s2t.exps.u2_st.model import U2STTester as Tester
from paddlespeech.s2t.training.cli import default_argument_parser
from paddlespeech.s2t.utils.utility import print_arguments


def main_sp(config, args):
    exp = Tester(config, args)
    with exp.eval():
        exp.setup()
        exp.run_export()


def main(config, args):
    main_sp(config, args)


if __name__ == "__main__":
    parser = default_argument_parser()
    # save jit model to
    parser.add_argument(
        "--export_path", type=str, help="path of the jit model to save")
    args = parser.parse_args()
    print_arguments(args, globals())

    # https://yaml.org/type/float.html
    config = CfgNode(new_allowed=True)
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
