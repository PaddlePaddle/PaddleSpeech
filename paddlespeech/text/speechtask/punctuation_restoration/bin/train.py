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
"""Trainer for punctuation_restoration task."""
import yaml
from paddle import distributed as dist
from speechtask.punctuation_restoration.training.trainer import Trainer
from speechtask.punctuation_restoration.utils.default_parser import default_argument_parser
from speechtask.punctuation_restoration.utils.utility import print_arguments


def main_sp(config, args):
    exp = Trainer(config, args)
    exp.setup()
    exp.run()


def main(config, args):
    if args.ngpu > 1:
        dist.spawn(main_sp, args=(config, args), nprocs=args.ngpu)
    else:
        main_sp(config, args)


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    print_arguments(args, globals())

    # https://yaml.org/type/float.html
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print(config)
    if args.dump_config:
        with open(args.dump_config, 'w') as f:
            print(config, file=f)

    main(config, args)
