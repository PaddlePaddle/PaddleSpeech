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
"""Data preparation for punctuation_restoration task."""
import yaml
from speechtask.punctuation_restoration.utils.default_parser import default_argument_parser
from speechtask.punctuation_restoration.utils.punct_pre import process_chinese_pure_senetence
from speechtask.punctuation_restoration.utils.punct_pre import process_english_pure_senetence
from speechtask.punctuation_restoration.utils.utility import print_arguments


# create dataset from raw data files
def main(config, args):
    print("Start preparing data from raw data.")
    if (config['type'] == 'chinese'):
        process_chinese_pure_senetence(config)
    elif (config['type'] == 'english'):
        print('english!!!!')
        process_english_pure_senetence(config)
    else:
        print('Error: Type should be chinese or english!!!!')
        raise ValueError('Type should be chinese or english')

    print("Finish preparing data.")


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    print_arguments(args, globals())

    # https://yaml.org/type/float.html
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # config.freeze()
    print(config)
    main(config, args)
