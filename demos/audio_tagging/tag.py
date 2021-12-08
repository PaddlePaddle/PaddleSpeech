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
import argparse

import paddle

from paddlespeech.cli import CLSExecutor

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument(
    '--input', type=str, required=True, help='Audio file to recognize.')
args = parser.parse_args()
# yapf: enable

if __name__ == '__main__':
    cls_executor = CLSExecutor()
    result = cls_executor(
        model_type='panns_cnn14',
        cfg_path=None,  # Set `cfg_path` and `ckpt_path` to None to use pretrained model.
        label_file=None,
        ckpt_path=None,
        audio_file=args.input,
        topk=10,
        device=paddle.get_device(), )
    print('CLS Result: \n{}'.format(result))
