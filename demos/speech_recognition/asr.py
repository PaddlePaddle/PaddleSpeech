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

from paddlespeech.cli import ASRExecutor

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument(
    '--input', type=str, required=True, help='Audio file to recognize.')
args = parser.parse_args()
# yapf: enable

if __name__ == '__main__':
    asr_executor = ASRExecutor()
    text = asr_executor(
        model='conformer_wenetspeech',
        lang='zh',
        sample_rate=16000,
        config=None,  # Set `conf` and `ckpt_path` to None to use pretrained model.
        ckpt_path=None,
        audio_file=args.input,
        device=paddle.get_device(), )
    print('ASR Result: \n{}'.format(text))
