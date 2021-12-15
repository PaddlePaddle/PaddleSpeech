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
import os

import paddle

from paddlespeech.cli import ASRExecutor
from paddlespeech.cli import TextExecutor

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--device", type=str, default=paddle.get_device())
args = parser.parse_args()
# yapf: enable

if __name__ == "__main__":
    asr_executor = ASRExecutor()
    text_executor = TextExecutor()

    text = asr_executor(
        audio_file=os.path.abspath(os.path.expanduser(args.input)),
        device=args.device)
    result = text_executor(
        text=text,
        task='punc',
        model='ernie_linear_p3_wudao',
        device=args.device)

    print('ASR Result: \n{}'.format(text))
    print('Text Result: \n{}'.format(result))
