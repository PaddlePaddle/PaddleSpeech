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
import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--lang", type=str, default='zh', choices=['zh', 'en'])
parser.add_argument("--device", type=str, default='gpu', choices=['cpu', 'gpu'])
parser.add_argument("--text", type=str, nargs='+')
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()
# yapf: enable

if __name__ == '__main__':
    paddle.set_device(args.device)

    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    if args.lang == 'zh':
        t2s_model = hub.Module(name='fastspeech2_baker', output_dir=output_dir)
    else:
        t2s_model = hub.Module(
            name='fastspeech2_ljspeech', output_dir=output_dir)

    if isinstance(args.text, list):
        args.text = ' '.join(args.text)

    wavs = t2s_model.generate([args.text], device=args.device)
    print('[T2S]Wav file has been generated: {}'.format(wavs[0]))
