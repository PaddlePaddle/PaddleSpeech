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
parser.add_argument("--device", type=str, default='gpu', choices=['cpu', 'gpu'])
parser.add_argument("--wav_en", type=str)
parser.add_argument("--wav_zh", type=str)
args = parser.parse_args()
# yapf: enable

if __name__ == '__main__':
    paddle.set_device(args.device)

    s2t_en_model = hub.Module(name='u2_conformer_librispeech')
    s2t_zh_model = hub.Module(name='u2_conformer_aishell')

    args.wav_en = os.path.abspath(os.path.expanduser(args.wav_en))
    args.wav_zh = os.path.abspath(os.path.expanduser(args.wav_zh))

    assert os.path.isfile(args.wav_en) and os.path.isfile(
        args.wav_zh), 'Wav files not exist.'

    print('[S2T][en]Wav: {}'.format(args.wav_en))
    text_en = s2t_en_model.speech_recognize(args.wav_en)
    print('[S2T][en]Text: {}'.format(text_en))

    print('[S2T][zh]Wav: {}'.format(args.wav_zh))
    text_zh = s2t_zh_model.speech_recognize(args.wav_zh)
    print('[S2T][zh]Text: {}'.format(text_zh))
