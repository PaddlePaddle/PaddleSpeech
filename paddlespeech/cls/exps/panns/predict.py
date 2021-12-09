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

import numpy as np
import paddle
import paddle.nn.functional as F
from paddleaudio.backends import load as load_audio
from paddleaudio.datasets import ESC50
from paddleaudio.features import LogMelSpectrogram
from paddleaudio.features import melspectrogram

from paddlespeech.cls.models import cnn14
from paddlespeech.cls.models import SoundClassifier

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--wav", type=str, required=True, help="Audio file to infer.")
parser.add_argument("--feat_backend", type=str, choices=['numpy', 'paddle'], default='numpy', help="Choose backend to extract features from audio files.")
parser.add_argument("--top_k", type=int, default=1, help="Show top k predicted results")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint of model.")
args = parser.parse_args()
# yapf: enable


def extract_features(file: str, feat_backend: str='numpy',
                     **kwargs) -> paddle.Tensor:
    waveform, sr = load_audio(file, sr=None)

    if args.feat_backend == 'numpy':
        feat = melspectrogram(waveform, sr, **kwargs).transpose()
        feat = np.expand_dims(feat, 0)
        feat = paddle.to_tensor(feat)
    else:
        feature_extractor = LogMelSpectrogram(sr=sr, **kwargs)
        feat = feature_extractor(paddle.to_tensor(waveform).unsqueeze(0))
        feat = paddle.transpose(feat, [0, 2, 1])
    return feat


if __name__ == '__main__':

    model = SoundClassifier(
        backbone=cnn14(pretrained=False, extract_embedding=True),
        num_class=len(ESC50.label_list))
    model.set_state_dict(paddle.load(args.checkpoint))
    model.eval()

    feat = extract_features(args.wav, args.feat_backend)
    logits = model(feat)
    probs = F.softmax(logits, axis=1).numpy()

    sorted_indices = (-probs[0]).argsort()

    msg = f'[{args.wav}]\n'
    for idx in sorted_indices[:args.top_k]:
        msg += f'{ESC50.label_list[idx]}: {probs[0][idx]}\n'
    print(msg)
