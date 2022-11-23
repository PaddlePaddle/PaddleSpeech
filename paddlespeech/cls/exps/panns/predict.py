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
import paddle.nn.functional as F
import yaml

from paddlespeech.audio.backends import load as load_audio
from paddlespeech.audio.features import LogMelSpectrogram
from paddlespeech.audio.utils import logger
from paddlespeech.cls.models import SoundClassifier
from paddlespeech.utils.dynamic_import import dynamic_import

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--cfg_path", type=str, required=True)
args = parser.parse_args()
# yapf: enable


def extract_features(file: str, **feat_conf) -> paddle.Tensor:
    file = os.path.abspath(os.path.expanduser(file))
    waveform, _ = load_audio(file, sr=feat_conf['sr'])
    feature_extractor = LogMelSpectrogram(**feat_conf)
    feat = feature_extractor(paddle.to_tensor(waveform).unsqueeze(0))
    feat = paddle.transpose(feat, [0, 2, 1])
    return feat


if __name__ == '__main__':

    args.cfg_path = os.path.abspath(os.path.expanduser(args.cfg_path))
    with open(args.cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    model_conf = config['model']
    data_conf = config['data']
    feat_conf = config['feature']
    predicting_conf = config['predicting']

    ds_class = dynamic_import(data_conf['dataset'])
    backbone_class = dynamic_import(model_conf['backbone'])

    model = SoundClassifier(
        backbone=backbone_class(pretrained=False, extract_embedding=True),
        num_class=len(ds_class.label_list))
    model.set_state_dict(paddle.load(predicting_conf['checkpoint']))
    model.eval()

    feat = extract_features(predicting_conf['audio_file'], **feat_conf)
    logits = model(feat)
    probs = F.softmax(logits, axis=1).numpy()

    sorted_indices = (-probs[0]).argsort()

    msg = f"[{predicting_conf['audio_file']}]\n"
    for idx in sorted_indices[:predicting_conf['top_k']]:
        msg += f'{ds_class.label_list[idx]}: {probs[0][idx]}\n'
    logger.info(msg)
