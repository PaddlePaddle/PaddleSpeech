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
from paddleaudio.datasets import ESC50

from paddlespeech.cls.models import cnn14
from paddlespeech.cls.models import SoundClassifier

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint of model.")
parser.add_argument("--output_dir", type=str, default='./export', help="Path to save static model and its parameters.")
args = parser.parse_args()
# yapf: enable

if __name__ == '__main__':
    model = SoundClassifier(
        backbone=cnn14(pretrained=False, extract_embedding=True),
        num_class=len(ESC50.label_list))
    model.set_state_dict(paddle.load(args.checkpoint))
    model.eval()

    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, None, 64], dtype=paddle.float32)
        ],
        full_graph=True)

    # Save in static graph model.
    paddle.jit.save(model, os.path.join(args.output_dir, "inference"))
