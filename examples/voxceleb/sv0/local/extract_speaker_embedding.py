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
import ast
import os

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.io import BatchSampler
from paddle.io import DataLoader
from tqdm import tqdm

from paddleaudio.datasets.voxceleb import VoxCeleb1
from paddleaudio.features.core import melspectrogram
from paddleaudio.backends import load as load_audio
from paddlespeech.vector.io.batch import feature_normalize
from paddlespeech.s2t.utils.log import Log
from paddlespeech.vector.models.ecapa_tdnn import EcapaTdnn
from paddlespeech.vector.modules.sid_model import SpeakerIdetification
from paddlespeech.vector.training.metrics import compute_eer
from paddlespeech.vector.training.seeding import seed_everything

logger = Log(__name__).getlog()

# feat configuration
cpu_feat_conf = {
    'n_mels': 80,
    'window_size': 400,  #ms
    'hop_length': 160,  #ms
}

def extract_audio_embedding(args):
    # stage 0: set the training device, cpu or gpu
    paddle.set_device(args.device)
    # set the random seed, it is a must for multiprocess training
    seed_everything(args.seed)

    # stage 1: build the dnn backbone model network
    ##"channels": [1024, 1024, 1024, 1024, 3072],
    model_conf = {
        "input_size": 80,
        "channels": [512, 512, 512, 512, 1536],
        "kernel_sizes": [5, 3, 3, 3, 1],
        "dilations": [1, 2, 3, 4, 1],
        "attention_channels": 128,
        "lin_neurons": 192,
    }
    ecapa_tdnn = EcapaTdnn(**model_conf)

    # stage 2: load the pre-trained model
    args.load_checkpoint = os.path.abspath(
        os.path.expanduser(args.load_checkpoint))

    # load model checkpoint to sid model
    state_dict = paddle.load(
        os.path.join(args.load_checkpoint, 'model.pdparams'))
    model.set_state_dict(state_dict)
    logger.info(f'Checkpoint loaded from {args.load_checkpoint}')

    # stage 3: we must set the model to eval mode
    model.eval()
    
    # stage 4: read the audio data and extract the embedding
    waveform, sr = load_audio(args.audio_path)
    feat = melspectrogram(x=waveform, **cpu_feat_conf)
    feat = paddle.to_tensor(feat).unsqueeze(0)
    lengths = paddle.ones([1]) # in paddle inference model, the lengths is all one without padding
    feat = feature_normalize(feat, mean_norm=True, std_norm=False)
    embedding = ecapa_tdnn(feat, lengths
                ).squeeze().numpy() # (1, emb_size, 1) -> (emb_size)

    # stage 5: do global norm with external mean and std
    # todo
    return embedding


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device',
                        choices=['cpu', 'gpu'],
                        default="gpu",
                        help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--seed",
                        default=0,
                        type=int,
                        help="random seed for paddle, numpy and python random package")
    parser.add_argument("--load-checkpoint",
                        type=str,
                        default='',
                        help="Directory to load model checkpoint to contiune trainning.")
    parser.add_argument("--global-embedding-norm",
                        type=str,
                        default=None,
                        help="Apply global normalization on speaker embeddings.")
    parser.add_argument("--audio-path",
                        default="./data/demo.wav",
                        type=str,
                        help="Single audio file path")
    args = parser.parse_args()
    # yapf: enable

    extract_audio_embedding(args)
