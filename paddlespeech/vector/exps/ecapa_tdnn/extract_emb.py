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
import time

import paddle
from yacs.config import CfgNode

from paddleaudio.backends import load as load_audio
from paddleaudio.compliance.librosa import melspectrogram
from paddlespeech.s2t.utils.log import Log
from paddlespeech.vector.io.batch import feature_normalize
from paddlespeech.vector.models.ecapa_tdnn import EcapaTdnn
from paddlespeech.vector.modules.sid_model import SpeakerIdetification
from paddlespeech.vector.training.seeding import seed_everything

logger = Log(__name__).getlog()

class VectorWrapper:
    """ VectorWrapper extract the audio embedding,
        and single audio will get only an embedding
    """
    def __init__(self,
                 device,
                 config_path,
                 model_path,):
        super(VectorWrapper, self).__init__()
        # stage 0: config the 
        self.device = device
        self.config_path = config_path
        self.model_path = model_path

        # stage 1: set the run host device
        paddle.device.set_device(device)

        # stage 2: read the yaml config and set the seed factor
        self.read_yaml_config(self.config_path)
        seed_everything(self.config.seed)

        # stage 3: init the speaker verification model
        self.init_vector_model(self.config, self.model_path)
        
    def read_yaml_config(self, config_path):
        """Read the yaml config from the config path

        Args:
            config_path (str): yaml config path
        """
        config = CfgNode(new_allowed=True)

        if config_path:
            config.merge_from_file(config_path)

        config.freeze()
        self.config = config

    def init_vector_model(self, config, model_path):
        """Init the vector model from yaml config

        Args:
            config (CfgNode): yaml config 
            model_path (str): pretrained model path and the stored model is named as model.pdparams
        """
        # get the backbone network instance
        ecapa_tdnn = EcapaTdnn(**config.model)
        
        # get the sid instance
        model = SpeakerIdetification(backbone=ecapa_tdnn, num_class=config.num_speakers)

        # read the model parameters to sid model
        model_path = os.path.abspath(os.path.expanduser(model_path))
        state_dict = paddle.load(os.path.join(model_path, "model.pdparams"))
        model.set_state_dict(state_dict)

        model.eval()
        self.model = model

    def extract_audio_embedding(self, audio_path):
        """Extract the audio embedding

        Args:
            audio_path (str): audio path, which will be extracted the embedding

        Returns:
            embedding (numpy.array) : audio embedding
        """
        waveform, sr = load_audio(audio_path)
        feat = melspectrogram(x=waveform,
                sr=self.config.sr,
                n_mels=self.config.n_mels,
                window_size=self.config.window_size,
                hop_length=self.config.hop_size)
        # conver the audio feat to batch shape, which means batch_size is equal to one
        feat = paddle.to_tensor(feat).unsqueeze(0)

        # in inference period, the lengths is all one without padding
        lengths = paddle.ones([1])
        feat = feature_normalize(feat, mean_norm=True, std_norm=False)
        
        # model backbone network forward the feats and get the embedding
        embedding = self.model.backbone(feat, lengths).squeeze().numpy()  # (1, emb_size, 1) -> (emb_size)

        return embedding

def extract_audio_embedding(args, config):
    # stage 0: set the training device, cpu or gpu
    paddle.set_device(args.device)
    # set the random seed, it is a must for multiprocess training
    seed_everything(config.seed)

    # stage 1: build the dnn backbone model network
    ecapa_tdnn = EcapaTdnn(**config.model)

    # stage4: build the speaker verification train instance with backbone model
    model = SpeakerIdetification(
        backbone=ecapa_tdnn, num_class=config.num_speakers)
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
    # wavform is one dimension numpy array 
    waveform, sr = load_audio(args.audio_path)

    # feat type is numpy array, whose shape is [dim, time]
    # we need convert the audio feat to one-batch shape [batch, dim, time], where the batch is one
    # so the final shape is [1, dim, time]
    start_time = time.time()
    feat = melspectrogram(
        x=waveform,
        sr=config.sr,
        n_mels=config.n_mels,
        window_size=config.window_size,
        hop_length=config.hop_size)
    feat = paddle.to_tensor(feat).unsqueeze(0)

    # in inference period, the lengths is all one without padding
    lengths = paddle.ones([1])
    feat = feature_normalize(feat, mean_norm=True, std_norm=False)

    # model backbone network forward the feats and get the embedding
    embedding = model.backbone(
        feat, lengths).squeeze().numpy()  # (1, emb_size, 1) -> (emb_size)
    elapsed_time = time.time() - start_time
    audio_length = waveform.shape[0] / sr

    # stage 5: do global norm with external mean and std
    rtf = elapsed_time / audio_length
    logger.info(f"{args.device} rft={rtf}")
    paddle.save(embedding, "emb1")
    return embedding


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device',
                        choices=['cpu', 'gpu'],
                        default="gpu",
                        help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--config",
                        default=None,
                        type=str,
                        help="configuration file")
    parser.add_argument("--load-checkpoint",
                        type=str,
                        default='',
                        help="Directory to load model checkpoint to contiune trainning.")
    parser.add_argument("--audio-path",
                        default="./data/demo.wav",
                        type=str,
                        help="Single audio file path")
    args = parser.parse_args()
    # yapf: enable
    # https://yaml.org/type/float.html
    config = CfgNode(new_allowed=True)
    if args.config:
        config.merge_from_file(args.config)

    config.freeze()
    print(config)

    extract_audio_embedding(args, config)

    # use the VectorWrapper to extract the audio embedding
    vector_inst = VectorWrapper(device="gpu", 
                    config_path=args.config, 
                    model_path=args.load_checkpoint)
    
    embedding = vector_inst.extract_audio_embedding(args.audio_path)
