#!/usr/bin/python3
#! coding:utf-8
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

"""
Compute the utterance feature according to the config.

Feature type and augmentation defined in the yaml config
"""


import argparse
import yaml
from paddlespeech.s2t.utils.log import Log 
from yacs.config import CfgNode
import json
from paddleaudio.features.audiopipeline import AudioPipeline
from paddlespeech.s2t.frontend.augmentor.augmentation import AugmentationPipeline
from paddlespeech.s2t.frontend.featurizer.audio_featurizer import AudioFeaturizer
from paddleaudio.datasets.dataset import SpeechDataset
import paddle
logger = Log(__name__).getlog()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
        "--config",
        default="./config/config.yaml",
        type=str,
        help="feature config file"
)
parser.add_argument(
    "--dataset",
    default="./data/train.json",
    type=str,
    help="audio dataset"
)

parser.add_argument(
    "--feat",
    default="./data/train.feat",
    type=str,
    help="feat store file")

args = parser.parse_args()

def main():
    config = CfgNode(new_allowed=True)
    print("config: {}".format(args.config))
    config.merge_from_file(args.config)
    audio_pipeline = AudioPipeline(config)

    with open(args.dataset, 'r') as f:
        audio_dataset = json.load(f)

    utts = list(audio_dataset.keys())
    for item in utts:
        # logger.info("process the utt: {}".format(item))
        feat = audio_pipeline.process_utterance(audio_dataset[item])
        feat = paddle.to_tensor(feat) # 转换为paddle的tensor
        audio_dataset[item]["wav"] = feat
    
    # logger.info(audio_dataset)
    paddle.save(audio_dataset, args.feat)
    
if __name__ == "__main__":
    main()