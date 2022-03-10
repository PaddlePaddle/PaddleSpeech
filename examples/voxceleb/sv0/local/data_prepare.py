# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle

from paddleaudio.paddleaudio.datasets.voxceleb import VoxCeleb
from paddlespeech.s2t.utils.log import Log
from paddlespeech.vector.io.augment import build_augment_pipeline
from paddlespeech.vector.training.seeding import seed_everything

logger = Log(__name__).getlog()


def main(args):

    # stage0: set the cpu device, all data prepare process will be done in cpu mode
    paddle.set_device("cpu")
    # set the random seed, it is a must for multiprocess training
    seed_everything(args.seed)

    # stage 1: generate the voxceleb csv file
    # Note: this may occurs c++ execption, but the program will execute fine
    # so we ignore the execption 
    # we explicitly pass the vox2 base path to data prepare and generate the audio info
    train_dataset = VoxCeleb(
        'train', target_dir=args.data_dir, vox2_base_path=args.vox2_base_path)
    dev_dataset = VoxCeleb(
        'dev', target_dir=args.data_dir, vox2_base_path=args.vox2_base_path)

    # stage 2: generate the augment noise csv file
    if args.augment:
        augment_pipeline = build_augment_pipeline(target_dir=args.data_dir)


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--seed",
                        default=0,
                        type=int,
                        help="random seed for paddle, numpy and python random package")
    parser.add_argument("--data-dir",
                        default="./data/",
                        type=str,
                        help="data directory")
    parser.add_argument("--vox2-base-path",
                        default=None,
                        type=str,
                        help="vox2 base path, where is store the wav audio")
    parser.add_argument("--augment",
                        action="store_true",
                        default=False,
                        help="Apply audio augments.")
    args = parser.parse_args()
    # yapf: enable
    main(args)
