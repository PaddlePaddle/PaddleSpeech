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

import paddle
from paddleaudio.datasets.voxceleb import VoxCeleb
from yacs.config import CfgNode

from paddlespeech.s2t.utils.log import Log
from paddlespeech.vector.io.augment import build_augment_pipeline
from paddlespeech.vector.training.seeding import seed_everything

logger = Log(__name__).getlog()


def main(args, config):

    # stage0: set the cpu device, all data prepare process will be done in cpu mode
    paddle.set_device("cpu")
    # set the random seed, it is a must for multiprocess training
    seed_everything(config.seed)

    # stage 1: generate the voxceleb csv file
    # Note: this may occurs c++ execption, but the program will execute fine
    # so we ignore the execption
    # we explicitly pass the vox2 base path to data prepare and generate the audio info
    logger.info("start to generate the voxceleb dataset info")
    train_dataset = VoxCeleb('train',
                             target_dir=args.data_dir,
                             vox2_base_path=config.vox2_base_path)

    # stage 2: generate the augment noise csv file
    if config.augment:
        logger.info("start to generate the augment dataset info")
        augment_pipeline = build_augment_pipeline(target_dir=args.data_dir)


if __name__ == "__main__":
    # yapf: disable
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data-dir",
                        default="./data/",
                        type=str,
                        help="data directory")
    parser.add_argument("--config",
                        default=None,
                        type=str,
                        help="configuration file")
    args = parser.parse_args()
    # yapf: enable

    # https://yaml.org/type/float.html
    config = CfgNode(new_allowed=True)
    if args.config:
        config.merge_from_file(args.config)

    config.freeze()
    print(config)

    main(args, config)
