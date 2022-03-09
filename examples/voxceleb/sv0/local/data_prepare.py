import argparse
import os

import numpy as np
import paddle
from paddle.io import BatchSampler
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler

from paddleaudio.datasets.voxceleb import VoxCeleb1
from paddleaudio.features.core import melspectrogram
from paddlespeech.s2t.utils.log import Log
from paddlespeech.vector.io.augment import build_augment_pipeline
from paddlespeech.vector.io.augment import waveform_augment
from paddlespeech.vector.io.batch import feature_normalize
from paddlespeech.vector.io.batch import waveform_collate_fn
from paddlespeech.vector.models.ecapa_tdnn import EcapaTdnn
from paddlespeech.vector.modules.loss import AdditiveAngularMargin
from paddlespeech.vector.modules.loss import LogSoftmaxWrapper
from paddlespeech.vector.modules.lr import CyclicLRScheduler
from paddlespeech.vector.modules.sid_model import SpeakerIdetification
from paddlespeech.vector.training.seeding import seed_everything
from paddlespeech.vector.utils.time import Timer

logger = Log(__name__).getlog()

def main(args):
    # stage0: set the cpu device, all data prepare process will be done in cpu mode
    paddle.set_device("cpu")
    # set the random seed, it is a must for multiprocess training
    seed_everything(args.seed)

    # stage 1: generate the voxceleb csv file
    # Note: this may occurs c++ execption, but the program will execute fine
    # so we can ignore the execption 
    train_dataset = VoxCeleb1('train', target_dir=args.data_dir)
    dev_dataset = VoxCeleb1('dev', target_dir=args.data_dir)

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
    parser.add_argument("--augment",
                        action="store_true",
                        default=False,
                        help="Apply audio augments.")
    args = parser.parse_args()
    # yapf: enable
    main(args)                    