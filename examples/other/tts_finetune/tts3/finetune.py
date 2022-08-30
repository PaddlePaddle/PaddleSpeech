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
from pathlib import Path
from typing import Union

import yaml
from local.check_oov import get_check_result
from local.extract import extract_feature
from local.label_process import get_single_label
from local.prepare_env import generate_finetune_env
from paddle import distributed as dist
from yacs.config import CfgNode

from paddlespeech.t2s.exps.fastspeech2.train import train_sp
from utils.gen_duration_from_textgrid import gen_duration_from_textgrid

DICT_EN = 'tools/aligner/cmudict-0.7b'
DICT_ZH = 'tools/aligner/simple.lexicon'
MODEL_DIR_EN = 'tools/aligner/vctk_model.zip'
MODEL_DIR_ZH = 'tools/aligner/aishell3_model.zip'
MFA_PHONE_EN = 'tools/aligner/vctk_model/meta.yaml'
MFA_PHONE_ZH = 'tools/aligner/aishell3_model/meta.yaml'
MFA_PATH = 'tools/montreal-forced-aligner/bin'
os.environ['PATH'] = MFA_PATH + '/:' + os.environ['PATH']


class TrainArgs():
    def __init__(self, ngpu, config_file, dump_dir: Path, output_dir: Path):
        self.config = str(config_file)
        self.train_metadata = str(dump_dir / "train/norm/metadata.jsonl")
        self.dev_metadata = str(dump_dir / "dev/norm/metadata.jsonl")
        self.output_dir = str(output_dir)
        self.ngpu = ngpu
        self.phones_dict = str(dump_dir / "phone_id_map.txt")
        self.speaker_dict = str(dump_dir / "speaker_id_map.txt")
        self.voice_cloning = False


def get_mfa_result(
        input_dir: Union[str, Path],
        mfa_dir: Union[str, Path],
        lang: str='en', ):
    """get mfa result

    Args:
        input_dir (Union[str, Path]): input dir including wav file and label
        mfa_dir (Union[str, Path]): mfa result dir
        lang (str, optional): input audio language. Defaults to 'en'.
    """
    # MFA
    if lang == 'en':
        DICT = DICT_EN
        MODEL_DIR = MODEL_DIR_EN

    elif lang == 'zh':
        DICT = DICT_ZH
        MODEL_DIR = MODEL_DIR_ZH
    else:
        print('please input right lang!!')

    CMD = 'mfa_align' + ' ' + str(
        input_dir) + ' ' + DICT + ' ' + MODEL_DIR + ' ' + str(mfa_dir)
    os.system(CMD)


if __name__ == '__main__':
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")

    parser.add_argument(
        "--input_dir",
        type=str,
        default="./input/baker_mini",
        help="directory containing audio and label file")

    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
        default="./pretrained_models/fastspeech2_aishell3_ckpt_1.1.0",
        help="Path to pretrained model")

    parser.add_argument(
        "--mfa_dir",
        type=str,
        default="./mfa_result",
        help="directory to save aligned files")

    parser.add_argument(
        "--dump_dir",
        type=str,
        default="./dump",
        help="directory to save feature files and metadata.")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./exp/default/",
        help="directory to save finetune model.")

    parser.add_argument(
        '--lang',
        type=str,
        default='zh',
        choices=['zh', 'en'],
        help='Choose input audio language. zh or en')

    parser.add_argument(
        "--ngpu", type=int, default=2, help="if ngpu=0, use cpu.")

    parser.add_argument("--epoch", type=int, default=100, help="finetune epoch")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=-1,
        help="batch size, default -1 means same as pretrained model")

    args = parser.parse_args()

    fs = 24000
    n_shift = 300
    input_dir = Path(args.input_dir).expanduser()
    mfa_dir = Path(args.mfa_dir).expanduser()
    mfa_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = Path(args.dump_dir).expanduser()
    dump_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrained_model_dir = Path(args.pretrained_model_dir).expanduser()

    # read config
    config_file = pretrained_model_dir / "default.yaml"
    with open(config_file) as f:
        config = CfgNode(yaml.safe_load(f))
    config.max_epoch = config.max_epoch + args.epoch
    if args.batch_size > 0:
        config.batch_size = args.batch_size

    if args.lang == 'en':
        lexicon_file = DICT_EN
        mfa_phone_file = MFA_PHONE_EN
    elif args.lang == 'zh':
        lexicon_file = DICT_ZH
        mfa_phone_file = MFA_PHONE_ZH
    else:
        print('please input right lang!!')
    am_phone_file = pretrained_model_dir / "phone_id_map.txt"
    label_file = input_dir / "labels.txt"

    #check phone for mfa and am finetune
    oov_words, oov_files, oov_file_words = get_check_result(
        label_file, lexicon_file, mfa_phone_file, am_phone_file)
    input_dir = get_single_label(label_file, oov_files, input_dir)

    # get mfa result
    get_mfa_result(input_dir, mfa_dir, args.lang)

    # # generate durations.txt
    duration_file = "./durations.txt"
    gen_duration_from_textgrid(mfa_dir, duration_file, fs, n_shift)

    # generate phone and speaker map files
    extract_feature(duration_file, config, input_dir, dump_dir,
                    pretrained_model_dir)

    # create finetune env
    generate_finetune_env(output_dir, pretrained_model_dir)

    # create a new args for training
    train_args = TrainArgs(args.ngpu, config_file, dump_dir, output_dir)

    # finetune models
    # dispatch
    if args.ngpu > 1:
        dist.spawn(train_sp, (train_args, config), nprocs=args.ngpu)
    else:
        train_sp(train_args, config)
