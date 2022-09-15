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
import os
from pathlib import Path
from typing import Union
import yaml
from paddle import distributed as dist
from yacs.config import CfgNode
import argparse
from pathlib import Path
import paddle
import soundfile as sf
import yaml
from timer import timer
from yacs.config import CfgNode

from paddlespeech.t2s.exps.fastspeech2.train import train_sp

# from .check_oov import get_check_result
# from .extract import extract_feature
# from .label_process import get_single_label
# from .prepare_env import generate_finetune_env

from check_oov import get_check_result
from extract import extract_feature
from label_process import get_single_label
from prepare_env import generate_finetune_env

from utils.gen_duration_from_textgrid import gen_duration_from_textgrid

DICT_EN = 'source/tools/aligner/cmudict-0.7b'
DICT_EN_v2 = 'source/tools/aligner/cmudict-0.7b.dict'
DICT_ZH = 'source/tools/aligner/simple.lexicon'
DICT_ZH_v2 = 'source/tools/aligner/simple.dict'
MODEL_DIR_EN = 'source/tools/aligner/vctk_model.zip'
MODEL_DIR_ZH = 'source/tools/aligner/aishell3_model.zip'
MFA_PHONE_EN = 'source/tools/aligner/vctk_model/meta.yaml'
MFA_PHONE_ZH = 'source/tools/aligner/aishell3_model/meta.yaml'
MFA_PATH = 'source/tools/montreal-forced-aligner/bin'
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
        lang: str='en', 
        mfa_version='v1'):
    """get mfa result

    Args:
        input_dir (Union[str, Path]): input dir including wav file and label
        mfa_dir (Union[str, Path]): mfa result dir
        lang (str, optional): input audio language. Defaults to 'en'.
    """
    input_dir = str(input_dir).replace("/newdir", "")
    # MFA
    if mfa_version == 'v1':
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
    else:
        if lang == 'en':
            DICT = DICT_EN_v2
            MODEL_DIR = MODEL_DIR_EN

        elif lang == 'zh':
            DICT = DICT_ZH_v2
            MODEL_DIR = MODEL_DIR_ZH
        else:
            print('please input right lang!!')

        CMD = 'mfa align' + ' ' + str(
            input_dir) + ' ' + DICT + ' ' + MODEL_DIR + ' ' + str(mfa_dir)
        os.system(CMD)


def finetune_model(input_dir,
             pretrained_model_dir,
             mfa_dir,
             dump_dir,
             lang,
             output_dir,
             ngpu,
             epoch,
             batch_size,
             mfa_version='v1'):
    fs = 24000
    n_shift = 300
    input_dir = Path(input_dir).expanduser()
    mfa_dir = Path(mfa_dir).expanduser()
    mfa_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = Path(dump_dir).expanduser()
    dump_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrained_model_dir = Path(pretrained_model_dir).expanduser()

    # read config
    config_file = pretrained_model_dir / "default.yaml"
    print("config_path: ")
    print(f"########### { config_file } ###########")
    with open(config_file) as f:
        config = CfgNode(yaml.safe_load(f))
    config.max_epoch = config.max_epoch + epoch
    if batch_size > 0:
        config.batch_size = batch_size

    if lang == 'en':
        lexicon_file = DICT_EN
        mfa_phone_file = MFA_PHONE_EN
    elif lang == 'zh':
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
    print("input_dir: ", input_dir)
    get_mfa_result(input_dir, mfa_dir, lang, mfa_version=mfa_version)

    # # generate durations.txt
    duration_file = "./durations.txt"
    print("mfa_dir: ", mfa_dir)
    gen_duration_from_textgrid(mfa_dir, duration_file, fs, n_shift)

    # generate phone and speaker map files
    extract_feature(duration_file, config, input_dir, dump_dir,
                    pretrained_model_dir)

    # create finetune env
    generate_finetune_env(output_dir, pretrained_model_dir)

    # create a new args for training
    train_args = TrainArgs(ngpu, config_file, dump_dir, output_dir)

    # finetune models
    # dispatch
    if ngpu > 1:
        dist.spawn(train_sp, (train_args, config), nprocs=ngpu)
    else:
        train_sp(train_args, config)
    return output_dir

# 合成



if __name__ == '__main__':
    # parse config and args
    parser = argparse.ArgumentParser(
        description="Preprocess audio and then extract features.")

    parser.add_argument(
        "--input_dir",
        type=str,
        help="directory containing audio and label file")

    parser.add_argument(
        "--pretrained_model_dir",
        type=str,
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
        "--ngpu", type=int, default=1, help="if ngpu=0, use cpu.")

    parser.add_argument("--epoch", type=int, default=100, help="finetune epoch")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=-1,
        help="batch size, default -1 means same as pretrained model")
    
    parser.add_argument(
        "--mfa_version",
        type=str,
        default='v1',
        help="mfa version , you can choose v1 or v2")

    args = parser.parse_args()
    
    finetune_model(input_dir=args.input_dir,
             pretrained_model_dir=args.pretrained_model_dir,
             mfa_dir=args.mfa_dir,
             dump_dir=args.dump_dir,
             lang=args.lang,
             output_dir=args.output_dir,
             ngpu=args.ngpu,
             epoch=args.epoch,
             batch_size=args.batch_size,
             mfa_version=args.mfa_version)
    
    
    # 10 句话 finetune 测试
    # input_dir = "source/wav/finetune/default"
    # pretrained_model_dir = "source/model/fastspeech2_aishell3_ckpt_1.1.0"
    # mfa_dir = "tmp_dir/finetune/mfa"
    # dump_dir = "tmp_dir/finetune/dump"
    # lang = "zh"
    # output_dir = "tmp_dir/finetune/out"
    # ngpu = 0
    # epoch = 2
    # batch_size = 2
    # mfa_version = 'v2'
    # 微调
    # finetune_model(input_dir,
    #          pretrained_model_dir,
    #          mfa_dir,
    #          dump_dir,
    #          lang,
    #          output_dir,
    #          ngpu,
    #          epoch,
    #          batch_size,
    #          mfa_version=mfa_version)
    
    # # 合成测试
    # text = "source/wav/finetune/test.txt"
    
    # lang = "zh"
    # spk_id = 0
    # am = "fastspeech2_aishell3"
    # am_config = f"{pretrained_model_dir}/default.yaml"
    # am_ckpt = f"{output_dir}/checkpoints/snapshot_iter_96408.pdz"
    # am_stat = f"{pretrained_model_dir}/speech_stats.npy"
    # speaker_dict = f"{dump_dir}/speaker_id_map.txt"
    # phones_dict  = f"{dump_dir}/phone_id_map.txt"
    # tones_dict = None
    # voc = "hifigan_aishell3"
    # voc_config = "source/model/hifigan_aishell3_ckpt_0.2.0/default.yaml"
    # voc_ckpt = "source/model/hifigan_aishell3_ckpt_0.2.0/snapshot_iter_2500000.pdz"
    # voc_stat = "source/model/hifigan_aishell3_ckpt_0.2.0/feats_stats.npy"
    
    # wav_output_dir = "source/wav/finetune/out"
    
    # synthesize(text,
    #            wav_output_dir,
    #            lang,
    #            spk_id,
    #            am,
    #            am_config,
    #            am_ckpt,
    #            am_stat,
    #            speaker_dict,
    #            phones_dict,
    #            tones_dict,
    #            voc,
    #            voc_config,
    #            voc_ckpt,
    #            voc_stat
    #            )
