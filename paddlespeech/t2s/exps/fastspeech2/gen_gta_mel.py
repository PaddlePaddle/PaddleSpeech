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
# generate mels using durations.txt
# for mb melgan finetune
import argparse
import os
from pathlib import Path

import numpy as np
import paddle
import yaml
from tqdm import tqdm
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.preprocess_utils import get_phn_dur
from paddlespeech.t2s.datasets.preprocess_utils import merge_silence
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2
from paddlespeech.t2s.models.fastspeech2 import StyleFastSpeech2Inference
from paddlespeech.t2s.modules.normalizer import ZScore
from paddlespeech.t2s.utils import str2bool


def evaluate(args, fastspeech2_config):
    rootdir = Path(args.rootdir).expanduser()
    assert rootdir.is_dir()

    # construct dataset for evaluation
    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)

    phone_dict = {}
    for phn, id in phn_id:
        phone_dict[phn] = int(id)

    if args.speaker_dict:
        with open(args.speaker_dict, 'rt') as f:
            spk_id_list = [line.strip().split() for line in f.readlines()]
            spk_num = len(spk_id_list)
    else:
        spk_num = None

    odim = fastspeech2_config.n_mels
    model = FastSpeech2(idim=vocab_size,
                        odim=odim,
                        **fastspeech2_config["model"],
                        spk_num=spk_num)

    model.set_state_dict(
        paddle.load(args.fastspeech2_checkpoint)["main_params"])
    model.eval()

    stat = np.load(args.fastspeech2_stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    fastspeech2_normalizer = ZScore(mu, std)

    fastspeech2_inference = StyleFastSpeech2Inference(fastspeech2_normalizer,
                                                      model)
    fastspeech2_inference.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences, speaker_set = get_phn_dur(args.dur_file)
    merge_silence(sentences)

    if args.dataset == "baker":
        wav_files = sorted(list((rootdir / "Wave").rglob("*.wav")))
        # split data into 3 sections
        num_train = 9800
        num_dev = 100
        train_wav_files = wav_files[:num_train]
        dev_wav_files = wav_files[num_train:num_train + num_dev]
        test_wav_files = wav_files[num_train + num_dev:]
    elif args.dataset == "aishell3":
        sub_num_dev = 5
        wav_dir = rootdir / "train" / "wav"
        train_wav_files = []
        dev_wav_files = []
        test_wav_files = []
        for speaker in os.listdir(wav_dir):
            wav_files = sorted(list((wav_dir / speaker).rglob("*.wav")))
            if len(wav_files) > 100:
                train_wav_files += wav_files[:-sub_num_dev * 2]
                dev_wav_files += wav_files[-sub_num_dev * 2:-sub_num_dev]
                test_wav_files += wav_files[-sub_num_dev:]
            else:
                train_wav_files += wav_files

    train_wav_files = [
        os.path.basename(str(str_path)) for str_path in train_wav_files
    ]
    dev_wav_files = [
        os.path.basename(str(str_path)) for str_path in dev_wav_files
    ]
    test_wav_files = [
        os.path.basename(str(str_path)) for str_path in test_wav_files
    ]

    for i, utt_id in enumerate(tqdm(sentences)):
        phones = sentences[utt_id][0]
        durations = sentences[utt_id][1]
        speaker = sentences[utt_id][2]
        # 裁剪掉开头和结尾的 sil
        if args.cut_sil:
            if phones[0] == "sil" and len(durations) > 1:
                durations = durations[1:]
                phones = phones[1:]
            if phones[-1] == 'sil' and len(durations) > 1:
                durations = durations[:-1]
                phones = phones[:-1]
            # sentences[utt_id][0] = phones
            # sentences[utt_id][1] = durations

        phone_ids = [phone_dict[phn] for phn in phones]
        phone_ids = paddle.to_tensor(np.array(phone_ids))

        if args.speaker_dict:
            speaker_id = int(
                [item[1] for item in spk_id_list if speaker == item[0]][0])
            speaker_id = paddle.to_tensor(speaker_id)
        else:
            speaker_id = None

        durations = paddle.to_tensor(np.array(durations))
        # 生成的和真实的可能有 1, 2 帧的差距，但是 batch_fn 会修复
        # split data into 3 sections

        wav_path = utt_id + ".wav"

        if wav_path in train_wav_files:
            sub_output_dir = output_dir / ("train/raw")
        elif wav_path in dev_wav_files:
            sub_output_dir = output_dir / ("dev/raw")
        elif wav_path in test_wav_files:
            sub_output_dir = output_dir / ("test/raw")

        sub_output_dir.mkdir(parents=True, exist_ok=True)

        with paddle.no_grad():
            mel = fastspeech2_inference(phone_ids,
                                        durations=durations,
                                        spk_id=speaker_id)
        np.save(sub_output_dir / (utt_id + "_feats.npy"), mel)


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(
        description="Synthesize with fastspeech2 & parallel wavegan.")
    parser.add_argument(
        "--dataset",
        default="baker",
        type=str,
        help="name of dataset, should in {baker, ljspeech, vctk} now")
    parser.add_argument("--rootdir",
                        default=None,
                        type=str,
                        help="directory to dataset.")
    parser.add_argument("--fastspeech2-config",
                        type=str,
                        help="fastspeech2 config file.")
    parser.add_argument("--fastspeech2-checkpoint",
                        type=str,
                        help="fastspeech2 checkpoint to load.")
    parser.add_argument(
        "--fastspeech2-stat",
        type=str,
        help=
        "mean and standard deviation used to normalize spectrogram when training fastspeech2."
    )

    parser.add_argument("--phones-dict",
                        type=str,
                        default="phone_id_map.txt",
                        help="phone vocabulary file.")

    parser.add_argument("--speaker-dict",
                        type=str,
                        default=None,
                        help="speaker id map file.")

    parser.add_argument("--dur-file",
                        default=None,
                        type=str,
                        help="path to durations.txt.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument("--ngpu",
                        type=int,
                        default=1,
                        help="if ngpu == 0, use cpu.")

    parser.add_argument("--cut-sil",
                        type=str2bool,
                        default=True,
                        help="whether cut sil in the edge of audio")

    args = parser.parse_args()

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    with open(args.fastspeech2_config) as f:
        fastspeech2_config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(fastspeech2_config)

    evaluate(args, fastspeech2_config)


if __name__ == "__main__":
    main()
