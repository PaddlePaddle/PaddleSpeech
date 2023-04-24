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

from paddlespeech.t2s.datasets.preprocess_utils import get_sentences_svs
from paddlespeech.t2s.models.diffsinger import DiffSinger
from paddlespeech.t2s.models.diffsinger import DiffSingerInference
from paddlespeech.t2s.modules.normalizer import ZScore
from paddlespeech.t2s.utils import str2bool


def evaluate(args, diffsinger_config):
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

    with open(args.diffsinger_stretch, "r") as f:
        spec_min = np.load(args.diffsinger_stretch)[0]
        spec_max = np.load(args.diffsinger_stretch)[1]
        spec_min = paddle.to_tensor(spec_min)
        spec_max = paddle.to_tensor(spec_max)
    print("min and max spec done!")

    odim = diffsinger_config.n_mels
    diffsinger_config["model"]["fastspeech2_params"]["spk_num"] = spk_num
    model = DiffSinger(
        spec_min=spec_min,
        spec_max=spec_max,
        idim=vocab_size,
        odim=odim,
        **diffsinger_config["model"], )

    model.set_state_dict(paddle.load(args.diffsinger_checkpoint)["main_params"])
    model.eval()

    stat = np.load(args.diffsinger_stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    diffsinger_normalizer = ZScore(mu, std)

    diffsinger_inference = DiffSingerInference(diffsinger_normalizer, model)
    diffsinger_inference.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences, speaker_set = get_sentences_svs(
        args.dur_file,
        dataset=args.dataset,
        sample_rate=diffsinger_config.fs,
        n_shift=diffsinger_config.n_shift, )

    if args.dataset == "opencpop":
        wavdir = rootdir / "wavs"
        # split data into 3 sections
        train_file = rootdir / "train.txt"
        train_wav_files = []
        with open(train_file, "r") as f_train:
            for line in f_train.readlines():
                utt = line.split("|")[0]
                wav_name = utt + ".wav"
                wav_path = wavdir / wav_name
                train_wav_files.append(wav_path)

        test_file = rootdir / "test.txt"
        dev_wav_files = []
        test_wav_files = []
        num_dev = 106
        count = 0
        with open(test_file, "r") as f_test:
            for line in f_test.readlines():
                count += 1
                utt = line.split("|")[0]
                wav_name = utt + ".wav"
                wav_path = wavdir / wav_name
                if count > num_dev:
                    test_wav_files.append(wav_path)
                else:
                    dev_wav_files.append(wav_path)
    else:
        print("dataset should in {opencpop} now!")

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
        note = sentences[utt_id][2]
        note_dur = sentences[utt_id][3]
        is_slur = sentences[utt_id][4]
        speaker = sentences[utt_id][-1]

        phone_ids = [phone_dict[phn] for phn in phones]
        phone_ids = paddle.to_tensor(np.array(phone_ids))

        if args.speaker_dict:
            speaker_id = int(
                [item[1] for item in spk_id_list if speaker == item[0]][0])
            speaker_id = paddle.to_tensor(speaker_id)
        else:
            speaker_id = None

        durations = paddle.to_tensor(np.array(durations))
        note = paddle.to_tensor(np.array(note))
        note_dur = paddle.to_tensor(np.array(note_dur))
        is_slur = paddle.to_tensor(np.array(is_slur))
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
            mel = diffsinger_inference(
                text=phone_ids,
                note=note,
                note_dur=note_dur,
                is_slur=is_slur,
                get_mel_fs2=False)
        np.save(sub_output_dir / (utt_id + "_feats.npy"), mel)


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(
        description="Generate mel with diffsinger.")
    parser.add_argument(
        "--dataset",
        default="opencpop",
        type=str,
        help="name of dataset, should in {opencpop} now")
    parser.add_argument(
        "--rootdir", default=None, type=str, help="directory to dataset.")
    parser.add_argument(
        "--diffsinger-config", type=str, help="diffsinger config file.")
    parser.add_argument(
        "--diffsinger-checkpoint",
        type=str,
        help="diffsinger checkpoint to load.")
    parser.add_argument(
        "--diffsinger-stat",
        type=str,
        help="mean and standard deviation used to normalize spectrogram when training diffsinger."
    )
    parser.add_argument(
        "--diffsinger-stretch",
        type=str,
        help="min and max mel used to stretch before training diffusion.")

    parser.add_argument(
        "--phones-dict",
        type=str,
        default="phone_id_map.txt",
        help="phone vocabulary file.")

    parser.add_argument(
        "--speaker-dict", type=str, default=None, help="speaker id map file.")

    parser.add_argument(
        "--dur-file", default=None, type=str, help="path to durations.txt.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")

    args = parser.parse_args()

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    with open(args.diffsinger_config) as f:
        diffsinger_config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(diffsinger_config)

    evaluate(args, diffsinger_config)


if __name__ == "__main__":
    main()
