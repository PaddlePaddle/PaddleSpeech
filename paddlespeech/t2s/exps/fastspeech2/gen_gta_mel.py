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
# 长度和原本的 mel 不一致怎么办？
import argparse
from pathlib import Path

import numpy as np
import paddle
import yaml
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.preprocess_utils import get_phn_dur
from paddlespeech.t2s.datasets.preprocess_utils import merge_silence
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2
from paddlespeech.t2s.models.fastspeech2 import StyleFastSpeech2Inference
from paddlespeech.t2s.modules.normalizer import ZScore


def evaluate(args, fastspeech2_config):

    # construct dataset for evaluation
    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)

    phone_dict = {}
    for phn, id in phn_id:
        phone_dict[phn] = int(id)

    odim = fastspeech2_config.n_mels
    model = FastSpeech2(
        idim=vocab_size, odim=odim, **fastspeech2_config["model"])

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

    for i, utt_id in enumerate(sentences):
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
        durations = paddle.to_tensor(np.array(durations))
        # 生成的和真实的可能有 1, 2 帧的差距，但是 batch_fn 会修复
        # split data into 3 sections
        if args.dataset == "baker":
            num_train = 9800
            num_dev = 100
        if i in range(0, num_train):
            sub_output_dir = output_dir / ("train/raw")
        elif i in range(num_train, num_train + num_dev):
            sub_output_dir = output_dir / ("dev/raw")
        else:
            sub_output_dir = output_dir / ("test/raw")
        sub_output_dir.mkdir(parents=True, exist_ok=True)
        with paddle.no_grad():
            mel = fastspeech2_inference(phone_ids, durations=durations)
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
    parser.add_argument(
        "--fastspeech2-config", type=str, help="fastspeech2 config file.")
    parser.add_argument(
        "--fastspeech2-checkpoint",
        type=str,
        help="fastspeech2 checkpoint to load.")
    parser.add_argument(
        "--fastspeech2-stat",
        type=str,
        help="mean and standard deviation used to normalize spectrogram when training fastspeech2."
    )

    parser.add_argument(
        "--phones-dict",
        type=str,
        default="phone_id_map.txt",
        help="phone vocabulary file.")

    parser.add_argument(
        "--dur-file", default=None, type=str, help="path to durations.txt.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")
    parser.add_argument("--verbose", type=int, default=1, help="verbose.")

    def str2bool(str):
        return True if str.lower() == 'true' else False

    parser.add_argument(
        "--cut-sil",
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
