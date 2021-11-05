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
import logging
import os
from pathlib import Path

import jsonlines
import numpy as np
import paddle
import soundfile as sf
import yaml
from paddle import jit
from paddle.static import InputSpec
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.models.parallel_wavegan import PWGGenerator
from paddlespeech.t2s.models.parallel_wavegan import PWGInference
from paddlespeech.t2s.models.speedyspeech import SpeedySpeech
from paddlespeech.t2s.models.speedyspeech import SpeedySpeechInference
from paddlespeech.t2s.modules.normalizer import ZScore


def evaluate(args, speedyspeech_config, pwg_config):
    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True

    # construct dataset for evaluation
    with jsonlines.open(args.test_metadata, 'r') as reader:
        test_metadata = list(reader)
    test_dataset = DataTable(
        data=test_metadata, fields=["utt_id", "phones", "tones"])

    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)
    with open(args.tones_dict, "r") as f:
        tone_id = [line.strip().split() for line in f.readlines()]
    tone_size = len(tone_id)
    print("tone_size:", tone_size)

    model = SpeedySpeech(
        vocab_size=vocab_size,
        tone_size=tone_size,
        **speedyspeech_config["model"])
    model.set_state_dict(
        paddle.load(args.speedyspeech_checkpoint)["main_params"])
    model.eval()

    vocoder = PWGGenerator(**pwg_config["generator_params"])
    vocoder.set_state_dict(paddle.load(args.pwg_checkpoint)["generator_params"])
    vocoder.remove_weight_norm()
    vocoder.eval()
    print("model done!")

    stat = np.load(args.speedyspeech_stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    speedyspeech_normalizer = ZScore(mu, std)
    speedyspeech_normalizer.eval()

    stat = np.load(args.pwg_stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    pwg_normalizer = ZScore(mu, std)
    pwg_normalizer.eval()

    speedyspeech_inference = SpeedySpeechInference(speedyspeech_normalizer,
                                                   model)
    speedyspeech_inference.eval()
    speedyspeech_inference = jit.to_static(
        speedyspeech_inference,
        input_spec=[
            InputSpec([-1], dtype=paddle.int64), InputSpec(
                [-1], dtype=paddle.int64)
        ])
    paddle.jit.save(speedyspeech_inference,
                    os.path.join(args.inference_dir, "speedyspeech"))
    speedyspeech_inference = paddle.jit.load(
        os.path.join(args.inference_dir, "speedyspeech"))

    pwg_inference = PWGInference(pwg_normalizer, vocoder)
    pwg_inference.eval()
    pwg_inference = jit.to_static(
        pwg_inference, input_spec=[
            InputSpec([-1, 80], dtype=paddle.float32),
        ])
    paddle.jit.save(pwg_inference, os.path.join(args.inference_dir, "pwg"))
    pwg_inference = paddle.jit.load(os.path.join(args.inference_dir, "pwg"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for datum in test_dataset:
        utt_id = datum["utt_id"]
        phones = paddle.to_tensor(datum["phones"])
        tones = paddle.to_tensor(datum["tones"])

        with paddle.no_grad():
            wav = pwg_inference(speedyspeech_inference(phones, tones))
        sf.write(
            output_dir / (utt_id + ".wav"),
            wav.numpy(),
            samplerate=speedyspeech_config.fs)
        print(f"{utt_id} done!")


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(
        description="Synthesize with speedyspeech & parallel wavegan.")
    parser.add_argument(
        "--speedyspeech-config", type=str, help="config file for speedyspeech.")
    parser.add_argument(
        "--speedyspeech-checkpoint",
        type=str,
        help="speedyspeech checkpoint to load.")
    parser.add_argument(
        "--speedyspeech-stat",
        type=str,
        help="mean and standard deviation used to normalize spectrogram when training speedyspeech."
    )
    parser.add_argument(
        "--pwg-config", type=str, help="config file for parallelwavegan.")
    parser.add_argument(
        "--pwg-checkpoint",
        type=str,
        help="parallel wavegan generator parameters to load.")
    parser.add_argument(
        "--pwg-stat",
        type=str,
        help="mean and standard deviation used to normalize spectrogram when training speedyspeech."
    )
    parser.add_argument(
        "--phones-dict", type=str, default=None, help="phone vocabulary file.")
    parser.add_argument(
        "--tones-dict", type=str, default=None, help="tone vocabulary file.")
    parser.add_argument("--test-metadata", type=str, help="test metadata")
    parser.add_argument("--output-dir", type=str, help="output dir")
    parser.add_argument(
        "--inference-dir", type=str, help="dir to save inference models")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")
    parser.add_argument("--verbose", type=int, default=1, help="verbose")

    args, _ = parser.parse_known_args()

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    with open(args.speedyspeech_config) as f:
        speedyspeech_config = CfgNode(yaml.safe_load(f))
    with open(args.pwg_config) as f:
        pwg_config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(speedyspeech_config)
    print(pwg_config)

    evaluate(args, speedyspeech_config, pwg_config)


if __name__ == "__main__":
    main()
