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

import librosa
import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.get_feats import LogMelFBank
from paddlespeech.t2s.models.parallel_wavegan import PWGGenerator
from paddlespeech.t2s.models.parallel_wavegan import PWGInference
from paddlespeech.t2s.modules.normalizer import ZScore


def evaluate(args, config):
    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True

    vocoder = PWGGenerator(**config["generator_params"])
    state_dict = paddle.load(args.checkpoint)
    vocoder.set_state_dict(state_dict["generator_params"])
    vocoder.remove_weight_norm()
    vocoder.eval()
    print("model done!")

    stat = np.load(args.stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    normalizer = ZScore(mu, std)

    pwg_inference = PWGInference(normalizer, vocoder)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mel_extractor = LogMelFBank(
        sr=config.fs,
        n_fft=config.n_fft,
        hop_length=config.n_shift,
        win_length=config.win_length,
        window=config.window,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax)

    for utt_name in os.listdir(input_dir):
        wav, _ = librosa.load(str(input_dir / utt_name), sr=config.fs)
        # extract mel feats
        mel = mel_extractor.get_log_mel_fbank(wav)
        mel = paddle.to_tensor(mel)
        with paddle.no_grad():
            gen_wav = pwg_inference(mel)
        sf.write(
            str(output_dir / ("gen_" + utt_name)),
            gen_wav.numpy(),
            samplerate=config.fs)
        print(f"{utt_name} done!")


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(
        description="Synthesize with parallel wavegan.")

    parser.add_argument(
        "--config", type=str, help="parallel wavegan config file.")
    parser.add_argument("--checkpoint", type=str, help="snapshot to load.")
    parser.add_argument(
        "--stat",
        type=str,
        help="mean and standard deviation used to normalize spectrogram when training parallel wavegan."
    )
    parser.add_argument("--input-dir", type=str, help="input dir of wavs.")
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

    with open(args.config) as f:
        config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(config)

    evaluate(args, config)


if __name__ == "__main__":
    main()
