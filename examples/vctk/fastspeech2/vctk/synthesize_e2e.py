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
from pathlib import Path

import numpy as np
import paddle
import soundfile as sf
import yaml
from parakeet.frontend import English
from parakeet.models.fastspeech2 import FastSpeech2
from parakeet.models.fastspeech2 import FastSpeech2Inference
from parakeet.models.parallel_wavegan import PWGGenerator
from parakeet.models.parallel_wavegan import PWGInference
from parakeet.modules.normalizer import ZScore
from yacs.config import CfgNode


def evaluate(args, fastspeech2_config, pwg_config):
    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True

    # construct dataset for evaluation
    sentences = []
    with open(args.text, 'rt') as f:
        for line in f:
            line_list = line.strip().split()
            utt_id = line_list[0]
            sentence = " ".join(line_list[1:])
            sentences.append((utt_id, sentence))

    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    phone_id_map = {}
    for phn, id in phn_id:
        phone_id_map[phn] = int(id)
    print("vocab_size:", vocab_size)
    with open(args.speaker_dict, 'rt') as f:
        spk_id = [line.strip().split() for line in f.readlines()]
    num_speakers = len(spk_id)
    print("num_speakers:", num_speakers)

    odim = fastspeech2_config.n_mels
    model = FastSpeech2(
        idim=vocab_size,
        odim=odim,
        num_speakers=num_speakers,
        **fastspeech2_config["model"])

    model.set_state_dict(
        paddle.load(args.fastspeech2_checkpoint)["main_params"])
    model.eval()

    vocoder = PWGGenerator(**pwg_config["generator_params"])
    vocoder.set_state_dict(paddle.load(args.pwg_checkpoint)["generator_params"])
    vocoder.remove_weight_norm()
    vocoder.eval()
    print("model done!")

    frontend = English()
    punc = "：，；。？！“”‘’':,;.?!"
    print("frontend done!")

    stat = np.load(args.fastspeech2_stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    fastspeech2_normalizer = ZScore(mu, std)

    stat = np.load(args.pwg_stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    pwg_normalizer = ZScore(mu, std)

    fastspeech2_inference = FastSpeech2Inference(fastspeech2_normalizer, model)
    pwg_inference = PWGInference(pwg_normalizer, vocoder)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # only test the number 0 speaker
    spk_id = 0
    for utt_id, sentence in sentences:
        phones = frontend.phoneticize(sentence)
        # remove start_symbol and end_symbol
        phones = phones[1:-1]
        phones = [phn for phn in phones if not phn.isspace()]
        phones = [
            phn if (phn in phone_id_map and phn not in punc) else "sp"
            for phn in phones
        ]
        phone_ids = [phone_id_map[phn] for phn in phones]
        phone_ids = paddle.to_tensor(phone_ids)

        with paddle.no_grad():
            mel = fastspeech2_inference(
                phone_ids, spk_id=paddle.to_tensor(spk_id))
            wav = pwg_inference(mel)

        sf.write(
            str(output_dir / (str(spk_id) + "_" + utt_id + ".wav")),
            wav.numpy(),
            samplerate=fastspeech2_config.fs)
        print(f"{spk_id}_{utt_id} done!")


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(
        description="Synthesize with fastspeech2 & parallel wavegan.")
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
        "--pwg-config", type=str, help="parallel wavegan config file.")
    parser.add_argument(
        "--pwg-checkpoint",
        type=str,
        help="parallel wavegan generator parameters to load.")
    parser.add_argument(
        "--pwg-stat",
        type=str,
        help="mean and standard deviation used to normalize spectrogram when training parallel wavegan."
    )
    parser.add_argument(
        "--phones-dict", type=str, default=None, help="phone vocabulary file.")
    parser.add_argument(
        "--speaker-dict", type=str, default=None, help="speaker id map file.")
    parser.add_argument(
        "--text",
        type=str,
        help="text to synthesize, a 'utt_id sentence' pair per line.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--device", type=str, default="gpu", help="device type to use.")
    parser.add_argument("--verbose", type=int, default=1, help="verbose.")

    args = parser.parse_args()

    paddle.set_device(args.device)

    with open(args.fastspeech2_config) as f:
        fastspeech2_config = CfgNode(yaml.safe_load(f))
    with open(args.pwg_config) as f:
        pwg_config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(fastspeech2_config)
    print(pwg_config)

    evaluate(args, fastspeech2_config, pwg_config)


if __name__ == "__main__":
    main()
