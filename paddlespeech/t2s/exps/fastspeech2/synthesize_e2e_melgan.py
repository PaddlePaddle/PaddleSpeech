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

import numpy as np
import paddle
import soundfile as sf
import yaml
from paddle import jit
from paddle.static import InputSpec
from yacs.config import CfgNode

from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2Inference
from paddlespeech.t2s.models.melgan import MelGANGenerator
from paddlespeech.t2s.models.melgan import MelGANInference
from paddlespeech.t2s.modules.normalizer import ZScore


def evaluate(args, fastspeech2_config, melgan_config):
    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True

    # construct dataset for evaluation
    sentences = []
    with open(args.text, 'rt') as f:
        for line in f:
            items = line.strip().split()
            utt_id = items[0]
            sentence = "".join(items[1:])
            sentences.append((utt_id, sentence))

    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)
    odim = fastspeech2_config.n_mels
    model = FastSpeech2(
        idim=vocab_size, odim=odim, **fastspeech2_config["model"])

    model.set_state_dict(
        paddle.load(args.fastspeech2_checkpoint)["main_params"])
    model.eval()

    vocoder = MelGANGenerator(**melgan_config["generator_params"])
    vocoder.set_state_dict(
        paddle.load(args.melgan_checkpoint)["generator_params"])
    vocoder.remove_weight_norm()
    vocoder.eval()
    print("model done!")

    frontend = Frontend(phone_vocab_path=args.phones_dict)
    print("frontend done!")

    stat = np.load(args.fastspeech2_stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    fastspeech2_normalizer = ZScore(mu, std)

    stat = np.load(args.melgan_stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    pwg_normalizer = ZScore(mu, std)

    fastspeech2_inference = FastSpeech2Inference(fastspeech2_normalizer, model)
    fastspeech2_inference.eval()
    fastspeech2_inference = jit.to_static(
        fastspeech2_inference, input_spec=[InputSpec([-1], dtype=paddle.int64)])
    paddle.jit.save(fastspeech2_inference,
                    os.path.join(args.inference_dir, "fastspeech2"))
    fastspeech2_inference = paddle.jit.load(
        os.path.join(args.inference_dir, "fastspeech2"))

    mb_melgan_inference = MelGANInference(pwg_normalizer, vocoder)
    mb_melgan_inference.eval()
    mb_melgan_inference = jit.to_static(
        mb_melgan_inference,
        input_spec=[
            InputSpec([-1, 80], dtype=paddle.float32),
        ])
    paddle.jit.save(mb_melgan_inference,
                    os.path.join(args.inference_dir, "mb_melgan"))
    mb_melgan_inference = paddle.jit.load(
        os.path.join(args.inference_dir, "mb_melgan"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for utt_id, sentence in sentences:
        input_ids = frontend.get_input_ids(sentence, merge_sentences=True)
        phone_ids = input_ids["phone_ids"]
        flags = 0
        for part_phone_ids in phone_ids:
            with paddle.no_grad():
                mel = fastspeech2_inference(part_phone_ids)
                temp_wav = mb_melgan_inference(mel)
            if flags == 0:
                wav = temp_wav
                flags = 1
            else:
                wav = paddle.concat([wav, temp_wav])
        sf.write(
            str(output_dir / (utt_id + ".wav")),
            wav.numpy(),
            samplerate=fastspeech2_config.fs)
        print(f"{utt_id} done!")


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
        "--melgan-config", type=str, help="parallel wavegan config file.")
    parser.add_argument(
        "--melgan-checkpoint",
        type=str,
        help="parallel wavegan generator parameters to load.")
    parser.add_argument(
        "--melgan-stat",
        type=str,
        help="mean and standard deviation used to normalize spectrogram when training parallel wavegan."
    )
    parser.add_argument(
        "--phones-dict",
        type=str,
        default="phone_id_map.txt",
        help="phone vocabulary file.")
    parser.add_argument(
        "--text",
        type=str,
        help="text to synthesize, a 'utt_id sentence' pair per line.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--inference-dir", type=str, help="dir to save inference models")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")
    parser.add_argument("--verbose", type=int, default=1, help="verbose.")

    args = parser.parse_args()

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    with open(args.fastspeech2_config) as f:
        fastspeech2_config = CfgNode(yaml.safe_load(f))
    with open(args.melgan_config) as f:
        melgan_config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(fastspeech2_config)
    print(melgan_config)

    evaluate(args, fastspeech2_config, melgan_config)


if __name__ == "__main__":
    main()
