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
import os
from pathlib import Path

import jsonlines
import numpy as np
import paddle
import soundfile as sf
import yaml
from paddle import distributed as dist
from timer import timer
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.models.melgan import MelGANGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize with multi band melgan.")
    parser.add_argument(
        "--config", type=str, help="multi band melgan config file.")
    parser.add_argument("--checkpoint", type=str, help="snapshot to load.")
    parser.add_argument("--test-metadata", type=str, help="dev data.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")
    parser.add_argument("--verbose", type=int, default=1, help="verbose.")

    args = parser.parse_args()

    with open(args.config) as f:
        config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(config)
    print(
        f"master see the word size: {dist.get_world_size()}, from pid: {os.getpid()}"
    )

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")
    generator = MelGANGenerator(**config["generator_params"])
    state_dict = paddle.load(args.checkpoint)
    generator.set_state_dict(state_dict["generator_params"])

    generator.remove_weight_norm()
    generator.eval()
    with jsonlines.open(args.test_metadata, 'r') as reader:
        metadata = list(reader)

    test_dataset = DataTable(
        metadata,
        fields=['utt_id', 'feats'],
        converters={
            'utt_id': None,
            'feats': np.load,
        })
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N = 0
    T = 0
    for example in test_dataset:
        utt_id = example['utt_id']
        mel = example['feats']
        mel = paddle.to_tensor(mel)  # (T, C)
        with timer() as t:
            with paddle.no_grad():
                wav = generator.inference(c=mel)
            wav = wav.numpy()
            N += wav.size
            T += t.elapse
            speed = wav.size / t.elapse
            rtf = config.fs / speed
        print(
            f"{utt_id}, mel: {mel.shape}, wave: {wav.shape}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )
        sf.write(str(output_dir / (utt_id + ".wav")), wav, samplerate=config.fs)
    print(f"generation speed: {N / T}Hz, RTF: {config.fs / (N / T) }")


if __name__ == "__main__":
    main()
