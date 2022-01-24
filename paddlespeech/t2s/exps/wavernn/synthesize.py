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

import numpy as np
import paddle
import soundfile as sf
import yaml
from paddle import distributed as dist
from yacs.config import CfgNode

from paddlespeech.t2s.models.wavernn import WaveRNN


def main():
    parser = argparse.ArgumentParser(description="Synthesize with WaveRNN.")

    parser.add_argument("--config", type=str, help="GANVocoder config file.")
    parser.add_argument("--checkpoint", type=str, help="snapshot to load.")
    parser.add_argument(
        "--input",
        type=str,
        help="path of directory containing mel spectrogram (in .npy format)")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")

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

    model = WaveRNN(
        hop_length=config.n_shift, sample_rate=config.fs, **config["model"])
    state_dict = paddle.load(args.checkpoint)
    model.set_state_dict(state_dict["main_params"])

    model.eval()

    mel_dir = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_path in sorted(mel_dir.iterdir()):
        mel = np.load(str(file_path))
        mel = paddle.to_tensor(mel)
        mel = mel.transpose([1, 0])
        # input shape is (T', C_aux)
        audio = model.generate(
            c=mel,
            batched=config.inference.gen_batched,
            target=config.inference.target,
            overlap=config.inference.overlap,
            mu_law=config.mu_law,
            gen_display=True)
        audio_path = output_dir / (os.path.splitext(file_path.name)[0] + ".wav")
        sf.write(audio_path, audio.numpy(), samplerate=config.fs)
        print("[synthesize] {} -> {}".format(file_path, audio_path))


if __name__ == "__main__":
    main()
