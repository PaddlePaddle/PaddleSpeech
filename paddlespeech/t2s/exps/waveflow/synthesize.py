# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddlespeech.t2s.exps.waveflow.config import get_cfg_defaults
from paddlespeech.t2s.models.waveflow import ConditionalWaveFlow
from paddlespeech.t2s.utils import layer_tools


def main(config, args):
    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    model = ConditionalWaveFlow.from_pretrained(config, args.checkpoint_path)
    layer_tools.recursively_remove_weight_norm(model)
    model.eval()

    mel_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_path in mel_dir.glob("*.npy"):
        mel = np.load(str(file_path))
        with paddle.amp.auto_cast():
            audio = model.predict(mel)
        audio_path = output_dir / (os.path.splitext(file_path.name)[0] + ".wav")
        sf.write(audio_path, audio, config.data.sample_rate)
        print("[synthesize] {} -> {}".format(file_path, audio_path))


if __name__ == "__main__":
    config = get_cfg_defaults()

    parser = argparse.ArgumentParser(
        description="generate mel spectrogram with TransformerTTS.")
    parser.add_argument("--config",
                        type=str,
                        metavar="FILE",
                        help="extra config to overwrite the default config")
    parser.add_argument("--checkpoint_path",
                        type=str,
                        help="path of the checkpoint to load.")
    parser.add_argument(
        "--input",
        type=str,
        help="path of directory containing mel spectrogram (in .npy format)")
    parser.add_argument("--output", type=str, help="path to save outputs")
    parser.add_argument("--ngpu",
                        type=int,
                        default=1,
                        help="if ngpu=0, use cpu.")
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        help=
        "options to overwrite --config file and the default config, passing in KEY VALUE pairs"
    )

    args = parser.parse_args()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config)
    print(args)

    main(config, args)
