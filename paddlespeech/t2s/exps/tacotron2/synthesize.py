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
from pathlib import Path

import numpy as np
import paddle
from matplotlib import pyplot as plt

from paddlespeech.t2s.exps.tacotron2.config import get_cfg_defaults
from paddlespeech.t2s.frontend import EnglishCharacter
from paddlespeech.t2s.models.tacotron2 import Tacotron2
from paddlespeech.t2s.utils import display


def main(config, args):
    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    # model
    frontend = EnglishCharacter()
    model = Tacotron2.from_pretrained(config, args.checkpoint_path)
    model.eval()

    # inputs
    input_path = Path(args.input).expanduser()
    sentences = []
    with open(input_path, "rt") as f:
        for line in f:
            line_list = line.strip().split()
            utt_id = line_list[0]
            sentence = " ".join(line_list[1:])
            sentences.append((utt_id, sentence))

    if args.output is None:
        output_dir = input_path.parent / "synthesis"
    else:
        output_dir = Path(args.output).expanduser()
    output_dir.mkdir(exist_ok=True)

    for i, sentence in enumerate(sentences):
        sentence = paddle.to_tensor(frontend(sentence)).unsqueeze(0)
        outputs = model.infer(sentence)
        mel_output = outputs["mel_outputs_postnet"][0].numpy().T
        alignment = outputs["alignments"][0].numpy().T

        np.save(str(output_dir / f"sentence_{i}"), mel_output)
        display.plot_alignment(alignment)
        plt.savefig(str(output_dir / f"sentence_{i}.png"))
        if args.verbose:
            print("spectrogram saved at {}".format(output_dir /
                                                   f"sentence_{i}.npy"))


if __name__ == "__main__":
    config = get_cfg_defaults()

    parser = argparse.ArgumentParser(
        description="generate mel spectrogram with TransformerTTS.")
    parser.add_argument(
        "--config",
        type=str,
        metavar="FILE",
        help="extra config to overwrite the default config")
    parser.add_argument(
        "--checkpoint_path", type=str, help="path of the checkpoint to load.")
    parser.add_argument("--input", type=str, help="path of the text sentences")
    parser.add_argument("--output", type=str, help="path to save outputs")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")
    parser.add_argument(
        "--opts",
        nargs=argparse.REMAINDER,
        help="options to overwrite --config file and the default config, passing in KEY VALUE pairs"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="print msg")

    args = parser.parse_args()
    if args.config:
        config.merge_from_file(args.config)
    if args.opts:
        config.merge_from_list(args.opts)
    config.freeze()
    print(config)
    print(args)

    main(config, args)
