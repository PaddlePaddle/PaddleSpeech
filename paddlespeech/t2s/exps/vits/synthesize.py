# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import jsonlines
import numpy as np
import paddle
import soundfile as sf
import yaml
from timer import timer
from yacs.config import CfgNode

from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.models.vits import VITS
from paddlespeech.t2s.utils import str2bool


def evaluate(args):

    # construct dataset for evaluation
    with jsonlines.open(args.test_metadata, 'r') as reader:
        test_metadata = list(reader)
    # Init body.
    with open(args.config) as f:
        config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(config)

    fields = ["utt_id", "text"]
    converters = {}

    spk_num = None
    if args.speaker_dict is not None:
        print("multiple speaker vits!")
        with open(args.speaker_dict, 'rt') as f:
            spk_id = [line.strip().split() for line in f.readlines()]
        spk_num = len(spk_id)
        fields += ["spk_id"]
    elif args.voice_cloning:
        print("Evaluating voice cloning!")
        fields += ["spk_emb"]
    else:
        print("single speaker vits!")
    print("spk_num:", spk_num)

    test_dataset = DataTable(
        data=test_metadata,
        fields=fields,
        converters=converters,
    )

    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)

    odim = config.n_fft // 2 + 1
    config["model"]["generator_params"]["spks"] = spk_num

    vits = VITS(idim=vocab_size, odim=odim, **config["model"])
    vits.set_state_dict(paddle.load(args.ckpt)["main_params"])
    vits.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N = 0
    T = 0

    for datum in test_dataset:
        utt_id = datum["utt_id"]
        phone_ids = paddle.to_tensor(datum["text"])
        with timer() as t:
            with paddle.no_grad():
                spk_emb = None
                spk_id = None
                # multi speaker
                if args.voice_cloning and "spk_emb" in datum:
                    spk_emb = paddle.to_tensor(np.load(datum["spk_emb"]))
                elif "spk_id" in datum:
                    spk_id = paddle.to_tensor(datum["spk_id"])
                out = vits.inference(text=phone_ids,
                                     sids=spk_id,
                                     spembs=spk_emb)
            wav = out["wav"]
            wav = wav.numpy()
            N += wav.size
            T += t.elapse
            speed = wav.size / t.elapse
            rtf = config.fs / speed
        print(
            f"{utt_id}, wave: {wav.size}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )
        sf.write(str(output_dir / (utt_id + ".wav")), wav, samplerate=config.fs)
        print(f"{utt_id} done!")
    print(f"generation speed: {N / T}Hz, RTF: {config.fs / (N / T) }")


def parse_args():
    # parse args and config
    parser = argparse.ArgumentParser(description="Synthesize with VITS")
    # model
    parser.add_argument('--config',
                        type=str,
                        default=None,
                        help='Config of VITS.')
    parser.add_argument('--ckpt',
                        type=str,
                        default=None,
                        help='Checkpoint file of VITS.')
    parser.add_argument("--phones_dict",
                        type=str,
                        default=None,
                        help="phone vocabulary file.")
    parser.add_argument("--speaker_dict",
                        type=str,
                        default=None,
                        help="speaker id map file.")
    parser.add_argument("--voice-cloning",
                        type=str2bool,
                        default=False,
                        help="whether training voice cloning model.")
    # other
    parser.add_argument("--ngpu",
                        type=int,
                        default=1,
                        help="if ngpu == 0, use cpu.")
    parser.add_argument("--test_metadata", type=str, help="test metadata.")
    parser.add_argument("--output_dir", type=str, help="output dir.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    evaluate(args)


if __name__ == "__main__":
    main()
