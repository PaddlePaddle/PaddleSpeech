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

import paddle
import soundfile as sf
import yaml
from timer import timer
from yacs.config import CfgNode

from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_sentences
from paddlespeech.t2s.models.vits import VITS
from paddlespeech.t2s.utils import str2bool


def evaluate(args):

    # Init body.
    with open(args.config) as f:
        config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(config)

    sentences = get_sentences(text_file=args.text, lang=args.lang)

    # frontend
    frontend = get_frontend(lang=args.lang, phones_dict=args.phones_dict)

    spk_num = None
    if args.speaker_dict is not None:
        print("multiple speaker vits!")
        with open(args.speaker_dict, 'rt') as f:
            spk_id = [line.strip().split() for line in f.readlines()]
        spk_num = len(spk_id)
    else:
        print("single speaker vits!")
    print("spk_num:", spk_num)

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
    merge_sentences = False
    add_blank = args.add_blank

    N = 0
    T = 0
    for utt_id, sentence in sentences:
        with timer() as t:
            if args.lang == 'zh':
                input_ids = frontend.get_input_ids(
                    sentence,
                    merge_sentences=merge_sentences,
                    add_blank=add_blank)
                phone_ids = input_ids["phone_ids"]
            elif args.lang == 'en':
                input_ids = frontend.get_input_ids(
                    sentence, merge_sentences=merge_sentences)
                phone_ids = input_ids["phone_ids"]
            else:
                print("lang should in {'zh', 'en'}!")
            with paddle.no_grad():
                flags = 0
                for i in range(len(phone_ids)):
                    part_phone_ids = phone_ids[i]
                    spk_id = None
                    if spk_num is not None:
                        spk_id = paddle.to_tensor(args.spk_id)
                    out = vits.inference(text=part_phone_ids, sids=spk_id)
                    wav = out["wav"]
                    if flags == 0:
                        wav_all = wav
                        flags = 1
                    else:
                        wav_all = paddle.concat([wav_all, wav])
        wav = wav_all.numpy()
        N += wav.size
        T += t.elapse
        speed = wav.size / t.elapse
        rtf = config.fs / speed
        print(
            f"{utt_id}, wave: {wav.shape}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )
        sf.write(str(output_dir / (utt_id + ".wav")), wav, samplerate=config.fs)
        print(f"{utt_id} done!")
    print(f"generation speed: {N / T}Hz, RTF: {config.fs / (N / T) }")


def parse_args():
    # parse args and config 
    parser = argparse.ArgumentParser(description="Synthesize with VITS")

    # model
    parser.add_argument(
        '--config', type=str, default=None, help='Config of VITS.')
    parser.add_argument(
        '--ckpt', type=str, default=None, help='Checkpoint file of VITS.')
    parser.add_argument(
        "--phones_dict", type=str, default=None, help="phone vocabulary file.")
    parser.add_argument(
        "--speaker_dict", type=str, default=None, help="speaker id map file.")
    parser.add_argument(
        '--spk_id',
        type=int,
        default=0,
        help='spk id for multi speaker acoustic model')
    # other
    parser.add_argument(
        '--lang',
        type=str,
        default='zh',
        help='Choose model language. zh or en')

    parser.add_argument(
        "--inference_dir",
        type=str,
        default=None,
        help="dir to save inference models")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")
    parser.add_argument(
        "--text",
        type=str,
        help="text to synthesize, a 'utt_id sentence' pair per line.")
    parser.add_argument("--output_dir", type=str, help="output dir.")

    parser.add_argument(
        "--add-blank",
        type=str2bool,
        default=True,
        help="whether to add blank between phones")

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
