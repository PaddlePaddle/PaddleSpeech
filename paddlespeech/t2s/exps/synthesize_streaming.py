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
import math
from pathlib import Path

import numpy as np
import paddle
import soundfile as sf
import yaml
from timer import timer
from yacs.config import CfgNode

from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_sentences
from paddlespeech.t2s.exps.syn_utils import get_voc_inference
from paddlespeech.t2s.exps.syn_utils import model_alias
from paddlespeech.t2s.utils import str2bool


def denorm(data, mean, std):
    return data * std + mean


def get_chunks(data, chunk_size, pad_size):
    data_len = data.shape[1]
    chunks = []
    n = math.ceil(data_len / chunk_size)
    for i in range(n):
        start = max(0, i * chunk_size - pad_size)
        end = min((i + 1) * chunk_size + pad_size, data_len)
        chunks.append(data[:, start:end, :])
    return chunks


def evaluate(args):

    # Init body.
    with open(args.am_config) as f:
        am_config = CfgNode(yaml.safe_load(f))
    with open(args.voc_config) as f:
        voc_config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(am_config)
    print(voc_config)

    sentences = get_sentences(args)

    # frontend
    frontend = get_frontend(args)

    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)

    # acoustic model, only support fastspeech2 here now!
    # am_inference, am_name, am_dataset = get_am_inference(args, am_config)
    # model: {model_name}_{dataset}
    am_name = args.am[:args.am.rindex('_')]
    am_dataset = args.am[args.am.rindex('_') + 1:]
    odim = am_config.n_mels

    am_class = dynamic_import(am_name, model_alias)
    am = am_class(idim=vocab_size, odim=odim, **am_config["model"])
    am.set_state_dict(paddle.load(args.am_ckpt)["main_params"])
    am.eval()
    am_mu, am_std = np.load(args.am_stat)
    am_mu = paddle.to_tensor(am_mu)
    am_std = paddle.to_tensor(am_std)

    # vocoder
    voc_inference = get_voc_inference(args, voc_config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merge_sentences = True
    get_tone_ids = False

    N = 0
    T = 0
    chunk_size = args.chunk_size
    pad_size = args.pad_size

    for utt_id, sentence in sentences:
        with timer() as t:
            if args.lang == 'zh':
                input_ids = frontend.get_input_ids(
                    sentence,
                    merge_sentences=merge_sentences,
                    get_tone_ids=get_tone_ids)

                phone_ids = input_ids["phone_ids"]
            else:
                print("lang should in be 'zh' here!")
            # merge_sentences=True here, so we only use the first item of phone_ids
            phone_ids = phone_ids[0]
            with paddle.no_grad():
                # acoustic model
                orig_hs, h_masks = am.encoder_infer(phone_ids)

                if args.am_streaming:
                    hss = get_chunks(orig_hs, chunk_size, pad_size)
                    chunk_num = len(hss)
                    mel_list = []
                    for i, hs in enumerate(hss):
                        before_outs, _ = am.decoder(hs)
                        after_outs = before_outs + am.postnet(
                            before_outs.transpose((0, 2, 1))).transpose(
                                (0, 2, 1))
                        normalized_mel = after_outs[0]
                        sub_mel = denorm(normalized_mel, am_mu, am_std)
                        # clip output part of pad
                        if i == 0:
                            sub_mel = sub_mel[:-pad_size]
                        elif i == chunk_num - 1:
                            # 最后一块的右侧一定没有 pad 够
                            sub_mel = sub_mel[pad_size:]
                        else:
                            # 倒数几块的右侧也可能没有 pad 够
                            sub_mel = sub_mel[pad_size:(chunk_size + pad_size) -
                                              sub_mel.shape[0]]
                        mel_list.append(sub_mel)
                    mel = paddle.concat(mel_list, axis=0)

                else:
                    before_outs, _ = am.decoder(orig_hs)
                    after_outs = before_outs + am.postnet(
                        before_outs.transpose((0, 2, 1))).transpose((0, 2, 1))
                    normalized_mel = after_outs[0]
                    mel = denorm(normalized_mel, am_mu, am_std)

                # vocoder
                wav = voc_inference(mel)

        wav = wav.numpy()
        N += wav.size
        T += t.elapse
        speed = wav.size / t.elapse
        rtf = am_config.fs / speed
        print(
            f"{utt_id}, mel: {mel.shape}, wave: {wav.shape}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )
        sf.write(
            str(output_dir / (utt_id + ".wav")), wav, samplerate=am_config.fs)
        print(f"{utt_id} done!")
    print(f"generation speed: {N / T}Hz, RTF: {am_config.fs / (N / T) }")


def parse_args():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(
        description="Synthesize with acoustic model & vocoder")
    # acoustic model
    parser.add_argument(
        '--am',
        type=str,
        default='fastspeech2_csmsc',
        choices=['fastspeech2_csmsc'],
        help='Choose acoustic model type of tts task.')
    parser.add_argument(
        '--am_config',
        type=str,
        default=None,
        help='Config of acoustic model. Use deault config when it is None.')
    parser.add_argument(
        '--am_ckpt',
        type=str,
        default=None,
        help='Checkpoint file of acoustic model.')
    parser.add_argument(
        "--am_stat",
        type=str,
        default=None,
        help="mean and standard deviation used to normalize spectrogram when training acoustic model."
    )
    parser.add_argument(
        "--phones_dict", type=str, default=None, help="phone vocabulary file.")
    parser.add_argument(
        "--tones_dict", type=str, default=None, help="tone vocabulary file.")

    # vocoder
    parser.add_argument(
        '--voc',
        type=str,
        default='pwgan_csmsc',
        choices=[
            'pwgan_csmsc',
            'pwgan_ljspeech',
            'pwgan_aishell3',
            'pwgan_vctk',
            'mb_melgan_csmsc',
            'style_melgan_csmsc',
            'hifigan_csmsc',
            'hifigan_ljspeech',
            'hifigan_aishell3',
            'hifigan_vctk',
            'wavernn_csmsc',
        ],
        help='Choose vocoder type of tts task.')
    parser.add_argument(
        '--voc_config',
        type=str,
        default=None,
        help='Config of voc. Use deault config when it is None.')
    parser.add_argument(
        '--voc_ckpt', type=str, default=None, help='Checkpoint file of voc.')
    parser.add_argument(
        "--voc_stat",
        type=str,
        default=None,
        help="mean and standard deviation used to normalize spectrogram when training voc."
    )
    # other
    parser.add_argument(
        '--lang',
        type=str,
        default='zh',
        help='Choose model language. zh or en')

    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")
    parser.add_argument(
        "--text",
        type=str,
        help="text to synthesize, a 'utt_id sentence' pair per line.")

    parser.add_argument(
        "--am_streaming",
        type=str2bool,
        default=False,
        help="whether use streaming acoustic model")
    parser.add_argument(
        "--chunk_size", type=int, default=42, help="chunk size of am streaming")
    parser.add_argument(
        "--pad_size", type=int, default=12, help="pad size of am streaming")

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
