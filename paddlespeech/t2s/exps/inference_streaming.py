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
from pathlib import Path

import numpy as np
import paddle
import soundfile as sf
from timer import timer

from paddlespeech.t2s.exps.syn_utils import denorm
from paddlespeech.t2s.exps.syn_utils import get_am_sublayer_output
from paddlespeech.t2s.exps.syn_utils import get_chunks
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_predictor
from paddlespeech.t2s.exps.syn_utils import get_sentences
from paddlespeech.t2s.exps.syn_utils import get_streaming_am_output
from paddlespeech.t2s.exps.syn_utils import get_voc_output
from paddlespeech.t2s.exps.syn_utils import run_frontend
from paddlespeech.t2s.utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paddle Infernce with acoustic model & vocoder.")
    # acoustic model
    parser.add_argument('--am',
                        type=str,
                        default='fastspeech2_csmsc',
                        choices=['fastspeech2_csmsc'],
                        help='Choose acoustic model type of tts task.')
    parser.add_argument(
        "--am_stat",
        type=str,
        default=None,
        help=
        "mean and standard deviation used to normalize spectrogram when training acoustic model."
    )
    parser.add_argument("--phones_dict",
                        type=str,
                        default=None,
                        help="phone vocabulary file.")
    parser.add_argument("--tones_dict",
                        type=str,
                        default=None,
                        help="tone vocabulary file.")
    parser.add_argument("--speaker_dict",
                        type=str,
                        default=None,
                        help="speaker id map file.")
    parser.add_argument('--spk_id',
                        type=int,
                        default=0,
                        help='spk id for multi speaker acoustic model')
    # voc
    parser.add_argument(
        '--voc',
        type=str,
        default='pwgan_csmsc',
        choices=['pwgan_csmsc', 'mb_melgan_csmsc', 'hifigan_csmsc'],
        help='Choose vocoder type of tts task.')
    # other
    parser.add_argument('--lang',
                        type=str,
                        default='zh',
                        help='Choose model language. zh or en')
    parser.add_argument(
        "--text",
        type=str,
        help="text to synthesize, a 'utt_id sentence' pair per line")
    parser.add_argument("--inference_dir",
                        type=str,
                        help="dir to save inference models")
    parser.add_argument("--output_dir", type=str, help="output dir")
    # inference
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu"],
        help="Device selected for inference.",
    )
    # streaming related
    parser.add_argument("--am_streaming",
                        type=str2bool,
                        default=False,
                        help="whether use streaming acoustic model")
    parser.add_argument("--block_size",
                        type=int,
                        default=42,
                        help="block size of am streaming")
    parser.add_argument("--pad_size",
                        type=int,
                        default=12,
                        help="pad size of am streaming")

    args, _ = parser.parse_known_args()
    return args


# only inference for models trained with csmsc now
def main():
    args = parse_args()

    paddle.set_device(args.device)

    # frontend
    frontend = get_frontend(lang=args.lang,
                            phones_dict=args.phones_dict,
                            tones_dict=args.tones_dict)

    # am_predictor

    am_encoder_infer_predictor = get_predictor(
        model_dir=args.inference_dir,
        model_file=args.am + "_am_encoder_infer" + ".pdmodel",
        params_file=args.am + "_am_encoder_infer" + ".pdiparams",
        device=args.device)
    am_decoder_predictor = get_predictor(
        model_dir=args.inference_dir,
        model_file=args.am + "_am_decoder" + ".pdmodel",
        params_file=args.am + "_am_decoder" + ".pdiparams",
        device=args.device)
    am_postnet_predictor = get_predictor(
        model_dir=args.inference_dir,
        model_file=args.am + "_am_postnet" + ".pdmodel",
        params_file=args.am + "_am_postnet" + ".pdiparams",
        device=args.device)
    am_mu, am_std = np.load(args.am_stat)
    # model: {model_name}_{dataset}
    am_dataset = args.am[args.am.rindex('_') + 1:]

    # voc_predictor
    voc_predictor = get_predictor(model_dir=args.inference_dir,
                                  model_file=args.voc + ".pdmodel",
                                  params_file=args.voc + ".pdiparams",
                                  device=args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = get_sentences(text_file=args.text, lang=args.lang)

    merge_sentences = True

    fs = 24000 if am_dataset != 'ljspeech' else 22050
    # warmup
    for utt_id, sentence in sentences[:3]:
        with timer() as t:
            normalized_mel = get_streaming_am_output(
                input=sentence,
                am_encoder_infer_predictor=am_encoder_infer_predictor,
                am_decoder_predictor=am_decoder_predictor,
                am_postnet_predictor=am_postnet_predictor,
                frontend=frontend,
                lang=args.lang,
                merge_sentences=merge_sentences,
            )
            mel = denorm(normalized_mel, am_mu, am_std)
            wav = get_voc_output(voc_predictor=voc_predictor, input=mel)
        speed = wav.size / t.elapse
        rtf = fs / speed
        print(
            f"{utt_id}, mel: {mel.shape}, wave: {wav.shape}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )

    print("warm up done!")

    N = 0
    T = 0
    block_size = args.block_size
    pad_size = args.pad_size
    get_tone_ids = False
    for utt_id, sentence in sentences:
        with timer() as t:
            # frontend
            frontend_dict = run_frontend(frontend=frontend,
                                         text=sentence,
                                         merge_sentences=merge_sentences,
                                         get_tone_ids=get_tone_ids,
                                         lang=args.lang)
            phone_ids = frontend_dict['phone_ids']
            phones = phone_ids[0].numpy()
            # acoustic model
            orig_hs = get_am_sublayer_output(am_encoder_infer_predictor,
                                             input=phones)

            if args.am_streaming:
                hss = get_chunks(orig_hs, block_size, pad_size)
                chunk_num = len(hss)
                mel_list = []
                for i, hs in enumerate(hss):
                    am_decoder_output = get_am_sublayer_output(
                        am_decoder_predictor, input=hs)
                    am_postnet_output = get_am_sublayer_output(
                        am_postnet_predictor,
                        input=np.transpose(am_decoder_output, (0, 2, 1)))
                    am_output_data = am_decoder_output + np.transpose(
                        am_postnet_output, (0, 2, 1))
                    normalized_mel = am_output_data[0]

                    sub_mel = denorm(normalized_mel, am_mu, am_std)
                    # clip output part of pad
                    if i == 0:
                        sub_mel = sub_mel[:-pad_size]
                    elif i == chunk_num - 1:
                        # 最后一块的右侧一定没有 pad 够
                        sub_mel = sub_mel[pad_size:]
                    else:
                        # 倒数几块的右侧也可能没有 pad 够
                        sub_mel = sub_mel[pad_size:(block_size + pad_size) -
                                          sub_mel.shape[0]]
                    mel_list.append(sub_mel)
                mel = np.concatenate(mel_list, axis=0)

            else:
                am_decoder_output = get_am_sublayer_output(am_decoder_predictor,
                                                           input=orig_hs)

                am_postnet_output = get_am_sublayer_output(
                    am_postnet_predictor,
                    input=np.transpose(am_decoder_output, (0, 2, 1)))
                am_output_data = am_decoder_output + np.transpose(
                    am_postnet_output, (0, 2, 1))
                normalized_mel = am_output_data[0]
                mel = denorm(normalized_mel, am_mu, am_std)
            # vocoder
            wav = get_voc_output(voc_predictor=voc_predictor, input=mel)

        N += wav.size
        T += t.elapse
        speed = wav.size / t.elapse
        rtf = fs / speed

        sf.write(output_dir / (utt_id + ".wav"), wav, samplerate=24000)
        print(
            f"{utt_id}, mel: {mel.shape}, wave: {wav.shape}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )

        print(f"{utt_id} done!")
    print(f"generation speed: {N / T}Hz, RTF: {fs / (N / T) }")


if __name__ == "__main__":
    main()
