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

import numpy as np
import soundfile as sf
from timer import timer

from paddlespeech.t2s.exps.syn_utils import denorm
from paddlespeech.t2s.exps.syn_utils import get_chunks
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_sentences
from paddlespeech.t2s.exps.syn_utils import get_sess
from paddlespeech.t2s.utils import str2bool


def ort_predict(args):

    # frontend
    frontend = get_frontend(
        lang=args.lang,
        phones_dict=args.phones_dict,
        tones_dict=args.tones_dict)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sentences = get_sentences(text_file=args.text, lang=args.lang)

    am_name = args.am[:args.am.rindex('_')]
    am_dataset = args.am[args.am.rindex('_') + 1:]
    fs = 24000 if am_dataset != 'ljspeech' else 22050

    # streaming acoustic model
    am_encoder_infer_sess = get_sess(
        model_dir=args.inference_dir,
        model_file=args.am + "_am_encoder_infer" + ".onnx",
        device=args.device,
        cpu_threads=args.cpu_threads)
    am_decoder_sess = get_sess(
        model_dir=args.inference_dir,
        model_file=args.am + "_am_decoder" + ".onnx",
        device=args.device,
        cpu_threads=args.cpu_threads)

    am_postnet_sess = get_sess(
        model_dir=args.inference_dir,
        model_file=args.am + "_am_postnet" + ".onnx",
        device=args.device,
        cpu_threads=args.cpu_threads)
    am_mu, am_std = np.load(args.am_stat)

    # vocoder
    voc_sess = get_sess(
        model_dir=args.inference_dir,
        model_file=args.voc + ".onnx",
        device=args.device,
        cpu_threads=args.cpu_threads)

    # frontend warmup
    # Loading model cost 0.5+ seconds
    if args.lang == 'zh':
        frontend.get_input_ids("你好，欢迎使用飞桨框架进行深度学习研究！", merge_sentences=True)
    else:
        print("lang should in be 'zh' here!")

    # am warmup
    for T in [27, 38, 54]:
        phone_ids = np.random.randint(1, 266, size=(T, ))
        am_encoder_infer_sess.run(None, input_feed={'text': phone_ids})

        am_decoder_input = np.random.rand(1, T * 15, 384).astype('float32')
        am_decoder_sess.run(None, input_feed={'xs': am_decoder_input})

        am_postnet_input = np.random.rand(1, 80, T * 15).astype('float32')
        am_postnet_sess.run(None, input_feed={'xs': am_postnet_input})

    # voc warmup
    for T in [227, 308, 544]:
        data = np.random.rand(T, 80).astype("float32")
        voc_sess.run(None, input_feed={"logmel": data})
    print("warm up done!")

    N = 0
    T = 0
    merge_sentences = True
    get_tone_ids = False
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
            phone_ids = phone_ids[0].numpy()
            orig_hs = am_encoder_infer_sess.run(
                None, input_feed={'text': phone_ids})
            if args.am_streaming:
                hss = get_chunks(orig_hs[0], chunk_size, pad_size)
                chunk_num = len(hss)
                mel_list = []
                for i, hs in enumerate(hss):
                    am_decoder_output = am_decoder_sess.run(
                        None, input_feed={'xs': hs})
                    am_postnet_output = am_postnet_sess.run(
                        None,
                        input_feed={
                            'xs': np.transpose(am_decoder_output[0], (0, 2, 1))
                        })
                    am_output_data = am_decoder_output + np.transpose(
                        am_postnet_output[0], (0, 2, 1))
                    normalized_mel = am_output_data[0][0]

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
                mel = np.concatenate(mel_list, axis=0)
            else:
                am_decoder_output = am_decoder_sess.run(
                    None, input_feed={'xs': orig_hs[0]})
                am_postnet_output = am_postnet_sess.run(
                    None,
                    input_feed={
                        'xs': np.transpose(am_decoder_output[0], (0, 2, 1))
                    })
                am_output_data = am_decoder_output + np.transpose(
                    am_postnet_output[0], (0, 2, 1))
                normalized_mel = am_output_data[0]
                mel = denorm(normalized_mel, am_mu, am_std)
                mel = mel[0]
            # vocoder

            wav = voc_sess.run(output_names=None, input_feed={'logmel': mel})

            N += len(wav[0])
            T += t.elapse
            speed = len(wav[0]) / t.elapse
            rtf = fs / speed
        sf.write(
            str(output_dir / (utt_id + ".wav")),
            np.array(wav)[0],
            samplerate=fs)
        print(
            f"{utt_id}, mel: {mel.shape}, wave: {len(wav[0])}, time: {t.elapse}s, Hz: {speed}, RTF: {rtf}."
        )
    print(f"generation speed: {N / T}Hz, RTF: {fs / (N / T) }")


def parse_args():
    parser = argparse.ArgumentParser(description="Infernce with onnxruntime.")
    # acoustic model
    parser.add_argument(
        '--am',
        type=str,
        default='fastspeech2_csmsc',
        choices=['fastspeech2_csmsc'],
        help='Choose acoustic model type of tts task.')
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

    # voc
    parser.add_argument(
        '--voc',
        type=str,
        default='hifigan_csmsc',
        choices=['hifigan_csmsc', 'mb_melgan_csmsc', 'pwgan_csmsc'],
        help='Choose vocoder type of tts task.')
    # other
    parser.add_argument(
        "--inference_dir", type=str, help="dir to save inference models")
    parser.add_argument(
        "--text",
        type=str,
        help="text to synthesize, a 'utt_id sentence' pair per line")
    parser.add_argument("--output_dir", type=str, help="output dir")
    parser.add_argument(
        '--lang',
        type=str,
        default='zh',
        help='Choose model language. zh or en')

    # inference
    parser.add_argument(
        "--use_trt",
        type=str2bool,
        default=False,
        help="Whether to use inference engin TensorRT.", )

    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu"],
        help="Device selected for inference.", )
    parser.add_argument('--cpu_threads', type=int, default=1)

    # streaming related
    parser.add_argument(
        "--am_streaming",
        type=str2bool,
        default=False,
        help="whether use streaming acoustic model")
    parser.add_argument(
        "--chunk_size", type=int, default=42, help="chunk size of am streaming")
    parser.add_argument(
        "--pad_size", type=int, default=12, help="pad size of am streaming")

    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    ort_predict(args)


if __name__ == "__main__":
    main()
