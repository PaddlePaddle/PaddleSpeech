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

import soundfile as sf
from paddle import inference

from paddlespeech.t2s.frontend.zh_frontend import Frontend


# only inference for models trained with csmsc now
def main():
    parser = argparse.ArgumentParser(
        description="Paddle Infernce with speedyspeech & parallel wavegan.")
    # acoustic model
    parser.add_argument(
        '--am',
        type=str,
        default='fastspeech2_csmsc',
        choices=['speedyspeech_csmsc', 'fastspeech2_csmsc'],
        help='Choose acoustic model type of tts task.')
    parser.add_argument(
        "--phones_dict", type=str, default=None, help="phone vocabulary file.")
    parser.add_argument(
        "--tones_dict", type=str, default=None, help="tone vocabulary file.")
    # voc
    parser.add_argument(
        '--voc',
        type=str,
        default='pwgan_csmsc',
        choices=['pwgan_csmsc', 'mb_melgan_csmsc', 'hifigan_csmsc'],
        help='Choose vocoder type of tts task.')
    # other
    parser.add_argument(
        "--text",
        type=str,
        help="text to synthesize, a 'utt_id sentence' pair per line")
    parser.add_argument(
        "--inference_dir", type=str, help="dir to save inference models")
    parser.add_argument("--output_dir", type=str, help="output dir")

    args, _ = parser.parse_known_args()

    frontend = Frontend(
        phone_vocab_path=args.phones_dict, tone_vocab_path=args.tones_dict)
    print("frontend done!")

    # model: {model_name}_{dataset}
    am_name = args.am[:args.am.rindex('_')]
    am_dataset = args.am[args.am.rindex('_') + 1:]

    am_config = inference.Config(
        str(Path(args.inference_dir) / (args.am + ".pdmodel")),
        str(Path(args.inference_dir) / (args.am + ".pdiparams")))
    am_config.enable_use_gpu(100, 0)
    # This line must be commented for fastspeech2, if not, it will OOM
    if am_name != 'fastspeech2':
        am_config.enable_memory_optim()
    am_predictor = inference.create_predictor(am_config)

    voc_config = inference.Config(
        str(Path(args.inference_dir) / (args.voc + ".pdmodel")),
        str(Path(args.inference_dir) / (args.voc + ".pdiparams")))
    voc_config.enable_use_gpu(100, 0)
    voc_config.enable_memory_optim()
    voc_predictor = inference.create_predictor(voc_config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sentences = []

    print("in new inference")

    with open(args.text, 'rt') as f:
        for line in f:
            items = line.strip().split()
            utt_id = items[0]
            sentence = "".join(items[1:])
            sentences.append((utt_id, sentence))

    get_tone_ids = False
    if am_name == 'speedyspeech':
        get_tone_ids = True

    am_input_names = am_predictor.get_input_names()

    for utt_id, sentence in sentences:
        input_ids = frontend.get_input_ids(
            sentence, merge_sentences=True, get_tone_ids=get_tone_ids)
        phone_ids = input_ids["phone_ids"]
        if get_tone_ids:
            tone_ids = input_ids["tone_ids"]
            tones = tone_ids[0].numpy()
            tones_handle = am_predictor.get_input_handle(am_input_names[1])
            tones_handle.reshape(tones.shape)
            tones_handle.copy_from_cpu(tones)

        phones = phone_ids[0].numpy()
        phones_handle = am_predictor.get_input_handle(am_input_names[0])
        phones_handle.reshape(phones.shape)
        phones_handle.copy_from_cpu(phones)

        am_predictor.run()
        am_output_names = am_predictor.get_output_names()
        am_output_handle = am_predictor.get_output_handle(am_output_names[0])
        am_output_data = am_output_handle.copy_to_cpu()

        voc_input_names = voc_predictor.get_input_names()
        mel_handle = voc_predictor.get_input_handle(voc_input_names[0])
        mel_handle.reshape(am_output_data.shape)
        mel_handle.copy_from_cpu(am_output_data)

        voc_predictor.run()
        voc_output_names = voc_predictor.get_output_names()
        voc_output_handle = voc_predictor.get_output_handle(voc_output_names[0])
        wav = voc_output_handle.copy_to_cpu()

        sf.write(output_dir / (utt_id + ".wav"), wav, samplerate=24000)

        print(f"{utt_id} done!")


if __name__ == "__main__":
    main()
