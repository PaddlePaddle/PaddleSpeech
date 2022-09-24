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
import logging
from pathlib import Path

import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode

from paddlespeech.t2s.frontend import Chinese
from paddlespeech.t2s.models.transformer_tts import TransformerTTS
from paddlespeech.t2s.models.transformer_tts import TransformerTTSInference
from paddlespeech.t2s.modules.normalizer import ZScore

#from paddlespeech.t2s.frontend import English
#from paddlespeech.t2s.models.waveflow import ConditionalWaveFlow

#from paddlespeech.t2s.utils import layer_tools


def evaluate(args, acoustic_model_config, vocoder_config):
    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True

    # construct dataset for evaluation
    sentences = []
    with open(args.text, 'rt') as f:
        for line in f:
            line_list = line.strip().split()
            utt_id = line_list[0]
            sentence = " ".join(line_list[1:])
            sentences.append((utt_id, sentence))

    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]

    vocab_size = len(phn_id)
    phone_id_map = {}
    for phn, id in phn_id:
        phone_id_map[phn] = int(id)
    print("vocab_size:", vocab_size)
    odim = acoustic_model_config.n_mels
    model = TransformerTTS(
        idim=vocab_size, odim=odim, **acoustic_model_config["model"])

    model.set_state_dict(
        paddle.load(args.transformer_tts_checkpoint)["main_params"])
    model.eval()

    # remove ".pdparams" in waveflow_checkpoint
    vocoder = get_voc_inference(
        voc=args.voc,
        voc_config=vocoder_config,
        voc_ckpt=args.voc_ckpt,
        voc_stat=args.voc_stat)
    vocoder.eval()
    print("model done!")

    frontend = Chinese()
    print("frontend done!")

    stat = np.load(args.transformer_tts_stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    transformer_tts_normalizer = ZScore(mu, std)

    transformer_tts_inference = TransformerTTSInference(
        transformer_tts_normalizer, model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for utt_id, sentence in sentences:
        phones = frontend.phoneticize(sentence)
        # remove start_symbol and end_symbol
        phones = phones[1:-1]
        phones = [phn for phn in phones if not phn.isspace()]
        phones = [phn if phn in phone_id_map else "," for phn in phones]
        phone_ids = [phone_id_map[phn] for phn in phones]
        with paddle.no_grad():
            tensor_phone_ids = paddle.to_tensor(phone_ids)
            mel = transformer_tts_inference(tensor_phone_ids)

            wav = vocoder(mel)

        sf.write(
            str(output_dir / (utt_id + ".wav")),
            wav.numpy(),
            samplerate=acoustic_model_config.fs)
        print(f"{utt_id} done!")


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(
        description="Synthesize with transformer tts & waveflow.")
    parser.add_argument(
        "--transformer-tts-config",
        type=str,
        help="transformer tts config file.")
    parser.add_argument(
        "--transformer-tts-checkpoint",
        type=str,
        help="transformer tts checkpoint to load.")
    parser.add_argument(
        "--transformer-tts-stat",
        type=str,
        help="mean and standard deviation used to normalize spectrogram when training transformer tts."
    )

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
        '--voc_config', type=str, default=None, help='Config of voc.')
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
        "--phones_dict", type=str, default=None, help="phone vocabulary file.")
    parser.add_argument(
        '--lang',
        type=str,
        default='zh',
        help='Choose model language. zh or en or mix')
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")
    parser.add_argument(
        "--text",
        type=str,
        help="text to synthesize, a 'utt_id sentence' pair per line.")
    parser.add_argument("--output_dir", type=str, help="output dir.")

    args = parser.parse_args()

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    with open(args.transformer_tts_config) as f:
        transformer_tts_config = CfgNode(yaml.safe_load(f))
    with open(args.voc_config) as f:
        voc_config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(transformer_tts_config)
    print(voc_config)

    evaluate(args, transformer_tts_config, voc_config)


if __name__ == "__main__":
    main()
