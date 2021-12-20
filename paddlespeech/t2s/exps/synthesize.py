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

import jsonlines
import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode

from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.modules.normalizer import ZScore

model_alias = {
    # acoustic model
    "speedyspeech":
    "paddlespeech.t2s.models.speedyspeech:SpeedySpeech",
    "speedyspeech_inference":
    "paddlespeech.t2s.models.speedyspeech:SpeedySpeechInference",
    "fastspeech2":
    "paddlespeech.t2s.models.fastspeech2:FastSpeech2",
    "fastspeech2_inference":
    "paddlespeech.t2s.models.fastspeech2:FastSpeech2Inference",
    # voc
    "pwgan":
    "paddlespeech.t2s.models.parallel_wavegan:PWGGenerator",
    "pwgan_inference":
    "paddlespeech.t2s.models.parallel_wavegan:PWGInference",
    "mb_melgan":
    "paddlespeech.t2s.models.melgan:MelGANGenerator",
    "mb_melgan_inference":
    "paddlespeech.t2s.models.melgan:MelGANInference",
}


def evaluate(args):
    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True

    # construct dataset for evaluation
    with jsonlines.open(args.test_metadata, 'r') as reader:
        test_metadata = list(reader)

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

    # construct dataset for evaluation

    # model: {model_name}_{dataset}
    am_name = args.am[:args.am.rindex('_')]
    am_dataset = args.am[args.am.rindex('_') + 1:]

    if am_name == 'fastspeech2':
        fields = ["utt_id", "text"]
        spk_num = None
        if am_dataset in {"aishell3", "vctk"} and args.speaker_dict:
            print("multiple speaker fastspeech2!")
            with open(args.speaker_dict, 'rt') as f:
                spk_id = [line.strip().split() for line in f.readlines()]
            spk_num = len(spk_id)
            fields += ["spk_id"]
        elif args.voice_cloning:
            print("voice cloning!")
            fields += ["spk_emb"]
        else:
            print("single speaker fastspeech2!")
        print("spk_num:", spk_num)
    elif am_name == 'speedyspeech':
        fields = ["utt_id", "phones", "tones"]

    test_dataset = DataTable(data=test_metadata, fields=fields)

    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)

    tone_size = None
    if args.tones_dict:
        with open(args.tones_dict, "r") as f:
            tone_id = [line.strip().split() for line in f.readlines()]
        tone_size = len(tone_id)
        print("tone_size:", tone_size)

    # acoustic model
    odim = am_config.n_mels
    am_class = dynamic_import(am_name, model_alias)
    am_inference_class = dynamic_import(am_name + '_inference', model_alias)

    if am_name == 'fastspeech2':
        am = am_class(
            idim=vocab_size, odim=odim, spk_num=spk_num, **am_config["model"])
    elif am_name == 'speedyspeech':
        am = am_class(
            vocab_size=vocab_size, tone_size=tone_size, **am_config["model"])

    am.set_state_dict(paddle.load(args.am_ckpt)["main_params"])
    am.eval()
    am_mu, am_std = np.load(args.am_stat)
    am_mu = paddle.to_tensor(am_mu)
    am_std = paddle.to_tensor(am_std)
    am_normalizer = ZScore(am_mu, am_std)
    am_inference = am_inference_class(am_normalizer, am)
    print("am_inference.training0:", am_inference.training)
    am_inference.eval()
    print("acoustic model done!")

    # vocoder
    # model: {model_name}_{dataset}
    voc_name = args.voc[:args.voc.rindex('_')]
    voc_class = dynamic_import(voc_name, model_alias)
    voc_inference_class = dynamic_import(voc_name + '_inference', model_alias)
    voc = voc_class(**voc_config["generator_params"])
    voc.set_state_dict(paddle.load(args.voc_ckpt)["generator_params"])
    voc.remove_weight_norm()
    voc.eval()
    voc_mu, voc_std = np.load(args.voc_stat)
    voc_mu = paddle.to_tensor(voc_mu)
    voc_std = paddle.to_tensor(voc_std)
    voc_normalizer = ZScore(voc_mu, voc_std)
    voc_inference = voc_inference_class(voc_normalizer, voc)
    print("voc_inference.training0:", voc_inference.training)
    voc_inference.eval()
    print("voc done!")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for datum in test_dataset:
        utt_id = datum["utt_id"]
        with paddle.no_grad():
            # acoustic model
            if am_name == 'fastspeech2':
                phone_ids = paddle.to_tensor(datum["text"])
                spk_emb = None
                spk_id = None
                # multi speaker
                if args.voice_cloning and "spk_emb" in datum:
                    spk_emb = paddle.to_tensor(np.load(datum["spk_emb"]))
                elif "spk_id" in datum:
                    spk_id = paddle.to_tensor(datum["spk_id"])
                mel = am_inference(phone_ids, spk_id=spk_id, spk_emb=spk_emb)
            elif am_name == 'speedyspeech':
                phone_ids = paddle.to_tensor(datum["phones"])
                tone_ids = paddle.to_tensor(datum["tones"])
                mel = am_inference(phone_ids, tone_ids)
            # vocoder
            wav = voc_inference(mel)
        sf.write(
            str(output_dir / (utt_id + ".wav")),
            wav.numpy(),
            samplerate=am_config.fs)
        print(f"{utt_id} done!")


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(
        description="Synthesize with acoustic model & vocoder")
    # acoustic model
    parser.add_argument(
        '--am',
        type=str,
        default='fastspeech2_csmsc',
        choices=[
            'speedyspeech_csmsc', 'fastspeech2_csmsc', 'fastspeech2_ljspeech',
            'fastspeech2_aishell3', 'fastspeech2_vctk'
        ],
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
    parser.add_argument(
        "--speaker_dict", type=str, default=None, help="speaker id map file.")

    def str2bool(str):
        return True if str.lower() == 'true' else False

    parser.add_argument(
        "--voice-cloning",
        type=str2bool,
        default=False,
        help="whether training voice cloning model.")
    # vocoder
    parser.add_argument(
        '--voc',
        type=str,
        default='pwgan_csmsc',
        choices=[
            'pwgan_csmsc', 'pwgan_ljspeech', 'pwgan_aishell3', 'pwgan_vctk',
            'mb_melgan_csmsc'
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
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")
    parser.add_argument("--test_metadata", type=str, help="test metadata.")
    parser.add_argument("--output_dir", type=str, help="output dir.")

    args = parser.parse_args()

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    evaluate(args)


if __name__ == "__main__":
    main()
