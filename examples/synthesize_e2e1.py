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

from paddlespeech.t2s.frontend import English,Chinese
from paddlespeech.t2s.models.transformer_tts import TransformerTTS
from paddlespeech.t2s.models.transformer_tts import TransformerTTSInference
from paddlespeech.t2s.models.waveflow import ConditionalWaveFlow
from paddlespeech.t2s.modules.normalizer import ZScore
from paddlespeech.t2s.utils import layer_tools
from paddlespeech.t2s.exps.syn_utils import get_voc_inference2
from paddlespeech.t2s.exps.syn_utils import get_am_inference

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
    model = TransformerTTS(idim=vocab_size, odim=odim, **acoustic_model_config["model"])
    model.set_state_dict(paddle.load(args.transformer_tts_checkpoint)["main_params"])
    model.eval()

    vocoder_checkpoint_path = args.waveflow_checkpoint[:-9] if args.waveflow_checkpoint.endswith(".pdparams") else args.waveflow_checkpoint
    vocoder = ConditionalWaveFlow.from_pretrained(vocoder_config,vocoder_checkpoint_path)
    layer_tools.recursively_remove_weight_norm(vocoder)
    vocoder.eval()
    print("model done!")

    # vocoder = get_voc_inference2(
    #     voc=args.voc,
    #     voc_config=vocoder_config,
    #     voc_ckpt=args.voc_ckpt,
    #     voc_stat=args.voc_stat)

    frontend = Chinese()
    print("frontend done!")

    stat = np.load(args.transformer_tts_stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    transformer_tts_normalizer = ZScore(mu, std)
    transformer_tts_inference = TransformerTTSInference(transformer_tts_normalizer, model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for utt_id, sentence in sentences:
        phones = frontend.phoneticize(sentence)
        print(sentence)
        print(phones)
        # remove start_symbol and end_symbol
        phones = phones[1:-1]
        phones = [phn for phn in phones if not phn.isspace()]
        phones = [phn if phn in phone_id_map else "," for phn in phones]
        phone_ids = [phone_id_map[phn] for phn in phones]
        print('1',phone_ids)
        with paddle.no_grad():
            tensor_phone_ids=paddle.to_tensor(phone_ids)
            mel = transformer_tts_inference(tensor_phone_ids)
            # mel shape is (T, feats) and waveflow's input shape is (batch, feats, T)
            mel = mel.unsqueeze(0).transpose([0, 2, 1])
            # wavflow's output shape is (B, T)
            wav = vocoder.infer(mel)[0]
            #wav = vocoder(mel)[0]
        sf.write(str(output_dir / (utt_id + ".wav")),wav.numpy(),samplerate=acoustic_model_config.fs)
        #sf.write(str(output_dir / (utt_id + ".wav")), wav.numpy(), samplerate=24000)
        print(f"{utt_id} done!")


def main():
    parser = argparse.ArgumentParser(
        description="Synthesize with transformer tts & waveflow.")
    parser.add_argument(
        "--transformer-tts-config",
        default="./out_put/default.yaml",
        type=str,
        help="transformer tts config file.")
    parser.add_argument(
        "--transformer-tts-checkpoint",
        default="./out_put/checkpoints/snapshot_iter_1113750.pdz",
        type=str,
        help="transformer tts checkpoint to load.")
    parser.add_argument(
        "--transformer-tts-stat",
        default="./dump/speech_stats.npy",
        type=str,
        help="mean and standard deviation used to normalize spectrogram when training transformer tts."
    )


    parser.add_argument(
        "--waveflow-config", default="./waveflow_ljspeech_ckpt_0.3/config.yaml", type=str, help="waveflow config file.")
    # not normalize when training waveflow
    parser.add_argument(
        "--waveflow-checkpoint", default="./waveflow_ljspeech_ckpt_0.3/step-2000000.pdparams", type=str,
        help="waveflow checkpoint to load.")

    parser.add_argument(
        "--phones-dict", type=str, default="./dump/phone_id_map.txt", help="phone vocabulary file.")
    parser.add_argument(
        "--text",
        default="./sentences.txt",
        type=str,
        help="text to synthesize, a 'utt_id sentence' pair per line.")
    parser.add_argument("--output-dir", default="./output222", type=str, help="output dir.")
    parser.add_argument("--ngpu", type=int, default=0, help="if ngpu == 0, use cpu.")

    args = parser.parse_args()

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    with open(args.transformer_tts_config) as f:
        transformer_tts_config = CfgNode(yaml.safe_load(f))
    with open(args.waveflow_config) as f:
        waveflow_config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(transformer_tts_config)
    print(waveflow_config)

    evaluate(args, transformer_tts_config, waveflow_config)


if __name__ == "__main__":
    main()
