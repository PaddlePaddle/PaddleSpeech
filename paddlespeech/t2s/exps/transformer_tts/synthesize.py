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

from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.models.transformer_tts import TransformerTTS
from paddlespeech.t2s.models.transformer_tts import TransformerTTSInference
from paddlespeech.t2s.models.waveflow import ConditionalWaveFlow
from paddlespeech.t2s.modules.normalizer import ZScore
from paddlespeech.t2s.utils import layer_tools


def evaluate(args, acoustic_model_config, vocoder_config):
    # dataloader has been too verbose
    logging.getLogger("DataLoader").disabled = True

    # construct dataset for evaluation
    with jsonlines.open(args.test_metadata, 'r') as reader:
        test_metadata = list(reader)
    test_dataset = DataTable(data=test_metadata, fields=["utt_id", "text"])

    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)
    odim = acoustic_model_config.n_mels
    model = TransformerTTS(
        idim=vocab_size, odim=odim, **acoustic_model_config["model"])

    model.set_state_dict(
        paddle.load(args.transformer_tts_checkpoint)["main_params"])
    model.eval()
    # remove ".pdparams" in waveflow_checkpoint
    vocoder_checkpoint_path = args.waveflow_checkpoint[:-9] if args.waveflow_checkpoint.endswith(
        ".pdparams") else args.waveflow_checkpoint
    vocoder = ConditionalWaveFlow.from_pretrained(vocoder_config,
                                                  vocoder_checkpoint_path)
    layer_tools.recursively_remove_weight_norm(vocoder)
    vocoder.eval()
    print("model done!")

    stat = np.load(args.transformer_tts_stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    transformer_tts_normalizer = ZScore(mu, std)

    transformer_tts_inference = TransformerTTSInference(
        transformer_tts_normalizer, model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for datum in test_dataset:
        utt_id = datum["utt_id"]
        text = paddle.to_tensor(datum["text"])

        with paddle.no_grad():
            mel = transformer_tts_inference(text)
            # mel shape is (T, feats) and waveflow's input shape is (batch, feats, T)
            mel = mel.unsqueeze(0).transpose([0, 2, 1])
            # wavflow's output shape is (B, T)
            wav = vocoder.infer(mel)[0]

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
    parser.add_argument(
        "--waveflow-config", type=str, help="waveflow config file.")
    # not normalize when training waveflow
    parser.add_argument(
        "--waveflow-checkpoint", type=str, help="waveflow checkpoint to load.")
    parser.add_argument(
        "--phones-dict", type=str, default=None, help="phone vocabulary file.")

    parser.add_argument("--test-metadata", type=str, help="test metadata.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")

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
