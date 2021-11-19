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
import os
from pathlib import Path

import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode

from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2Inference
from paddlespeech.t2s.models.parallel_wavegan import PWGGenerator
from paddlespeech.t2s.models.parallel_wavegan import PWGInference
from paddlespeech.t2s.modules.normalizer import ZScore
from paddlespeech.vector.exps.ge2e.audio_processor import SpeakerVerificationPreprocessor
from paddlespeech.vector.models.lstm_speaker_encoder import LSTMSpeakerEncoder


def voice_cloning(args, fastspeech2_config, pwg_config):
    # speaker encoder
    p = SpeakerVerificationPreprocessor(
        sampling_rate=16000,
        audio_norm_target_dBFS=-30,
        vad_window_length=30,
        vad_moving_average_width=8,
        vad_max_silence_length=6,
        mel_window_length=25,
        mel_window_step=10,
        n_mels=40,
        partial_n_frames=160,
        min_pad_coverage=0.75,
        partial_overlap_ratio=0.5)
    print("Audio Processor Done!")

    speaker_encoder = LSTMSpeakerEncoder(
        n_mels=40, num_layers=3, hidden_size=256, output_size=256)
    speaker_encoder.set_state_dict(paddle.load(args.ge2e_params_path))
    speaker_encoder.eval()
    print("GE2E Done!")

    with open(args.phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)
    odim = fastspeech2_config.n_mels
    model = FastSpeech2(
        idim=vocab_size, odim=odim, **fastspeech2_config["model"])

    model.set_state_dict(
        paddle.load(args.fastspeech2_checkpoint)["main_params"])
    model.eval()

    vocoder = PWGGenerator(**pwg_config["generator_params"])
    vocoder.set_state_dict(paddle.load(args.pwg_checkpoint)["generator_params"])
    vocoder.remove_weight_norm()
    vocoder.eval()
    print("model done!")

    frontend = Frontend(phone_vocab_path=args.phones_dict)
    print("frontend done!")

    stat = np.load(args.fastspeech2_stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    fastspeech2_normalizer = ZScore(mu, std)

    stat = np.load(args.pwg_stat)
    mu, std = stat
    mu = paddle.to_tensor(mu)
    std = paddle.to_tensor(std)
    pwg_normalizer = ZScore(mu, std)

    fastspeech2_inference = FastSpeech2Inference(fastspeech2_normalizer, model)
    fastspeech2_inference.eval()
    pwg_inference = PWGInference(pwg_normalizer, vocoder)
    pwg_inference.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)

    sentence = args.text

    input_ids = frontend.get_input_ids(sentence, merge_sentences=True)
    phone_ids = input_ids["phone_ids"][0]

    for name in os.listdir(input_dir):
        utt_id = name.split(".")[0]
        ref_audio_path = input_dir / name
        mel_sequences = p.extract_mel_partials(p.preprocess_wav(ref_audio_path))
        # print("mel_sequences: ", mel_sequences.shape)
        with paddle.no_grad():
            spk_emb = speaker_encoder.embed_utterance(
                paddle.to_tensor(mel_sequences))
        # print("spk_emb shape: ", spk_emb.shape)

        with paddle.no_grad():
            wav = pwg_inference(
                fastspeech2_inference(phone_ids, spk_emb=spk_emb))

        sf.write(
            str(output_dir / (utt_id + ".wav")),
            wav.numpy(),
            samplerate=fastspeech2_config.fs)
        print(f"{utt_id} done!")
    # Randomly generate numbers of 0 ~ 0.2, 256 is the dim of spk_emb
    random_spk_emb = np.random.rand(256) * 0.2
    random_spk_emb = paddle.to_tensor(random_spk_emb)
    utt_id = "random_spk_emb"
    with paddle.no_grad():
        wav = pwg_inference(fastspeech2_inference(phone_ids, spk_emb=spk_emb))
    sf.write(
        str(output_dir / (utt_id + ".wav")),
        wav.numpy(),
        samplerate=fastspeech2_config.fs)
    print(f"{utt_id} done!")


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--fastspeech2-config", type=str, help="fastspeech2 config file.")
    parser.add_argument(
        "--fastspeech2-checkpoint",
        type=str,
        help="fastspeech2 checkpoint to load.")
    parser.add_argument(
        "--fastspeech2-stat",
        type=str,
        help="mean and standard deviation used to normalize spectrogram when training fastspeech2."
    )
    parser.add_argument(
        "--pwg-config", type=str, help="parallel wavegan config file.")
    parser.add_argument(
        "--pwg-checkpoint",
        type=str,
        help="parallel wavegan generator parameters to load.")
    parser.add_argument(
        "--pwg-stat",
        type=str,
        help="mean and standard deviation used to normalize spectrogram when training parallel wavegan."
    )
    parser.add_argument(
        "--phones-dict",
        type=str,
        default="phone_id_map.txt",
        help="phone vocabulary file.")
    parser.add_argument(
        "--text",
        type=str,
        default="每当你觉得，想要批评什么人的时候，你切要记着，这个世界上的人，并非都具备你禀有的条件。",
        help="text to synthesize, a line")

    parser.add_argument(
        "--ge2e_params_path", type=str, help="ge2e params path.")

    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu=0, use cpu.")

    parser.add_argument(
        "--input-dir",
        type=str,
        help="input dir of *.wav, the sample rate will be resample to 16k.")
    parser.add_argument("--output-dir", type=str, help="output dir.")

    args = parser.parse_args()

    if args.ngpu == 0:
        paddle.set_device("cpu")
    elif args.ngpu > 0:
        paddle.set_device("gpu")
    else:
        print("ngpu should >= 0 !")

    with open(args.fastspeech2_config) as f:
        fastspeech2_config = CfgNode(yaml.safe_load(f))
    with open(args.pwg_config) as f:
        pwg_config = CfgNode(yaml.safe_load(f))

    print("========Args========")
    print(yaml.safe_dump(vars(args)))
    print("========Config========")
    print(fastspeech2_config)
    print(pwg_config)

    voice_cloning(args, fastspeech2_config, pwg_config)


if __name__ == "__main__":
    main()
