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
from typing import Union

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


class StyleFastSpeech2Inference(FastSpeech2Inference):
    def __init__(self, normalizer, model, pitch_stats_path, energy_stats_path):
        super().__init__(normalizer, model)
        pitch_mean, pitch_std = np.load(pitch_stats_path)
        self.pitch_mean = paddle.to_tensor(pitch_mean)
        self.pitch_std = paddle.to_tensor(pitch_std)
        energy_mean, energy_std = np.load(energy_stats_path)
        self.energy_mean = paddle.to_tensor(energy_mean)
        self.energy_std = paddle.to_tensor(energy_std)

    def denorm(self, data, mean, std):
        return data * std + mean

    def norm(self, data, mean, std):
        return (data - mean) / std

    def forward(self,
                text: paddle.Tensor,
                durations: Union[paddle.Tensor, np.ndarray]=None,
                durations_scale: Union[int, float]=None,
                durations_bias: Union[int, float]=None,
                pitch: Union[paddle.Tensor, np.ndarray]=None,
                pitch_scale: Union[int, float]=None,
                pitch_bias: Union[int, float]=None,
                energy: Union[paddle.Tensor, np.ndarray]=None,
                energy_scale: Union[int, float]=None,
                energy_bias: Union[int, float]=None,
                robot: bool=False):
        """
        Parameters
        ----------
        text : Tensor(int64)
            Input sequence of characters (T,).
        speech : Tensor, optional
            Feature sequence to extract style (N, idim).
        durations : paddle.Tensor/np.ndarray, optional (int64)
            Groundtruth of duration (T,), this will overwrite the set of durations_scale and durations_bias
        durations_scale: int/float, optional
        durations_bias: int/float, optional
        pitch : paddle.Tensor/np.ndarray, optional
            Groundtruth of token-averaged pitch (T, 1), this will overwrite the set of pitch_scale and pitch_bias
        pitch_scale: int/float, optional
            In denormed HZ domain.
        pitch_bias: int/float, optional
            In denormed HZ domain.
        energy : paddle.Tensor/np.ndarray, optional
            Groundtruth of token-averaged energy (T, 1), this will overwrite the set of energy_scale and energy_bias
        energy_scale: int/float, optional
            In denormed domain.
        energy_bias: int/float, optional
            In denormed domain.
        robot : bool, optional
            Weather output robot style
        Returns
        ----------
        Tensor
            Output sequence of features (L, odim).
        """
        normalized_mel, d_outs, p_outs, e_outs = self.acoustic_model.inference(
            text, durations=None, pitch=None, energy=None)
        # priority: groundtruth > scale/bias > previous output
        # set durations
        if isinstance(durations, np.ndarray):
            durations = paddle.to_tensor(durations)
        elif isinstance(durations, paddle.Tensor):
            durations = durations
        elif durations_scale or durations_bias:
            durations_scale = durations_scale if durations_scale is not None else 1
            durations_bias = durations_bias if durations_bias is not None else 0
            durations = durations_scale * d_outs + durations_bias
        else:
            durations = d_outs

        if robot:
            # set normed pitch to zeros have the same effect with set denormd ones to mean
            pitch = paddle.zeros(p_outs.shape)

        # set pitch, can overwrite robot set  
        if isinstance(pitch, np.ndarray):
            pitch = paddle.to_tensor(pitch)
        elif isinstance(pitch, paddle.Tensor):
            pitch = pitch
        elif pitch_scale or pitch_bias:
            pitch_scale = pitch_scale if pitch_scale is not None else 1
            pitch_bias = pitch_bias if pitch_bias is not None else 0
            p_Hz = paddle.exp(
                self.denorm(p_outs, self.pitch_mean, self.pitch_std))
            p_HZ = pitch_scale * p_Hz + pitch_bias
            pitch = self.norm(paddle.log(p_HZ), self.pitch_mean, self.pitch_std)
        else:
            pitch = p_outs

        # set energy
        if isinstance(energy, np.ndarray):
            energy = paddle.to_tensor(energy)
        elif isinstance(energy, paddle.Tensor):
            energy = energy
        elif energy_scale or energy_bias:
            energy_scale = energy_scale if energy_scale is not None else 1
            energy_bias = energy_bias if energy_bias is not None else 0
            e_dnorm = self.denorm(e_outs, self.energy_mean, self.energy_std)
            e_dnorm = energy_scale * e_dnorm + energy_bias
            energy = self.norm(e_dnorm, self.energy_mean, self.energy_std)
        else:
            energy = e_outs

        normalized_mel, d_outs, p_outs, e_outs = self.acoustic_model.inference(
            text,
            durations=durations,
            pitch=pitch,
            energy=energy,
            use_teacher_forcing=True)

        logmel = self.normalizer.inverse(normalized_mel)
        return logmel


def evaluate(args, fastspeech2_config, pwg_config):

    # construct dataset for evaluation
    sentences = []
    with open(args.text, 'rt') as f:
        for line in f:
            utt_id, sentence = line.strip().split()
            sentences.append((utt_id, sentence))

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

    fastspeech2_inference = StyleFastSpeech2Inference(
        fastspeech2_normalizer, model, args.fastspeech2_pitch_stat,
        args.fastspeech2_energy_stat)
    fastspeech2_inference.eval()

    pwg_inference = PWGInference(pwg_normalizer, vocoder)
    pwg_inference.eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    styles = ["normal", "robot", "1.2xspeed", "0.8xspeed", "child_voice"]
    for style in styles:
        robot = False
        durations = None
        durations_scale = None
        durations_bias = None
        pitch = None
        pitch_scale = None
        pitch_bias = None
        energy = None
        energy_scale = None
        energy_bias = None
        if style == "robot":
            # all tones in phones be `1`
            # all pitch should be the same, we use mean here
            robot = True
        if style == "1.2xspeed":
            durations_scale = 1 / 1.2
        if style == "0.8xspeed":
            durations_scale = 1 / 0.8
        if style == "child_voice":
            pitch_scale = 1.3
        sub_output_dir = output_dir / style
        sub_output_dir.mkdir(parents=True, exist_ok=True)
        for utt_id, sentence in sentences:
            input_ids = frontend.get_input_ids(
                sentence, merge_sentences=True, robot=robot)
            phone_ids = input_ids["phone_ids"][0]

            with paddle.no_grad():
                mel = fastspeech2_inference(
                    phone_ids,
                    durations=durations,
                    durations_scale=durations_scale,
                    durations_bias=durations_bias,
                    pitch=pitch,
                    pitch_scale=pitch_scale,
                    pitch_bias=pitch_bias,
                    energy=energy,
                    energy_scale=energy_scale,
                    energy_bias=energy_bias,
                    robot=robot)
                wav = pwg_inference(mel)

            sf.write(
                str(sub_output_dir / (utt_id + ".wav")),
                wav.numpy(),
                samplerate=fastspeech2_config.fs)
            print(f"{style}_{utt_id} done!")


def main():
    # parse args and config and redirect to train_sp
    parser = argparse.ArgumentParser(
        description="Synthesize with fastspeech2 & parallel wavegan.")
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
        "--fastspeech2-pitch-stat",
        type=str,
        help="mean and standard deviation used to normalize pitch when training fastspeech2"
    )
    parser.add_argument(
        "--fastspeech2-energy-stat",
        type=str,
        help="mean and standard deviation used to normalize energy when training fastspeech2."
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
        help="text to synthesize, a 'utt_id sentence' pair per line.")
    parser.add_argument("--output-dir", type=str, help="output dir.")
    parser.add_argument(
        "--ngpu", type=int, default=1, help="if ngpu == 0, use cpu.")
    parser.add_argument("--verbose", type=int, default=1, help="verbose.")

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

    evaluate(args, fastspeech2_config, pwg_config)


if __name__ == "__main__":
    main()
