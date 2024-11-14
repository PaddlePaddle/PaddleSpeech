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

import numpy as np
from paddle import inference
from paddle.audio.datasets import ESC50
from paddle.audio.features import LogMelSpectrogram
from paddleaudio.backends import soundfile_load as load_audio
from scipy.special import softmax

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True, default="./export", help="The directory to static model.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu', 'gcu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--wav", type=str, required=True, help="Audio file to infer.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU/CPU for training.")
parser.add_argument('--use_tensorrt', type=eval, default=False, choices=[True, False], help='Enable to use tensorrt to speed up.')
parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16"], help='The tensorrt precision.')
parser.add_argument('--cpu_threads', type=int, default=10, help='Number of threads to predict when using cpu.')
parser.add_argument('--enable_mkldnn', type=eval, default=False, choices=[True, False], help='Enable to use mkldnn to speed up when using cpu.')
parser.add_argument("--log_dir", type=str, default="./log", help="The path to save log.")
args = parser.parse_args()
# yapf: enable


def extract_features(files: str, **kwargs):
    waveforms = []
    srs = []
    max_length = float('-inf')
    for file in files:
        waveform, sr = load_audio(file)
        max_length = max(max_length, len(waveform))
        waveforms.append(waveform)
        srs.append(sr)

    feats = []
    for i in range(len(waveforms)):
        # padding
        if len(waveforms[i]) < max_length:
            pad_width = max_length - len(waveforms[i])
            waveforms[i] = np.pad(waveforms[i], pad_width=(0, pad_width))

        feature_extractor = LogMelSpectrogram(sr, **kwargs)
        feat = feature_extractor(paddle.to_tensor(waveforms[i]))
        feat = paddle.transpose(feat, perm=[1, 0]).unsqueeze(0)

        feats.append(feat)

    return np.stack(feats, axis=0)


class Predictor(object):
    def __init__(self,
                 model_dir,
                 device="gpu",
                 batch_size=1,
                 use_tensorrt=False,
                 precision="fp32",
                 cpu_threads=10,
                 enable_mkldnn=False):
        self.batch_size = batch_size

        model_file = os.path.join(model_dir, "inference.pdmodel")
        params_file = os.path.join(model_dir, "inference.pdiparams")

        assert os.path.isfile(model_file) and os.path.isfile(
            params_file), 'Please check model and parameter files.'

        config = inference.Config(model_file, params_file)
        if device == "gpu":
            # set GPU configs accordingly
            # such as intialize the gpu memory, enable tensorrt
            config.enable_use_gpu(100, 0)
            precision_map = {
                "fp16": inference.PrecisionType.Half,
                "fp32": inference.PrecisionType.Float32,
            }
            precision_mode = precision_map[precision]

            if use_tensorrt:
                config.enable_tensorrt_engine(
                    max_batch_size=batch_size,
                    min_subgraph_size=30,
                    precision_mode=precision_mode)
        elif device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
            if enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
            config.set_cpu_math_library_num_threads(cpu_threads)
        elif device == "xpu":
            # set XPU configs accordingly
            config.enable_xpu(100)

        config.switch_use_feed_fetch_ops(False)
        self.predictor = inference.create_predictor(config)
        self.input_handles = [
            self.predictor.get_input_handle(name)
            for name in self.predictor.get_input_names()
        ]
        self.output_handle = self.predictor.get_output_handle(
            self.predictor.get_output_names()[0])

    def predict(self, wavs):
        feats = extract_features(wavs)

        self.input_handles[0].copy_from_cpu(feats)
        self.predictor.run()
        logits = self.output_handle.copy_to_cpu()
        probs = softmax(logits, axis=1)
        indices = np.argmax(probs, axis=1)

        return indices


if __name__ == "__main__":
    # Define predictor to do prediction.
    predictor = Predictor(args.model_dir, args.device, args.batch_size,
                          args.use_tensorrt, args.precision, args.cpu_threads,
                          args.enable_mkldnn)

    wavs = [args.wav]

    for i in range(len(wavs)):
        wavs[i] = os.path.abspath(os.path.expanduser(wavs[i]))
        assert os.path.isfile(
            wavs[i]), f'Please check input wave file: {wavs[i]}'

    results = predictor.predict(wavs)
    for idx, wav in enumerate(wavs):
        print(f'Wav: {wav} \t Label: {ESC50.label_list[results[idx]]}')
