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
import unittest

import numpy as np
import paddle
import paddleaudio
import torch
import torchaudio

from .base import FeatTest


class TestKaldi(FeatTest):
    def initParmas(self):
        self.window_size = 1024
        self.dtype = 'float32'

    def test_window(self):
        t_hann_window = torch.hann_window(
            self.window_size, periodic=False, dtype=eval(f'torch.{self.dtype}'))
        t_hamm_window = torch.hamming_window(
            self.window_size,
            periodic=False,
            alpha=0.54,
            beta=0.46,
            dtype=eval(f'torch.{self.dtype}'))
        t_povey_window = torch.hann_window(
            self.window_size, periodic=False,
            dtype=eval(f'torch.{self.dtype}')).pow(0.85)

        p_hann_window = paddleaudio.functional.window.get_window(
            'hann',
            self.window_size,
            fftbins=False,
            dtype=eval(f'paddle.{self.dtype}'))
        p_hamm_window = paddleaudio.functional.window.get_window(
            'hamming',
            self.window_size,
            fftbins=False,
            dtype=eval(f'paddle.{self.dtype}'))
        p_povey_window = paddleaudio.functional.window.get_window(
            'hann',
            self.window_size,
            fftbins=False,
            dtype=eval(f'paddle.{self.dtype}')).pow(0.85)

        np.testing.assert_array_almost_equal(t_hann_window, p_hann_window)
        np.testing.assert_array_almost_equal(t_hamm_window, p_hamm_window)
        np.testing.assert_array_almost_equal(t_povey_window, p_povey_window)

    def test_fbank(self):
        ta_features = torchaudio.compliance.kaldi.fbank(
            torch.from_numpy(self.waveform.astype(self.dtype)))
        pa_features = paddleaudio.compliance.kaldi.fbank(
            paddle.to_tensor(self.waveform.astype(self.dtype)))
        np.testing.assert_array_almost_equal(
            ta_features, pa_features, decimal=4)

    def test_mfcc(self):
        ta_features = torchaudio.compliance.kaldi.mfcc(
            torch.from_numpy(self.waveform.astype(self.dtype)))
        pa_features = paddleaudio.compliance.kaldi.mfcc(
            paddle.to_tensor(self.waveform.astype(self.dtype)))
        np.testing.assert_array_almost_equal(
            ta_features, pa_features, decimal=4)


if __name__ == '__main__':
    unittest.main()
