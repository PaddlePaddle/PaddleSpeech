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

from .base import FeatTest
from paddlespeech.s2t.transform.spectrogram import Spectrogram


class TestSpectrogram(FeatTest):
    def initParmas(self):
        self.n_fft = 512
        self.hop_length = 128

    def test_spectrogram(self):
        ps_spect = Spectrogram(self.n_fft, self.hop_length)
        ps_res = ps_spect(self.waveform.T).squeeze(1).T  # Magnitude

        x = paddle.to_tensor(self.waveform)
        pa_spect = paddleaudio.features.Spectrogram(
            self.n_fft, self.hop_length, power=1.0)
        pa_res = pa_spect(x).squeeze(0).numpy()

        np.testing.assert_array_almost_equal(ps_res, pa_res, decimal=5)


if __name__ == '__main__':
    unittest.main()
