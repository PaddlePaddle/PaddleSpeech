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
import filecmp
import os
import unittest

import numpy as np
import paddleaudio
import soundfile as sf

from ..base import BackendTest


class TestIO(BackendTest):
    def test_load_mono_channel(self):
        sf_data, sf_sr = sf.read(self.files[0])
        pa_data, pa_sr = paddleaudio.load(
            self.files[0], normal=False, dtype='float64')

        self.assertEqual(sf_data.dtype, pa_data.dtype)
        self.assertEqual(sf_sr, pa_sr)
        np.testing.assert_array_almost_equal(sf_data, pa_data)

    def test_load_multi_channels(self):
        sf_data, sf_sr = sf.read(self.files[1])
        sf_data = sf_data.T  # Channel dim first
        pa_data, pa_sr = paddleaudio.load(
            self.files[1], mono=False, normal=False, dtype='float64')

        self.assertEqual(sf_data.dtype, pa_data.dtype)
        self.assertEqual(sf_sr, pa_sr)
        np.testing.assert_array_almost_equal(sf_data, pa_data)

    def test_save_mono_channel(self):
        waveform, sr = np.random.randint(
            low=-32768, high=32768, size=(48000), dtype=np.int16), 16000
        sf_tmp_file = 'sf_tmp.wav'
        pa_tmp_file = 'pa_tmp.wav'

        sf.write(sf_tmp_file, waveform, sr)
        paddleaudio.save(waveform, sr, pa_tmp_file)

        self.assertTrue(filecmp.cmp(sf_tmp_file, pa_tmp_file))
        for file in [sf_tmp_file, pa_tmp_file]:
            os.remove(file)

    def test_save_multi_channels(self):
        waveform, sr = np.random.randint(
            low=-32768, high=32768, size=(2, 48000), dtype=np.int16), 16000
        sf_tmp_file = 'sf_tmp.wav'
        pa_tmp_file = 'pa_tmp.wav'

        sf.write(sf_tmp_file, waveform.T, sr)
        paddleaudio.save(waveform.T, sr, pa_tmp_file)

        self.assertTrue(filecmp.cmp(sf_tmp_file, pa_tmp_file))
        for file in [sf_tmp_file, pa_tmp_file]:
            os.remove(file)


if __name__ == '__main__':
    unittest.main()
