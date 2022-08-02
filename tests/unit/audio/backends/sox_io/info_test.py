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

from paddlespeech.audio.backends import sox_io_backend

class TestInfo(unittest.TestCase):

    def test_wav(self, dtype, sample_rate, num_channels, sample_size):
        """check wav file correctly """
        path = 'testdata/test.wav'
        info = sox_io_backend.get_info_file(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_size # duration*sample_rate
        assert info.num_channels == num_channels
        assert info.bits_per_sample == get_bit_depth(dtype)
        assert info.encoding == get_encoding('wav', dtype)
        
if __name__ == '__main__':
    unittest.main()