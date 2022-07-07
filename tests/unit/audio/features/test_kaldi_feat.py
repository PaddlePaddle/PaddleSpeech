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

import paddlespeech.audio.kaldi.kaldi.fbank as fbank
import paddlespeech.audio.kaldi.kaldi.pitch as pitch
import kaldiio import ReadHelper

class TestKaldiFbank(unittest.TestCase):

    def test_fbank_pitch(self):
        fbank_groundtruth = None
        pitch_groundtruth = None
        with ReadHelper('ark:fbank_feat.ark') as reader:
            for key, feat in reader:
                fbank_groundtruth = feat

        with ReadHelper('ark:pitch_feat.ark') as reader:
           for key, feat in reader:
               pitch_groundtruth = feat

        with ReadHelper('ark:wav.ark') as reader:
            for key, wav in reader:
                fbank_feat = fbank(wav)
                pitch_feat = pitch(wav)
                np.testing.assert_array_almost_equal(
                    fbank_feat, fbank_groundtruth, decimal=4)
                np.testing.assert_array_almost_equal(
                    pitch_feat, pitch_groundtruth, decimal=4)

if __name__ == '__main__':
    unittest.main()
