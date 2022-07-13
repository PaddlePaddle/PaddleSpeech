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

import paddlespeech.audio.kaldi.fbank as fbank
import paddlespeech.audio.kaldi.pitch as pitch
from kaldiio import ReadHelper

# the groundtruth feats computed in kaldi command below.
#compute-fbank-feats  --dither=0 scp:$wav_scp ark,t:fbank_feat.ark
#compute-kaldi-pitch-feats --sample-frequency=16000 scp:$wav_scp ark,t:pitch_feat.ark

class TestKaldiFbank(unittest.TestCase):

    def test_fbank(self):
        fbank_groundtruth = {}
        with ReadHelper('ark:testdata/fbank_feat.ark') as reader:
            for key, feat in reader:
                fbank_groundtruth[key] = feat

        with ReadHelper('ark:testdata/wav.ark') as reader:
            for key, wav in reader:
                fbank_feat = fbank(wav)
                fbank_check = fbank_groundtruth[key]
                np.testing.assert_array_almost_equal(
                    fbank_feat, fbank_check, decimal=4)

    def test_pitch(self):
        pitch_groundtruth = {}
        with ReadHelper('ark:testdata/pitch_feat.ark') as reader:
           for key, feat in reader:
               pitch_groundtruth[key] = feat

        with ReadHelper('ark:testdata/wav.ark') as reader:
            for key, wav in reader:
                pitch_feat = pitch(wav)
                pitch_check = pitch_groundtruth[key]
                np.testing.assert_array_almost_equal(
                    pitch_feat, pitch_check, decimal=4)



if __name__ == '__main__':
    unittest.main()
