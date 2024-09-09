# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
This modules contains normalizers for spectrogram magnitude.
Normalizers are invertible transformations. They can be used to process 
magnitude of spectrogram before training and can also be used to recover from 
the generated spectrogram so as to be used with vocoders like griffin lim.

The base class describe the interface. `transform` is used to perform 
transformation and `inverse` is used to perform the inverse transformation.

check issues:
https://github.com/mozilla/TTS/issues/377
"""
import numpy as np

__all__ = ["NormalizerBase", "LogMagnitude", "UnitMagnitude"]


class NormalizerBase(object):
    def transform(self, spec):
        raise NotImplementedError("transform must be implemented")

    def inverse(self, normalized):
        raise NotImplementedError("inverse must be implemented")


class LogMagnitude(NormalizerBase):
    """
    This is a simple normalizer used in Waveglow, Waveflow, tacotron2...
    """

    def __init__(self, min=1e-5):
        self.min = min

    def transform(self, x):
        x = np.maximum(x, self.min)
        x = np.log(x)
        return x

    def inverse(self, x):
        return np.exp(x)


class UnitMagnitude(NormalizerBase):
    # dbscale and (0, 1) normalization
    """
    This is the normalizer used in the 
    """

    def __init__(self, min=1e-5):
        self.min = min

    def transform(self, x):
        db_scale = 20 * np.log10(np.maximum(self.min, x)) - 20
        normalized = (db_scale + 100) / 100
        clipped = np.clip(normalized, 0, 1)
        return clipped

    def inverse(self, x):
        denormalized = np.clip(x, 0, 1) * 100 - 100
        out = np.exp((denormalized + 20) / 20 * np.log(10))
        return out
