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
"""Contains the volume perturb augmentation model."""
import numpy as np

from deepspeech.frontend.augmentor.base import AugmentorBase
from deepspeech.utils.log import Log

logger = Log(__name__).getlog()


class SpecAugmentor(AugmentorBase):
    """Augmentation model for Time warping, Frequency masking, Time masking.

    SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
        https://arxiv.org/abs/1904.08779
        
    SpecAugment on Large Scale Datasets
        https://arxiv.org/abs/1912.05533
    
    """

    def __init__(self,
                 rng,
                 F,
                 T,
                 n_freq_masks,
                 n_time_masks,
                 p=1.0,
                 W=40,
                 adaptive_number_ratio=0,
                 adaptive_size_ratio=0,
                 max_n_time_masks=20):
        """SpecAugment class.
        Args:
            rng (random.Random): random generator object.
            F (int): parameter for frequency masking
            T (int): parameter for time masking
            n_freq_masks (int): number of frequency masks
            n_time_masks (int): number of time masks
            p (float): parameter for upperbound of the time mask
            W (int): parameter for time warping
            adaptive_number_ratio (float): adaptive multiplicity ratio for time masking
            adaptive_size_ratio (float): adaptive size ratio for time masking
            max_n_time_masks (int): maximum number of time masking
        """
        super().__init__()
        self._rng = rng

        self.W = W
        self.F = F
        self.T = T
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p

        # adaptive SpecAugment
        self.adaptive_number_ratio = adaptive_number_ratio
        self.adaptive_size_ratio = adaptive_size_ratio
        self.max_n_time_masks = max_n_time_masks

        if adaptive_number_ratio > 0:
            self.n_time_masks = 0
            logger.info('n_time_masks is set ot zero for adaptive SpecAugment.')
        if adaptive_size_ratio > 0:
            self.T = 0
            logger.info('T is set to zero for adaptive SpecAugment.')

        self._freq_mask = None
        self._time_mask = None

    def librispeech_basic(self):
        self.W = 80
        self.F = 27
        self.T = 100
        self.n_freq_masks = 1
        self.n_time_masks = 1
        self.p = 1.0

    def librispeech_double(self):
        self.W = 80
        self.F = 27
        self.T = 100
        self.n_freq_masks = 2
        self.n_time_masks = 2
        self.p = 1.0

    def switchboard_mild(self):
        self.W = 40
        self.F = 15
        self.T = 70
        self.n_freq_masks = 2
        self.n_time_masks = 2
        self.p = 0.2

    def switchboard_strong(self):
        self.W = 40
        self.F = 27
        self.T = 70
        self.n_freq_masks = 2
        self.n_time_masks = 2
        self.p = 0.2

    @property
    def freq_mask(self):
        return self._freq_mask

    @property
    def time_mask(self):
        return self._time_mask

    def __repr__(self):
        return f"specaug: F-{F}, T-{T}, F-n-{n_freq_masks}, T-n-{n_time_masks}"

    def time_warp(xs, W=40):
        raise NotImplementedError

    def mask_freq(self, xs, replace_with_zero=False):
        n_bins = xs.shape[0]
        for i in range(0, self.n_freq_masks):
            f = int(self._rng.uniform(low=0, high=self.F))
            f_0 = int(self._rng.uniform(low=0, high=n_bins - f))
            xs[f_0:f_0 + f, :] = 0
            assert f_0 <= f_0 + f
            self._freq_mask = (f_0, f_0 + f)
        return xs

    def mask_time(self, xs, replace_with_zero=False):
        n_frames = xs.shape[1]

        if self.adaptive_number_ratio > 0:
            n_masks = int(n_frames * self.adaptive_number_ratio)
            n_masks = min(n_masks, self.max_n_time_masks)
        else:
            n_masks = self.n_time_masks

        if self.adaptive_size_ratio > 0:
            T = self.adaptive_size_ratio * n_frames
        else:
            T = self.T

        for i in range(n_masks):
            t = int(self._rng.uniform(low=0, high=T))
            t = min(t, int(n_frames * self.p))
            t_0 = int(self._rng.uniform(low=0, high=n_frames - t))
            xs[:, t_0:t_0 + t] = 0
            assert t_0 <= t_0 + t
            self._time_mask = (t_0, t_0 + t)
        return xs

    def __call__(self, x, train=True):
        if not train:
            return
        return self.transform_feature(x)

    def transform_feature(self, xs: np.ndarray):
        """
        Args:
            xs (FloatTensor): `[F, T]`
        Returns:
            xs (FloatTensor): `[F, T]`
        """
        # xs = self.time_warp(xs)
        xs = self.mask_freq(xs)
        xs = self.mask_time(xs)
        return xs
