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
import random

import numpy as np
from PIL import Image
from PIL.Image import BICUBIC

from paddlespeech.s2t.frontend.augmentor.base import AugmentorBase
from paddlespeech.s2t.utils.log import Log

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
                 max_n_time_masks=20,
                 replace_with_zero=True,
                 warp_mode='PIL'):
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
            replace_with_zero (bool): pad zero on mask if true else use mean
            warp_mode (str):  "PIL" (default, fast, not differentiable)
                 or "sparse_image_warp" (slow, differentiable)
        """
        super().__init__()
        self._rng = rng
        self.inplace = True
        self.replace_with_zero = replace_with_zero

        self.mode = warp_mode
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
        return f"specaug: F-{self.F}, T-{self.T}, F-n-{self.n_freq_masks}, T-n-{self.n_time_masks}"

    def time_warp(self, x, mode='PIL'):
        """time warp for spec augment
        move random center frame by the random width ~ uniform(-window, window)

        Args:
            x (np.ndarray): spectrogram (time, freq)
            mode (str): PIL or sparse_image_warp

        Raises:
            NotImplementedError: [description]
            NotImplementedError: [description]

        Returns:
            np.ndarray: time warped spectrogram (time, freq)
        """
        window = max_time_warp = self.W
        if window == 0:
            return x

        if mode == "PIL":
            t = x.shape[0]
            if t - window <= window:
                return x
            # NOTE: randrange(a, b) emits a, a + 1, ..., b - 1
            center = random.randrange(window, t - window)
            warped = random.randrange(center - window, center +
                                      window) + 1  # 1 ... t - 1

            left = Image.fromarray(x[:center]).resize((x.shape[1], warped),
                                                      BICUBIC)
            right = Image.fromarray(x[center:]).resize((x.shape[1], t - warped),
                                                       BICUBIC)
            if self.inplace:
                x[:warped] = left
                x[warped:] = right
                return x
            return np.concatenate((left, right), 0)
        elif mode == "sparse_image_warp":
            raise NotImplementedError('sparse_image_warp')
        else:
            raise NotImplementedError(
                "unknown resize mode: " + mode +
                ", choose one from (PIL, sparse_image_warp).")

    def mask_freq(self, x, replace_with_zero=False):
        """freq mask

        Args:
            x (np.ndarray): spectrogram (time, freq)
            replace_with_zero (bool, optional): Defaults to False.

        Returns:
            np.ndarray: freq mask spectrogram (time, freq)
        """
        n_bins = x.shape[1]
        for i in range(0, self.n_freq_masks):
            f = int(self._rng.uniform(low=0, high=self.F))
            f_0 = int(self._rng.uniform(low=0, high=n_bins - f))
            assert f_0 <= f_0 + f
            if replace_with_zero:
                x[:, f_0:f_0 + f] = 0
            else:
                x[:, f_0:f_0 + f] = x.mean()
            self._freq_mask = (f_0, f_0 + f)
        return x

    def mask_time(self, x, replace_with_zero=False):
        """time mask

        Args:
            x (np.ndarray): spectrogram (time, freq)
            replace_with_zero (bool, optional): Defaults to False.

        Returns:
            np.ndarray: time mask spectrogram (time, freq)
        """
        n_frames = x.shape[0]

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
            assert t_0 <= t_0 + t
            if replace_with_zero:
                x[t_0:t_0 + t, :] = 0
            else:
                x[t_0:t_0 + t, :] = x.mean()
            self._time_mask = (t_0, t_0 + t)
        return x

    def __call__(self, x, train=True):
        if not train:
            return x
        return self.transform_feature(x)

    def transform_feature(self, x: np.ndarray):
        """
        Args:
            x (np.ndarray): `[T, F]`
        Returns:
            x (np.ndarray): `[T, F]`
        """
        assert isinstance(x, np.ndarray)
        assert x.ndim == 2
        x = self.time_warp(x, self.mode)
        x = self.mask_freq(x, self.replace_with_zero)
        x = self.mask_time(x, self.replace_with_zero)
        return x
