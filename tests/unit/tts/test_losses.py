# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np

from paddlespeech.audiotools.core.audio_signal import AudioSignal
from paddlespeech.t2s.modules.losses import GANLoss
from paddlespeech.t2s.modules.losses import MultiScaleSTFTLoss
from paddlespeech.t2s.modules.losses import SISDRLoss


def get_input():
    x = AudioSignal("https://paddlespeech.cdn.bcebos.com/PaddleAudio/en.wav",
                    2_05)
    y = x * 0.01
    return x, y


def test_multi_scale_stft_loss():
    x, y = get_input()
    loss = MultiScaleSTFTLoss()
    pd_loss = loss(x, y)
    assert np.abs(pd_loss.numpy() - 7.562150) < 1e-06


def test_sisdr_loss():
    x, y = get_input()
    loss = SISDRLoss()
    pd_loss = loss(x, y)
    assert np.abs(pd_loss.numpy() - (-145.377640)) < 1e-06


def test_gan_loss():
    class My_discriminator0:
        def __call__(self, x):
            return x.sum()

    class My_discriminator1:
        def __call__(self, x):
            return x * (-0.2)

    x, y = get_input()
    loss = GANLoss(My_discriminator0())
    pd_loss0, pd_loss1 = loss(x, y)
    assert np.abs(pd_loss0.numpy() - (-0.102722)) < 1e-06
    assert np.abs(pd_loss1.numpy() - (-0.001027)) < 1e-06
    loss = GANLoss(My_discriminator1())
    pd_loss0, _ = loss.generator_loss(x, y)
    assert np.abs(pd_loss0.numpy() - 1.000199) < 1e-06
    pd_loss = loss.discriminator_loss(x, y)
    assert np.abs(pd_loss.numpy() - 1.000200) < 1e-06
