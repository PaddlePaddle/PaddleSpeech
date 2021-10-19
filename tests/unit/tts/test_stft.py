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
import librosa
import numpy as np
import paddle
import torch
from parallel_wavegan.losses import stft_loss as sl
from scipy import signal

from parakeet.modules.stft_loss import MultiResolutionSTFTLoss
from parakeet.modules.stft_loss import STFT


def test_stft():
    stft = STFT(n_fft=1024, hop_length=256, win_length=1024)
    x = paddle.uniform([4, 46080])
    S = stft.magnitude(x)
    window = signal.get_window('hann', 1024, fftbins=True)
    D2 = torch.stft(
        torch.as_tensor(x.numpy()),
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=torch.as_tensor(window))
    S2 = (D2**2).sum(-1).sqrt()
    S3 = np.abs(
        librosa.stft(x.numpy()[0], n_fft=1024, hop_length=256, win_length=1024))
    print(S2.shape)
    print(S.numpy()[0])
    print(S2.data.cpu().numpy()[0])
    print(S3)


def test_torch_stft():
    # NOTE: torch.stft use no window by default
    x = np.random.uniform(-1.0, 1.0, size=(46080, ))
    window = signal.get_window('hann', 1024, fftbins=True)
    D2 = torch.stft(
        torch.as_tensor(x),
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        window=torch.as_tensor(window))
    D3 = librosa.stft(
        x, n_fft=1024, hop_length=256, win_length=1024, window='hann')
    print(D2[:, :, 0].data.cpu().numpy()[:, 30:60])
    print(D3.real[:, 30:60])
    # print(D3.imag[:, 30:60])


def test_multi_resolution_stft_loss():
    net = MultiResolutionSTFTLoss()
    net2 = sl.MultiResolutionSTFTLoss()

    x = paddle.uniform([4, 46080])
    y = paddle.uniform([4, 46080])
    sc, m = net(x, y)
    sc2, m2 = net2(torch.as_tensor(x.numpy()), torch.as_tensor(y.numpy()))
    print(sc.numpy())
    print(sc2.data.cpu().numpy())
    print(m.numpy())
    print(m2.data.cpu().numpy())
