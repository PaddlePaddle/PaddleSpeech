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
# Modified from espnet(https://github.com/espnet/espnet)
import librosa
import numpy as np
import paddle
from python_speech_features import logfbank

from ..compliance import kaldi


def stft(x,
         n_fft,
         n_shift,
         win_length=None,
         window="hann",
         center=True,
         pad_mode="reflect"):
    # x: [Time, Channel]
    if x.ndim == 1:
        single_channel = True
        # x: [Time] -> [Time, Channel]
        x = x[:, None]
    else:
        single_channel = False
    x = x.astype(np.float32)

    # FIXME(kamo): librosa.stft can't use multi-channel?
    # x: [Time, Channel, Freq]
    x = np.stack(
        [
            librosa.stft(
                y=x[:, ch],
                n_fft=n_fft,
                hop_length=n_shift,
                win_length=win_length,
                window=window,
                center=center,
                pad_mode=pad_mode, ).T for ch in range(x.shape[1])
        ],
        axis=1, )

    if single_channel:
        # x: [Time, Channel, Freq] -> [Time, Freq]
        x = x[:, 0]
    return x


def istft(x, n_shift, win_length=None, window="hann", center=True):
    # x: [Time, Channel, Freq]
    if x.ndim == 2:
        single_channel = True
        # x: [Time, Freq] -> [Time, Channel, Freq]
        x = x[:, None, :]
    else:
        single_channel = False

    # x: [Time, Channel]
    x = np.stack(
        [
            librosa.istft(
                stft_matrix=x[:, ch].T,  # [Time, Freq] -> [Freq, Time]
                hop_length=n_shift,
                win_length=win_length,
                window=window,
                center=center, ) for ch in range(x.shape[1])
        ],
        axis=1, )

    if single_channel:
        # x: [Time, Channel] -> [Time]
        x = x[:, 0]
    return x


def stft2logmelspectrogram(x_stft,
                           fs,
                           n_mels,
                           n_fft,
                           fmin=None,
                           fmax=None,
                           eps=1e-10):
    # x_stft: (Time, Channel, Freq) or (Time, Freq)
    fmin = 0 if fmin is None else fmin
    fmax = fs / 2 if fmax is None else fmax

    # spc: (Time, Channel, Freq) or (Time, Freq)
    spc = np.abs(x_stft)
    # mel_basis: (Mel_freq, Freq)
    mel_basis = librosa.filters.mel(
        sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    # lmspc: (Time, Channel, Mel_freq) or (Time, Mel_freq)
    lmspc = np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))

    return lmspc


def spectrogram(x, n_fft, n_shift, win_length=None, window="hann"):
    # x: (Time, Channel) -> spc: (Time, Channel, Freq)
    spc = np.abs(stft(x, n_fft, n_shift, win_length, window=window))
    return spc


def logmelspectrogram(
        x,
        fs,
        n_mels,
        n_fft,
        n_shift,
        win_length=None,
        window="hann",
        fmin=None,
        fmax=None,
        eps=1e-10,
        pad_mode="reflect", ):
    # stft: (Time, Channel, Freq) or (Time, Freq)
    x_stft = stft(
        x,
        n_fft=n_fft,
        n_shift=n_shift,
        win_length=win_length,
        window=window,
        pad_mode=pad_mode, )

    return stft2logmelspectrogram(
        x_stft,
        fs=fs,
        n_mels=n_mels,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax,
        eps=eps)


class Spectrogram():
    def __init__(self, n_fft, n_shift, win_length=None, window="hann"):
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window

    def __repr__(self):
        return ("{name}(n_fft={n_fft}, n_shift={n_shift}, "
                "win_length={win_length}, window={window})".format(
                    name=self.__class__.__name__,
                    n_fft=self.n_fft,
                    n_shift=self.n_shift,
                    win_length=self.win_length,
                    window=self.window, ))

    def __call__(self, x):
        return spectrogram(
            x,
            n_fft=self.n_fft,
            n_shift=self.n_shift,
            win_length=self.win_length,
            window=self.window, )


class LogMelSpectrogram():
    def __init__(
            self,
            fs,
            n_mels,
            n_fft,
            n_shift,
            win_length=None,
            window="hann",
            fmin=None,
            fmax=None,
            eps=1e-10, ):
        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window
        self.fmin = fmin
        self.fmax = fmax
        self.eps = eps

    def __repr__(self):
        return ("{name}(fs={fs}, n_mels={n_mels}, n_fft={n_fft}, "
                "n_shift={n_shift}, win_length={win_length}, window={window}, "
                "fmin={fmin}, fmax={fmax}, eps={eps}))".format(
                    name=self.__class__.__name__,
                    fs=self.fs,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    n_shift=self.n_shift,
                    win_length=self.win_length,
                    window=self.window,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    eps=self.eps, ))

    def __call__(self, x):
        return logmelspectrogram(
            x,
            fs=self.fs,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            n_shift=self.n_shift,
            win_length=self.win_length,
            window=self.window, )


class Stft2LogMelSpectrogram():
    def __init__(self, fs, n_mels, n_fft, fmin=None, fmax=None, eps=1e-10):
        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.eps = eps

    def __repr__(self):
        return ("{name}(fs={fs}, n_mels={n_mels}, n_fft={n_fft}, "
                "fmin={fmin}, fmax={fmax}, eps={eps}))".format(
                    name=self.__class__.__name__,
                    fs=self.fs,
                    n_mels=self.n_mels,
                    n_fft=self.n_fft,
                    fmin=self.fmin,
                    fmax=self.fmax,
                    eps=self.eps, ))

    def __call__(self, x):
        return stft2logmelspectrogram(
            x,
            fs=self.fs,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            fmin=self.fmin,
            fmax=self.fmax, )


class Stft():
    def __init__(
            self,
            n_fft,
            n_shift,
            win_length=None,
            window="hann",
            center=True,
            pad_mode="reflect", ):
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

    def __repr__(self):
        return ("{name}(n_fft={n_fft}, n_shift={n_shift}, "
                "win_length={win_length}, window={window},"
                "center={center}, pad_mode={pad_mode})".format(
                    name=self.__class__.__name__,
                    n_fft=self.n_fft,
                    n_shift=self.n_shift,
                    win_length=self.win_length,
                    window=self.window,
                    center=self.center,
                    pad_mode=self.pad_mode, ))

    def __call__(self, x):
        return stft(
            x,
            self.n_fft,
            self.n_shift,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode, )


class IStft():
    def __init__(self, n_shift, win_length=None, window="hann", center=True):
        self.n_shift = n_shift
        self.win_length = win_length
        self.window = window
        self.center = center

    def __repr__(self):
        return ("{name}(n_shift={n_shift}, "
                "win_length={win_length}, window={window},"
                "center={center})".format(
                    name=self.__class__.__name__,
                    n_shift=self.n_shift,
                    win_length=self.win_length,
                    window=self.window,
                    center=self.center, ))

    def __call__(self, x):
        return istft(
            x,
            self.n_shift,
            win_length=self.win_length,
            window=self.window,
            center=self.center, )


class LogMelSpectrogramKaldi():
    def __init__(
            self,
            fs=16000,
            n_mels=80,
            n_shift=160,  # unit:sample, 10ms
            win_length=400,  # unit:sample, 25ms
            energy_floor=0.0,
            dither=0.1):
        """
        The Kaldi implementation of LogMelSpectrogram 
        Args:
            fs (int): sample rate of the audio
            n_mels (int): number of mel filter banks
            n_shift (int): number of points in a frame shift
            win_length (int): number of points in a frame windows
            energy_floor (float): Floor on energy in Spectrogram computation (absolute)
            dither (float): Dithering constant

        Returns:
            LogMelSpectrogramKaldi
        """

        self.fs = fs
        self.n_mels = n_mels
        num_point_ms = fs / 1000
        self.n_frame_length = win_length / num_point_ms
        self.n_frame_shift = n_shift / num_point_ms
        self.energy_floor = energy_floor
        self.dither = dither

    def __repr__(self):
        return (
            "{name}(fs={fs}, n_mels={n_mels}, "
            "n_frame_shift={n_frame_shift}, n_frame_length={n_frame_length}, "
            "dither={dither}))".format(
                name=self.__class__.__name__,
                fs=self.fs,
                n_mels=self.n_mels,
                n_frame_shift=self.n_frame_shift,
                n_frame_length=self.n_frame_length,
                dither=self.dither, ))

    def __call__(self, x, train):
        """
        Args:
            x (np.ndarray): shape (Ti,)
            train (bool): True, train mode.

        Raises:
            ValueError: not support (Ti, C)

        Returns:
            np.ndarray: (T, D)
        """
        dither = self.dither if train else 0.0
        if x.ndim != 1:
            raise ValueError("Not support x: [Time, Channel]")
        waveform = paddle.to_tensor(np.expand_dims(x, 0), dtype=paddle.float32)
        mat = kaldi.fbank(
            waveform,
            n_mels=self.n_mels,
            frame_length=self.n_frame_length,
            frame_shift=self.n_frame_shift,
            dither=dither,
            energy_floor=self.energy_floor,
            sr=self.fs)
        mat = np.squeeze(mat.numpy())
        return mat


class WavProcess():
    def __init__(self):
        """
        Args:
            dither (float): Dithering constant

        Returns:
        """

    def __call__(self, x):
        """
        Args:
            x (np.ndarray): shape (Ti,)
            train (bool): True, train mode.

        Raises:
            ValueError: not support (Ti, C)

        Returns:
            np.ndarray: (T, D)
        """
        if x.ndim != 1:
            raise ValueError("Not support x: [Time, Channel]")
        waveform = x.astype("float32") / 32768.0
        waveform = np.expand_dims(waveform, -1)
        return waveform


class LogMelSpectrogramKaldi_decay():
    def __init__(
            self,
            fs=16000,
            n_mels=80,
            n_fft=512,  # fft point
            n_shift=160,  # unit:sample, 10ms
            win_length=400,  # unit:sample, 25ms
            window="povey",
            fmin=20,
            fmax=None,
            eps=1e-10,
            dither=1.0):
        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        if n_shift > win_length:
            raise ValueError("Stride size must not be greater than "
                             "window size.")
        self.n_shift = n_shift / fs  # unit: ms
        self.win_length = win_length / fs  # unit: ms

        self.window = window
        self.fmin = fmin
        if fmax is None:
            fmax_ = fmax if fmax else self.fs / 2
        elif fmax > int(self.fs / 2):
            raise ValueError("fmax must not be greater than half of "
                             "sample rate.")
        self.fmax = fmax_

        self.eps = eps
        self.remove_dc_offset = True
        self.preemph = 0.97
        self.dither = dither  # only work in train mode

    def __repr__(self):
        return (
            "{name}(fs={fs}, n_mels={n_mels}, n_fft={n_fft}, "
            "n_shift={n_shift}, win_length={win_length}, preemph={preemph}, window={window}, "
            "fmin={fmin}, fmax={fmax}, eps={eps}, dither={dither}))".format(
                name=self.__class__.__name__,
                fs=self.fs,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                n_shift=self.n_shift,
                preemph=self.preemph,
                win_length=self.win_length,
                window=self.window,
                fmin=self.fmin,
                fmax=self.fmax,
                eps=self.eps,
                dither=self.dither, ))

    def __call__(self, x, train):
        """

        Args:
            x (np.ndarray): shape (Ti,)
            train (bool): True, train mode.

        Raises:
            ValueError: not support (Ti, C)

        Returns:
            np.ndarray: (T, D)
        """
        dither = self.dither if train else 0.0
        if x.ndim != 1:
            raise ValueError("Not support x: [Time, Channel]")

        if x.dtype in np.sctypes['float']:
            # PCM32 -> PCM16
            bits = np.iinfo(np.int16).bits
            x = x * 2**(bits - 1)

        # logfbank need PCM16 input
        y = logfbank(
            signal=x,
            samplerate=self.fs,
            winlen=self.win_length,  # unit ms
            winstep=self.n_shift,  # unit ms
            nfilt=self.n_mels,
            nfft=self.n_fft,
            lowfreq=self.fmin,
            highfreq=self.fmax,
            dither=dither,
            remove_dc_offset=self.remove_dc_offset,
            preemph=self.preemph,
            wintype=self.window)
        return y
