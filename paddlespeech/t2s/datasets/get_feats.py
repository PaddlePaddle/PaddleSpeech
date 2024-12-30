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
from typing import List
from typing import Optional
from typing import Union

import librosa
import numpy as np
import pyworld
from scipy.interpolate import interp1d
from typing_extensions import Literal


class LogMelFBank():
    def __init__(self,
                 sr: int=24000,
                 n_fft: int=2048,
                 hop_length: int=300,
                 win_length: int=None,
                 window: str="hann",
                 n_mels: int=80,
                 fmin: int=80,
                 fmax: int=7600,
                 norm: Optional[Union[Literal["slaney"], float]]="slaney",
                 htk: bool=False,
                 power: float=1.0):
        self.sr = sr
        # stft
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window
        self.center = True
        self.pad_mode = "reflect"
        self.norm = norm
        self.htk = htk

        # mel
        self.n_mels = n_mels
        self.fmin = 0 if fmin is None else fmin
        self.fmax = sr / 2 if fmax is None else fmax
        self.power = power

        self.mel_filter = self._create_mel_filter()

    def _create_mel_filter(self):
        mel_filter = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            norm=self.norm,
            htk=self.htk)
        return mel_filter

    def _stft(self, wav: np.ndarray):
        D = librosa.core.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode)
        return D

    def _spectrogram(self, wav: np.ndarray):
        D = self._stft(wav)
        return np.abs(D)**self.power

    def _mel_spectrogram(self, wav: np.ndarray):
        S = self._spectrogram(wav)
        mel = np.dot(self.mel_filter, S)
        return mel

    # We use different definition for log-spec between TTS and ASR
    #   TTS: log_10(abs(stft))
    #   ASR: log_e(power(stft))

    def get_log_mel_fbank(self, wav, base='10'):
        mel = self._mel_spectrogram(wav)
        mel = np.clip(mel, a_min=1e-10, a_max=float("inf"))
        if base == '10':
            mel = np.log10(mel.T)
        elif base == 'e':
            mel = np.log(mel.T)
        # (num_frames, n_mels)
        return mel


class Pitch():
    def __init__(self,
                 sr: int=24000,
                 hop_length: int=300,
                 f0min: int=80,
                 f0max: int=7600):

        self.sr = sr
        self.hop_length = hop_length
        self.f0min = f0min
        self.f0max = f0max

    def _convert_to_continuous_f0(self, f0: np.ndarray) -> np.ndarray:
        if (f0 == 0).all():
            print("All frames seems to be unvoiced, this utt will be removed.")
            return f0
        # padding start and end of f0 sequence
        start_f0 = f0[f0 != 0][0]
        end_f0 = f0[f0 != 0][-1]
        start_idx = np.where(f0 == start_f0)[0][0]
        end_idx = np.where(f0 == end_f0)[0][-1]
        f0[:start_idx] = start_f0
        f0[end_idx:] = end_f0

        # get non-zero frame index
        nonzero_idxs = np.where(f0 != 0)[0]

        # perform linear interpolation
        interp_fn = interp1d(nonzero_idxs, f0[nonzero_idxs])
        f0 = interp_fn(np.arange(0, f0.shape[0]))

        return f0

    def _calculate_f0(self,
                      input: np.ndarray,
                      use_continuous_f0: bool=True,
                      use_log_f0: bool=True) -> np.ndarray:
        input = input.astype(np.float_)
        frame_period = 1000 * self.hop_length / self.sr
        f0, timeaxis = pyworld.dio(
            input,
            fs=self.sr,
            f0_floor=self.f0min,
            f0_ceil=self.f0max,
            frame_period=frame_period)
        f0 = pyworld.stonemask(input, f0, timeaxis, self.sr)
        if use_continuous_f0:
            f0 = self._convert_to_continuous_f0(f0)
        if use_log_f0:
            nonzero_idxs = np.where(f0 != 0)[0]
            f0[nonzero_idxs] = np.log(f0[nonzero_idxs])
        return f0.reshape(-1)

    def _average_by_duration(self, input: np.ndarray,
                             d: np.ndarray) -> np.ndarray:
        d_cumsum = np.pad(d.cumsum(0), (1, 0), 'constant')
        arr_list = []
        for start, end in zip(d_cumsum[:-1], d_cumsum[1:]):
            arr = input[start:end]
            mask = arr == 0
            arr[mask] = 0
            avg_arr = np.mean(arr, axis=0) if len(arr) != 0 else np.array(0)
            arr_list.append(avg_arr)
        # shape (T,1)
        arr_list = np.expand_dims(np.array(arr_list), 0).T

        return arr_list

    def get_pitch(self,
                  wav: np.ndarray,
                  use_continuous_f0: bool=True,
                  use_log_f0: bool=True,
                  use_token_averaged_f0: bool=True,
                  duration: np.ndarray=None):
        f0 = self._calculate_f0(wav, use_continuous_f0, use_log_f0)
        if use_token_averaged_f0 and duration is not None:
            f0 = self._average_by_duration(f0, duration)
        else:
            f0 = np.expand_dims(np.array(f0), 0).T
        return f0


class Energy():
    def __init__(self,
                 n_fft: int=2048,
                 hop_length: int=300,
                 win_length: int=None,
                 window: str="hann",
                 center: bool=True,
                 pad_mode: str="reflect"):

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

    def _stft(self, wav: np.ndarray):
        D = librosa.core.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode)
        return D

    def _calculate_energy(self, input: np.ndarray):
        input = input.astype(np.float32)
        input_stft = self._stft(input)
        input_power = np.abs(input_stft)**2
        energy = np.sqrt(
            np.clip(
                np.sum(input_power, axis=0), a_min=1.0e-10, a_max=float('inf')))
        return energy

    def _average_by_duration(self, input: np.ndarray,
                             d: np.ndarray) -> np.ndarray:
        d_cumsum = np.pad(d.cumsum(0), (1, 0), 'constant')
        arr_list = []
        for start, end in zip(d_cumsum[:-1], d_cumsum[1:]):
            arr = input[start:end]
            avg_arr = np.mean(arr, axis=0) if len(arr) != 0 else np.array(0)
            arr_list.append(avg_arr)
        # shape (T,1)
        arr_list = np.expand_dims(np.array(arr_list), 0).T
        return arr_list

    def get_energy(self,
                   wav: np.ndarray,
                   use_token_averaged_energy: bool=True,
                   duration: np.ndarray=None):
        energy = self._calculate_energy(wav)
        if use_token_averaged_energy and duration is not None:
            energy = self._average_by_duration(energy, duration)
        else:
            energy = np.expand_dims(np.array(energy), 0).T
        return energy


class LinearSpectrogram():
    def __init__(
            self,
            n_fft: int=1024,
            win_length: int=None,
            hop_length: int=256,
            window: str="hann",
            center: bool=True, ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.n_fft = n_fft
        self.pad_mode = "reflect"

    def _stft(self, wav: np.ndarray):
        D = librosa.core.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode)
        return D

    def _spectrogram(self, wav: np.ndarray):
        D = self._stft(wav)
        return np.abs(D)

    def get_linear_spectrogram(self, wav: np.ndarray):
        linear_spectrogram = self._spectrogram(wav)
        linear_spectrogram = np.clip(
            linear_spectrogram, a_min=1e-10, a_max=float("inf"))
        return linear_spectrogram.T
