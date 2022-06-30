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
import os
import warnings
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import resampy
import soundfile as sf
from scipy.io import wavfile

from ..utils import ParameterError
from ..utils import depth_convert

__all__ = [
    'resample',
    'to_mono',
    'normalize',
    'save',
    'load',
]
NORMALMIZE_TYPES = ['linear', 'gaussian']
MERGE_TYPES = ['ch0', 'ch1', 'random', 'average']
RESAMPLE_MODES = ['kaiser_best', 'kaiser_fast']
EPS = 1e-8


def resample(y: np.ndarray,
             src_sr: int,
             target_sr: int,
             mode: str='kaiser_fast') -> np.ndarray:
    """Audio resampling.

    Args:
        y (np.ndarray): Input waveform array in 1D or 2D.
        src_sr (int): Source sample rate.
        target_sr (int): Target sample rate.
        mode (str, optional): The resampling filter to use. Defaults to 'kaiser_fast'.

    Returns:
        np.ndarray: `y` resampled to `target_sr`
    """

    if mode == 'kaiser_best':
        warnings.warn(
            f'Using resampy in kaiser_best to {src_sr}=>{target_sr}. This function is pretty slow, \
        we recommend the mode kaiser_fast in large scale audio trainning')

    if not isinstance(y, np.ndarray):
        raise ParameterError(
            'Only support numpy np.ndarray, but received y in {type(y)}')

    if mode not in RESAMPLE_MODES:
        raise ParameterError(f'resample mode must in {RESAMPLE_MODES}')

    return resampy.resample(y, src_sr, target_sr, filter=mode)


def to_mono(y: np.ndarray, merge_type: str='average') -> np.ndarray:
    """Convert sterior audio to mono.

    Args:
        y (np.ndarray): Input waveform array in 1D or 2D.
        merge_type (str, optional): Merge type to generate mono waveform. Defaults to 'average'.

    Returns:
        np.ndarray: `y` with mono channel.
    """

    if merge_type not in MERGE_TYPES:
        raise ParameterError(
            f'Unsupported merge type {merge_type}, available types are {MERGE_TYPES}'
        )
    if y.ndim > 2:
        raise ParameterError(
            f'Unsupported audio array,  y.ndim > 2, the shape is {y.shape}')
    if y.ndim == 1:  # nothing to merge
        return y

    if merge_type == 'ch0':
        return y[0]
    if merge_type == 'ch1':
        return y[1]
    if merge_type == 'random':
        return y[np.random.randint(0, 2)]

    # need to do averaging according to dtype

    if y.dtype == 'float32':
        y_out = (y[0] + y[1]) * 0.5
    elif y.dtype == 'int16':
        y_out = y.astype('int32')
        y_out = (y_out[0] + y_out[1]) // 2
        y_out = np.clip(y_out, np.iinfo(y.dtype).min,
                        np.iinfo(y.dtype).max).astype(y.dtype)

    elif y.dtype == 'int8':
        y_out = y.astype('int16')
        y_out = (y_out[0] + y_out[1]) // 2
        y_out = np.clip(y_out, np.iinfo(y.dtype).min,
                        np.iinfo(y.dtype).max).astype(y.dtype)
    else:
        raise ParameterError(f'Unsupported dtype: {y.dtype}')
    return y_out


def sound_file_load(file: os.PathLike,
                    offset: Optional[float]=None,
                    dtype: str='int16',
                    duration: Optional[int]=None) -> Tuple[np.ndarray, int]:
    """Load audio using soundfile library. This function load audio file using libsndfile.

    Args:
        file (os.PathLike): File of waveform.
        offset (Optional[float], optional): Offset to the start of waveform. Defaults to None.
        dtype (str, optional): Data type of waveform. Defaults to 'int16'.
        duration (Optional[int], optional): Duration of waveform to read. Defaults to None.

    Returns:
        Tuple[np.ndarray, int]: Waveform in ndarray and its samplerate.
    """
    with sf.SoundFile(file) as sf_desc:
        sr_native = sf_desc.samplerate
        if offset:
            sf_desc.seek(int(offset * sr_native))
        if duration is not None:
            frame_duration = int(duration * sr_native)
        else:
            frame_duration = -1
        y = sf_desc.read(frames=frame_duration, dtype=dtype, always_2d=False).T

    return y, sf_desc.samplerate


def normalize(y: np.ndarray, norm_type: str='linear',
              mul_factor: float=1.0) -> np.ndarray:
    """Normalize an input audio with additional multiplier.

    Args:
        y (np.ndarray): Input waveform array in 1D or 2D.
        norm_type (str, optional): Type of normalization. Defaults to 'linear'.
        mul_factor (float, optional): Scaling factor. Defaults to 1.0.

    Returns:
        np.ndarray: `y` after normalization.
    """

    if norm_type == 'linear':
        amax = np.max(np.abs(y))
        factor = 1.0 / (amax + EPS)
        y = y * factor * mul_factor
    elif norm_type == 'gaussian':
        amean = np.mean(y)
        astd = np.std(y)
        astd = max(astd, EPS)
        y = mul_factor * (y - amean) / astd
    else:
        raise NotImplementedError(f'norm_type should be in {NORMALMIZE_TYPES}')

    return y


def save(y: np.ndarray, sr: int, file: os.PathLike) -> None:
    """Save audio file to disk. This function saves audio to disk using scipy.io.wavfile, with additional step to convert input waveform to int16.

    Args:
        y (np.ndarray): Input waveform array in 1D or 2D.
        sr (int): Sample rate.
        file (os.PathLike): Path of auido file to save.
    """
    if not file.endswith('.wav'):
        raise ParameterError(
            f'only .wav file supported, but dst file name is: {file}')

    if sr <= 0:
        raise ParameterError(
            f'Sample rate should be larger than 0, recieved sr = {sr}')

    if y.dtype not in ['int16', 'int8']:
        warnings.warn(
            f'input data type is {y.dtype}, will convert data to int16 format before saving'
        )
        y_out = depth_convert(y, 'int16')
    else:
        y_out = y

    wavfile.write(file, sr, y_out)


def load(
        file: os.PathLike,
        sr: Optional[int]=None,
        mono: bool=True,
        merge_type: str='average',  # ch0,ch1,random,average
        normal: bool=True,
        norm_type: str='linear',
        norm_mul_factor: float=1.0,
        offset: float=0.0,
        duration: Optional[int]=None,
        dtype: str='float32',
        resample_mode: str='kaiser_fast') -> Tuple[np.ndarray, int]:
    """Load audio file from disk. This function loads audio from disk using using audio beackend.

    Args:
        file (os.PathLike): Path of auido file to load.
        sr (Optional[int], optional): Sample rate of loaded waveform. Defaults to None.
        mono (bool, optional): Return waveform with mono channel. Defaults to True.
        merge_type (str, optional): Merge type of multi-channels waveform. Defaults to 'average'.
        normal (bool, optional): Waveform normalization. Defaults to True.
        norm_type (str, optional): Type of normalization. Defaults to 'linear'.
        norm_mul_factor (float, optional): Scaling factor. Defaults to 1.0.
        offset (float, optional): Offset to the start of waveform. Defaults to 0.0.
        duration (Optional[int], optional): Duration of waveform to read. Defaults to None.
        dtype (str, optional): Data type of waveform. Defaults to 'float32'.
        resample_mode (str, optional): The resampling filter to use. Defaults to 'kaiser_fast'.

    Returns:
        Tuple[np.ndarray, int]: Waveform in ndarray and its samplerate.
    """

    y, r = sound_file_load(file, offset=offset, dtype=dtype, duration=duration)

    if not ((y.ndim == 1 and len(y) > 0) or (y.ndim == 2 and len(y[0]) > 0)):
        raise ParameterError(f'audio file {file} looks empty')

    if mono:
        y = to_mono(y, merge_type)

    if sr is not None and sr != r:
        y = resample(y, r, sr, mode=resample_mode)
        r = sr

    if normal:
        y = normalize(y, norm_type, norm_mul_factor)
    elif dtype in ['int8', 'int16']:
        # still need to do normalization, before depth convertion
        y = normalize(y, 'linear', 1.0)

    y = depth_convert(y, dtype)
    return y, r


def info(filepath: str) -> None:
    raise RuntimeError("No audio I/O backend is available.")