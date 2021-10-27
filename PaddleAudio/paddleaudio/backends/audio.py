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
import warnings
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import resampy
import soundfile as sf
from numpy import ndarray as array
from scipy.io import wavfile

from ..utils import ParameterError

__all__ = [
    'resample',
    'to_mono',
    'depth_convert',
    'normalize',
    'save_wav',
    'load',
]
NORMALMIZE_TYPES = ['linear', 'gaussian']
MERGE_TYPES = ['ch0', 'ch1', 'random', 'average']
RESAMPLE_MODES = ['kaiser_best', 'kaiser_fast']
EPS = 1e-8


def resample(y: array, src_sr: int, target_sr: int,
             mode: str='kaiser_fast') -> array:
    """ Audio resampling

     This function is the same as using resampy.resample().

     Notes:
        The default mode is kaiser_fast.  For better audio quality, use mode = 'kaiser_fast'

     """

    if mode == 'kaiser_best':
        warnings.warn(
            f'Using resampy in kaiser_best to {src_sr}=>{target_sr}. This function is pretty slow, \
        we recommend the mode kaiser_fast in large scale audio trainning')

    if not isinstance(y, np.ndarray):
        raise ParameterError(
            'Only support numpy array, but received y in {type(y)}')

    if mode not in RESAMPLE_MODES:
        raise ParameterError(f'resample mode must in {RESAMPLE_MODES}')

    return resampy.resample(y, src_sr, target_sr, filter=mode)


def to_mono(y: array, merge_type: str='average') -> array:
    """ convert sterior audio to mono
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


def _safe_cast(y: array, dtype: Union[type, str]) -> array:
    """ data type casting in a safe way, i.e., prevent overflow or underflow

    This function is used internally.
    """
    return np.clip(y, np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)


def depth_convert(y: array, dtype: Union[type, str],
                  dithering: bool=True) -> array:
    """Convert audio array to target dtype safely

    This function convert audio waveform to a target dtype, with addition steps of
    preventing overflow/underflow and preserving audio range.

    """

    SUPPORT_DTYPE = ['int16', 'int8', 'float32', 'float64']
    if y.dtype not in SUPPORT_DTYPE:
        raise ParameterError(
            'Unsupported audio dtype, '
            f'y.dtype is {y.dtype}, supported dtypes are {SUPPORT_DTYPE}')

    if dtype not in SUPPORT_DTYPE:
        raise ParameterError(
            'Unsupported audio dtype, '
            f'target dtype  is {dtype}, supported dtypes are {SUPPORT_DTYPE}')

    if dtype == y.dtype:
        return y

    if dtype == 'float64' and y.dtype == 'float32':
        return _safe_cast(y, dtype)
    if dtype == 'float32' and y.dtype == 'float64':
        return _safe_cast(y, dtype)

    if dtype == 'int16' or dtype == 'int8':
        if y.dtype in ['float64', 'float32']:
            factor = np.iinfo(dtype).max
            y = np.clip(y * factor, np.iinfo(dtype).min,
                        np.iinfo(dtype).max).astype(dtype)
            y = y.astype(dtype)
        else:
            if dtype == 'int16' and y.dtype == 'int8':
                factor = np.iinfo('int16').max / np.iinfo('int8').max - EPS
                y = y.astype('float32') * factor
                y = y.astype('int16')

            else:  # dtype == 'int8' and y.dtype=='int16':
                y = y.astype('int32') * np.iinfo('int8').max / \
                    np.iinfo('int16').max
                y = y.astype('int8')

    if dtype in ['float32', 'float64']:
        org_dtype = y.dtype
        y = y.astype(dtype) / np.iinfo(org_dtype).max
    return y


def sound_file_load(file: str,
                    offset: Optional[float]=None,
                    dtype: str='int16',
                    duration: Optional[int]=None) -> Tuple[array, int]:
    """Load audio using soundfile library

    This function load audio file using libsndfile.

    Reference:
        http://www.mega-nerd.com/libsndfile/#Features

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


def audio_file_load():
    """Load audio using audiofile library

    This function load audio file using audiofile.

    Reference:
        https://audiofile.68k.org/

    """
    raise NotImplementedError()


def sox_file_load():
    """Load audio using sox library

    This function load audio file using sox.

    Reference:
        http://sox.sourceforge.net/
    """
    raise NotImplementedError()


def normalize(y: array, norm_type: str='linear',
              mul_factor: float=1.0) -> array:
    """ normalize an input audio with additional multiplier.

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


def save_wav(y: array, sr: int, file: str) -> None:
    """Save audio file to disk.
    This function saves audio to disk using scipy.io.wavfile, with additional step
    to convert input waveform to int16 unless it already is int16

    Notes:
        It only support raw wav format.

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
        file: str,
        sr: Optional[int]=None,
        mono: bool=True,
        merge_type: str='average',  # ch0,ch1,random,average
        normal: bool=True,
        norm_type: str='linear',
        norm_mul_factor: float=1.0,
        offset: float=0.0,
        duration: Optional[int]=None,
        dtype: str='float32',
        resample_mode: str='kaiser_fast') -> Tuple[array, int]:
    """Load audio file from disk.
    This function loads audio from disk using using audio beackend.

    Parameters:

    Notes:

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
