# MIT License, Copyright (c) 2022 OpenAI.
# Copyright (c) 2022 PaddlePaddle Authors and . All Rights Reserved.
# 
# Modified from OpenAI Whisper 2022 (https://github.com/openai/whisper/whisper/audio.py)
import os
from functools import lru_cache
from typing import Union

import numpy as np
import paddle
import paddle.nn.functional as F
import soundfile

from paddlespeech.s2t.models.whisper.utils import exact_div
from paddlespeech.s2t.utils.log import Log
logger = Log(__name__).getlog()
# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
N_FRAMES = exact_div(
    N_SAMPLES, HOP_LENGTH)  # 3000: number of frames in a mel spectrogram input


def pad_or_trim(array, length: int=N_SAMPLES, *, axis: int=-1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if paddle.is_tensor(array):
        if array.shape[axis] > length:
            #array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))
            array = array.index_select(axis=axis, index=paddle.arange(length))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = paddle.transpose(array, (1, 0))
            array = F.pad(
                array, [pad for sizes in pad_widths[::-1] for pad in sizes],
                data_format='NLC')
            array = paddle.transpose(array, (1, 0))
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = paddle.transpose(array, (1, 0))
            array = np.pad(array, pad_widths)
            array = paddle.transpose(array, (1, 0))

    return array


def hann_window(n_fft: int=N_FFT):
    """
    hanning window
    n_fft:  The number of frequency components of the discrete Fourier transform.
    """
    return paddle.to_tensor(
        [0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft) for n in range(n_fft)],
        dtype=paddle.float32)


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int=N_MELS) -> paddle.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(
            os.path.join(
                os.path.dirname(__file__), "assets", "mel_filters.npz")) as f:
        return paddle.to_tensor(f[f"mel_{n_mels}"])


def log_mel_spectrogram(audio: Union[str, np.ndarray, paddle.Tensor],
                        n_mels: int=N_MELS):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not paddle.is_tensor(audio):
        if isinstance(audio, str):
            audio, _ = soundfile.read(audio, dtype="float32", always_2d=True)
            audio = audio[:, 0]
            logger.info(f"audio shape: {audio.shape}")
        audio = paddle.to_tensor(audio)

    window = hann_window(N_FFT)
    stft = paddle.signal.stft(audio, N_FFT, HOP_LENGTH, window=window)

    magnitudes = stft[:, :-1].abs()**2

    filters = mel_filters(audio, n_mels)
    mel_spec = filters @ magnitudes
    mel_spec = paddle.to_tensor(mel_spec.numpy().tolist())

    log_spec = paddle.clip(mel_spec, min=1e-10).log10()
    log_spec = paddle.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
