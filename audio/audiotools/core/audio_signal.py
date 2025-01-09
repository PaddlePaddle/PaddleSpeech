# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# 
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py)
import copy
import functools
import hashlib
import math
import pathlib
import tempfile
import typing
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import paddle
import soundfile

from . import util
from ._julius import resample_frac
from .display import DisplayMixin
from .dsp import DSPMixin
from .effects import EffectMixin
from .effects import ImpulseResponseMixin
from .ffmpeg import FFMPEGMixin
from .loudness import LoudnessMixin

__all__ = ['STFTParams', 'AudioSignal']


def create_dct(n_mfcc: int, n_mels: int, norm: Optional[str]) -> paddle.Tensor:
    r"""Create a DCT transformation matrix with shape (``n_mels``, ``n_mfcc``),
    normalized depending on norm.

    Args:
        n_mfcc (int): Number of mfc coefficients to retain
        n_mels (int): Number of mel filterbanks
        norm (str or None): Norm to use (either "ortho" or None)

    Returns:
        paddle.Tensor: The transformation matrix, to be right-multiplied to
        row-wise data of size (``n_mels``, ``n_mfcc``).
    """

    if norm is not None and norm != "ortho":
        raise ValueError('norm must be either "ortho" or None')

    # http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
    n = paddle.arange(float(n_mels))
    k = paddle.arange(float(n_mfcc)).unsqueeze([1])
    dct = paddle.cos(math.pi / float(n_mels) * (n + 0.5) *
                     k)  # size (n_mfcc, n_mels)

    if norm is None:
        dct *= 2.0
    else:
        dct[0] *= 1.0 / math.sqrt(2.0)
        dct *= math.sqrt(2.0 / float(n_mels))
    return dct.transpose([1, 0])


STFTParams = namedtuple(
    "STFTParams",
    [
        "window_length",
        "hop_length",
        "window_type",
        "match_stride",
        "padding_type",
    ], )
"""
STFTParams object is a container that holds STFT parameters - window_length,
hop_length, and window_type. Not all parameters need to be specified. Ones that
are not specified will be inferred by the AudioSignal parameters.

Parameters
----------
window_length : int, optional
    Window length of STFT, by default ``0.032 * self.sample_rate``.
hop_length : int, optional
    Hop length of STFT, by default ``window_length // 4``.
window_type : str, optional
    Type of window to use, by default ``sqrt\_hann``.
match_stride : bool, optional
    Whether to match the stride of convolutional layers, by default False
padding_type : str, optional
    Type of padding to use, by default 'reflect'
"""
STFTParams.__new__.__defaults__ = (None, None, None, None, None)


class AudioSignal(
        EffectMixin,
        LoudnessMixin,
        ImpulseResponseMixin,
        DSPMixin,
        DisplayMixin,
        FFMPEGMixin, ):
    """This is the core object of this library. Audio is always
    loaded into an AudioSignal, which then enables all the features
    of this library, including audio augmentations, I/O, playback,
    and more.

    The structure of this object is that the base functionality
    is defined in ``core/audio_signal.py``, while extensions to
    that functionality are defined in the other ``core/*.py``
    files. For example, all the display-based functionality
    (e.g. plot spectrograms, waveforms, write to tensorboard)
    are in ``core/display.py``.

    Parameters
    ----------
    audio_path_or_array : typing.Union[paddle.Tensor, str, Path, np.ndarray]
        Object to create AudioSignal from. Can be a tensor, numpy array,
        or a path to a file. The file is always reshaped to
    sample_rate : int, optional
        Sample rate of the audio. If different from underlying file, resampling is
        performed. If passing in an array or tensor, this must be defined,
        by default None
    stft_params : STFTParams, optional
        Parameters of STFT to use. , by default None
    offset : float, optional
        Offset in seconds to read from file, by default 0
    duration : float, optional
        Duration in seconds to read from file, by default None
    device : str, optional
        Device to load audio onto, by default None

    Examples
    --------
    Loading an AudioSignal from an array, at a sample rate of
    44100.

    >>> signal = AudioSignal(paddle.randn([5*44100]), 44100)

    Note, the signal is reshaped to have a batch size, and one
    audio channel:

    >>> print(signal.shape)
    (1, 1, 44100)

    You can treat AudioSignals like tensors, and many of the same
    functions you might use on tensors are defined for AudioSignals
    as well:

    >>> signal.to("cuda")
    >>> signal.cuda()
    >>> signal.clone()
    >>> signal.detach()

    Indexing AudioSignals returns an AudioSignal:

    >>> signal[..., 3*44100:4*44100]

    The above signal is 1 second long, and is also an AudioSignal.
    """

    def __init__(
            self,
            audio_path_or_array: typing.Union[paddle.Tensor, str, Path,
                                              np.ndarray],
            sample_rate: int=None,
            stft_params: STFTParams=None,
            offset: float=0,
            duration: float=None,
            device: str=None, ):
        # 
        audio_path = None
        audio_array = None

        if isinstance(audio_path_or_array, str):
            audio_path = audio_path_or_array
        elif isinstance(audio_path_or_array, pathlib.Path):
            audio_path = audio_path_or_array
        elif isinstance(audio_path_or_array, np.ndarray):
            audio_array = audio_path_or_array
        elif paddle.is_tensor(audio_path_or_array):
            audio_array = audio_path_or_array
        else:
            raise ValueError("audio_path_or_array must be either a Path, "
                             "string, numpy array, or paddle Tensor!")

        self.path_to_file = None

        self.audio_data = None
        self.sources = None  # List of AudioSignal objects.
        self.stft_data = None
        if audio_path is not None:
            self.load_from_file(
                audio_path, offset=offset, duration=duration, device=device)
        elif audio_array is not None:
            assert sample_rate is not None, "Must set sample rate!"
            self.load_from_array(audio_array, sample_rate, device=device)

        self.window = None
        self.stft_params = stft_params

        self.metadata = {
            "offset": offset,
            "duration": duration,
        }

    @property
    def path_to_input_file(
            self, ):
        """
        Path to input file, if it exists.
        Alias to ``path_to_file`` for backwards compatibility
        """
        return self.path_to_file

    @classmethod
    def excerpt(
            cls,
            audio_path: typing.Union[str, Path],
            offset: float=None,
            duration: float=None,
            state: typing.Union[np.random.RandomState, int]=None,
            **kwargs, ):
        """Randomly draw an excerpt of ``duration`` seconds from an
        audio file specified at ``audio_path``, between ``offset`` seconds
        and end of file. ``state`` can be used to seed the random draw.

        Parameters
        ----------
        audio_path : typing.Union[str, Path]
            Path to audio file to grab excerpt from.
        offset : float, optional
            Lower bound for the start time, in seconds drawn from
            the file, by default None.
        duration : float, optional
            Duration of excerpt, in seconds, by default None
        state : typing.Union[np.random.RandomState, int], optional
            RandomState or seed of random state, by default None

        Returns
        -------
        AudioSignal
            AudioSignal containing excerpt.

        Examples
        --------
        >>> signal = AudioSignal.excerpt("path/to/audio", duration=5)
        """
        info = util.info(audio_path)
        total_duration = info.duration

        state = util.random_state(state)
        lower_bound = 0 if offset is None else offset
        upper_bound = max(total_duration - duration, 0)
        offset = state.uniform(lower_bound, upper_bound)

        signal = cls(audio_path, offset=offset, duration=duration, **kwargs)
        signal.metadata["offset"] = offset
        signal.metadata["duration"] = duration

        return signal

    @classmethod
    def salient_excerpt(
            cls,
            audio_path: typing.Union[str, Path],
            loudness_cutoff: float=None,
            num_tries: int=8,
            state: typing.Union[np.random.RandomState, int]=None,
            **kwargs, ):
        """Similar to AudioSignal.excerpt, except it extracts excerpts only
        if they are above a specified loudness threshold, which is computed via
        a fast LUFS routine.

        Parameters
        ----------
        audio_path : typing.Union[str, Path]
            Path to audio file to grab excerpt from.
        loudness_cutoff : float, optional
            Loudness threshold in dB. Typical values are ``-40, -60``,
            etc, by default None
        num_tries : int, optional
            Number of tries to grab an excerpt above the threshold
            before giving up, by default 8.
        state : typing.Union[np.random.RandomState, int], optional
            RandomState or seed of random state, by default None
        kwargs : dict
            Keyword arguments to AudioSignal.excerpt

        Returns
        -------
        AudioSignal
            AudioSignal containing excerpt.


        .. warning::
            if ``num_tries`` is set to None, ``salient_excerpt`` may try forever, which can
            result in an infinite loop if ``audio_path`` does not have
            any loud enough excerpts.

        Examples
        --------
        >>> signal = AudioSignal.salient_excerpt(
                "path/to/audio",
                loudness_cutoff=-40,
                duration=5
            )
        """
        state = util.random_state(state)
        if loudness_cutoff is None:
            excerpt = cls.excerpt(audio_path, state=state, **kwargs)
        else:
            loudness = -np.inf
            num_try = 0
            while loudness <= loudness_cutoff:
                excerpt = cls.excerpt(audio_path, state=state, **kwargs)
                loudness = excerpt.loudness()
                num_try += 1
                if num_tries is not None and num_try >= num_tries:
                    break
        return excerpt

    @classmethod
    def zeros(
            cls,
            duration: float,
            sample_rate: int,
            num_channels: int=1,
            batch_size: int=1,
            **kwargs, ):
        """Helper function create an AudioSignal of all zeros.

        Parameters
        ----------
        duration : float
            Duration of AudioSignal
        sample_rate : int
            Sample rate of AudioSignal
        num_channels : int, optional
            Number of channels, by default 1
        batch_size : int, optional
            Batch size, by default 1

        Returns
        -------
        AudioSignal
            AudioSignal containing all zeros.

        Examples
        --------
        Generate 5 seconds of all zeros at a sample rate of 44100.

        >>> signal = AudioSignal.zeros(5.0, 44100)
        """
        n_samples = int(duration * sample_rate)
        return cls(
            paddle.zeros([batch_size, num_channels, n_samples]),
            sample_rate,
            **kwargs, )

    @classmethod
    def wave(
            cls,
            frequency: float,
            duration: float,
            sample_rate: int,
            num_channels: int=1,
            shape: str="sine",
            **kwargs, ):
        """
        Generate a waveform of a given frequency and shape.

        Parameters
        ----------
        frequency : float
            Frequency of the waveform
        duration : float
            Duration of the waveform
        sample_rate : int
            Sample rate of the waveform
        num_channels : int, optional
            Number of channels, by default 1
        shape : str, optional
            Shape of the waveform, by default "saw"
            One of "sawtooth", "square", "sine", "triangle"
        kwargs : dict
            Keyword arguments to AudioSignal
        """
        n_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, n_samples)
        if shape == "sawtooth":
            from scipy.signal import sawtooth

            wave_data = sawtooth(2 * np.pi * frequency * t, 0.5)
        elif shape == "square":
            from scipy.signal import square

            wave_data = square(2 * np.pi * frequency * t)
        elif shape == "sine":
            wave_data = np.sin(2 * np.pi * frequency * t)
        elif shape == "triangle":
            from scipy.signal import sawtooth

            # frequency is doubled by the abs call, so omit the 2 in 2pi
            wave_data = sawtooth(np.pi * frequency * t, 0.5)
            wave_data = -np.abs(wave_data) * 2 + 1
        else:
            raise ValueError(f"Invalid shape {shape}")

        wave_data = paddle.to_tensor(wave_data, dtype=paddle.float32)
        wave_data = wave_data[None, None].expand([1, num_channels, -1])
        return cls(wave_data, sample_rate, **kwargs)

    @classmethod
    def batch(
            cls,
            audio_signals: list,
            pad_signals: bool=False,
            truncate_signals: bool=False,
            resample: bool=False,
            dim: int=0, ):
        """Creates a batched AudioSignal from a list of AudioSignals.

        Parameters
        ----------
        audio_signals : list[AudioSignal]
            List of AudioSignal objects
        pad_signals : bool, optional
            Whether to pad signals to length of the maximum length
            AudioSignal in the list, by default False
        truncate_signals : bool, optional
            Whether to truncate signals to length of shortest length
            AudioSignal in the list, by default False
        resample : bool, optional
            Whether to resample AudioSignal to the sample rate of
            the first AudioSignal in the list, by default False
        dim : int, optional
            Dimension along which to batch the signals.

        Returns
        -------
        AudioSignal
            Batched AudioSignal.

        Raises
        ------
        RuntimeError
            If not all AudioSignals are the same sample rate, and
            ``resample=False``, an error is raised.
        RuntimeError
            If not all AudioSignals are the same the length, and
            both ``pad_signals=False`` and ``truncate_signals=False``,
            an error is raised.

        Examples
        --------
        Batching a bunch of random signals:

        >>> signal_list = [AudioSignal(paddle.randn([44100]), 44100) for _ in range(10)]
        >>> signal = AudioSignal.batch(signal_list)
        >>> print(signal.shape)
        (10, 1, 44100)

        """
        signal_lengths = [x.signal_length for x in audio_signals]
        sample_rates = [x.sample_rate for x in audio_signals]

        if len(set(sample_rates)) != 1:
            if resample:
                for x in audio_signals:
                    x.resample(sample_rates[0])
            else:
                raise RuntimeError(
                    f"Not all signals had the same sample rate! Got {sample_rates}. "
                    f"All signals must have the same sample rate, or resample must be True. "
                )

        if len(set(signal_lengths)) != 1:
            if pad_signals:
                max_length = max(signal_lengths)
                for x in audio_signals:
                    pad_len = max_length - x.signal_length
                    x.zero_pad(0, pad_len)
            elif truncate_signals:
                min_length = min(signal_lengths)
                for x in audio_signals:
                    x.truncate_samples(min_length)
            else:
                raise RuntimeError(
                    f"Not all signals had the same length! Got {signal_lengths}. "
                    f"All signals must be the same length, or pad_signals/truncate_signals "
                    f"must be True. ")
        # Concatenate along the specified dimension (default 0)
        audio_data = paddle.concat(
            [x.audio_data for x in audio_signals], axis=dim)
        audio_paths = [x.path_to_file for x in audio_signals]

        batched_signal = cls(
            audio_data,
            sample_rate=audio_signals[0].sample_rate, )
        batched_signal.path_to_file = audio_paths
        return batched_signal

    # I/O
    def load_from_file(
            self,
            audio_path: typing.Union[str, Path],
            offset: float,
            duration: float,
            device: str="cpu", ):
        """Loads data from file. Used internally when AudioSignal
        is instantiated with a path to a file.

        Parameters
        ----------
        audio_path : typing.Union[str, Path]
            Path to file
        offset : float
            Offset in seconds
        duration : float
            Duration in seconds
        device : str, optional
            Device to put AudioSignal on, by default "cpu"

        Returns
        -------
        AudioSignal
            AudioSignal loaded from file
        """
        # need `ffmpeg`
        data, sample_rate = librosa.load(
            audio_path,
            offset=offset,
            duration=duration,
            sr=None,
            mono=False, )
        data = util.ensure_tensor(data)
        if data.shape[-1] == 0:
            raise RuntimeError(
                f"Audio file {audio_path} with offset {offset} and duration {duration} is empty!"
            )

        if data.ndim < 2:
            data = data.unsqueeze(0)
        if data.ndim < 3:
            data = data.unsqueeze(0)
        self.audio_data = data

        self.original_signal_length = self.signal_length

        self.sample_rate = sample_rate
        self.path_to_file = audio_path
        return self.to(device)

    def load_from_array(
            self,
            audio_array: typing.Union[paddle.Tensor, np.ndarray],
            sample_rate: int,
            device: str="cpu", ):
        """Loads data from array, reshaping it to be exactly 3
        dimensions. Used internally when AudioSignal is called
        with a tensor or an array.

        Parameters
        ----------
        audio_array : typing.Union[paddle.Tensor, np.ndarray]
            Array/tensor of audio of samples.
        sample_rate : int
            Sample rate of audio
        device : str, optional
            Device to move audio onto, by default "cpu"

        Returns
        -------
        AudioSignal
            AudioSignal loaded from array
        """
        audio_data = util.ensure_tensor(audio_array)

        if str(audio_data.dtype) == paddle.float64:
            audio_data = audio_data.astype("float32")

        if audio_data.ndim < 2:
            audio_data = audio_data.unsqueeze(0)
        if audio_data.ndim < 3:
            audio_data = audio_data.unsqueeze(0)
        self.audio_data = audio_data

        self.original_signal_length = self.signal_length

        self.sample_rate = sample_rate

        return self

    def write(self, audio_path: typing.Union[str, Path]):
        """Writes audio to a file. Only writes the audio
        that is in the very first item of the batch. To write other items
        in the batch, index the signal along the batch dimension
        before writing. After writing, the signal's ``path_to_file``
        attribute is updated to the new path.

        Parameters
        ----------
        audio_path : typing.Union[str, Path]
            Path to write audio to.

        Returns
        -------
        AudioSignal
            Returns original AudioSignal, so you can use this in a fluent
            interface.

        Examples
        --------
        Creating and writing a signal to disk:

        >>> signal = AudioSignal(paddle.randn([10, 1, 44100]), 44100)
        >>> signal.write("/tmp/out.wav")

        Writing a different element of the batch:

        >>> signal[5].write("/tmp/out.wav")

        Using this in a fluent interface:

        >>> signal.write("/tmp/original.wav").low_pass(4000).write("/tmp/lowpass.wav")

        """
        if self.audio_data[0].abs().max() > 1:
            warnings.warn("Audio amplitude > 1 clipped when saving")
        soundfile.write(
            str(audio_path), self.audio_data[0].numpy().T, self.sample_rate)

        self.path_to_file = audio_path
        return self

    def deepcopy(self):
        """Copies the signal and all of its attributes.

        Returns
        -------
        AudioSignal
            Deep copy of the audio signal.
        """
        return copy.deepcopy(self)

    def copy(self):
        """Shallow copy of signal.

        Returns
        -------
        AudioSignal
            Shallow copy of the audio signal.
        """
        return copy.copy(self)

    def clone(self):
        """Clones all tensors contained in the AudioSignal,
        and returns a copy of the signal with everything
        cloned. Useful when using AudioSignal within autograd
        computation graphs.

        Relevant attributes are the stft data, the audio data,
        and the loudness of the file.

        Returns
        -------
        AudioSignal
            Clone of AudioSignal.
        """
        clone = type(self)(
            self.audio_data.clone(),
            self.sample_rate,
            stft_params=self.stft_params, )
        if self.stft_data is not None:
            clone.stft_data = self.stft_data.clone()
        if self._loudness is not None:
            clone._loudness = self._loudness.clone()
        clone.path_to_file = copy.deepcopy(self.path_to_file)
        clone.metadata = copy.deepcopy(self.metadata)
        return clone

    def detach(self):
        """Detaches tensors contained in AudioSignal.

        Relevant attributes are the stft data, the audio data,
        and the loudness of the file.

        Returns
        -------
        AudioSignal
            Same signal, but with all tensors detached.
        """
        if self._loudness is not None:
            self._loudness = self._loudness.detach()
        if self.stft_data is not None:
            self.stft_data = self.stft_data.detach()

        self.audio_data = self.audio_data.detach()
        return self

    def hash(self):
        """Writes the audio data to a temporary file, and then
        hashes it using hashlib. Useful for creating a file
        name based on the audio content.

        Returns
        -------
        str
            Hash of audio data.

        Examples
        --------
        Creating a signal, and writing it to a unique file name:

        >>> signal = AudioSignal(paddle.randn([44100]), 44100)
        >>> hash = signal.hash()
        >>> signal.write(f"{hash}.wav")

        """
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            self.write(f.name)
            h = hashlib.sha256()
            b = bytearray(128 * 1024)
            mv = memoryview(b)
            with open(f.name, "rb", buffering=0) as f:
                for n in iter(lambda: f.readinto(mv), 0):
                    h.update(mv[:n])
            file_hash = h.hexdigest()
        return file_hash

    # Signal operations
    def to_mono(self):
        """Converts audio data to mono audio, by taking the mean
        along the channels dimension.

        Returns
        -------
        AudioSignal
            AudioSignal with mean of channels.
        """
        self.audio_data = self.audio_data.mean(1, keepdim=True)
        return self

    def resample(self, sample_rate: int):
        """Resamples the audio, using sinc interpolation. This works on both
        cpu and gpu, and is much faster on gpu.

        Parameters
        ----------
        sample_rate : int
            Sample rate to resample to.

        Returns
        -------
        AudioSignal
            Resampled AudioSignal
        """
        if sample_rate == self.sample_rate:
            return self
        self.audio_data = resample_frac(self.audio_data, self.sample_rate,
                                        sample_rate)
        self.sample_rate = sample_rate
        return self

    # Tensor operations
    def to(self, device: str):
        """Moves all tensors contained in signal to the specified device.

        Parameters
        ----------
        device : str
            Device to move AudioSignal onto. Typical values are
            "gpu", "cpu", or "gpu:x" to specify the nth gpu.

        Returns
        -------
        AudioSignal
            AudioSignal with all tensors moved to specified device.
        """
        if self._loudness is not None:
            self._loudness = util.move_to_device(self._loudness, device)
        if self.stft_data is not None:
            self.stft_data = util.move_to_device(self.stft_data, device)
        if self.audio_data is not None:
            self.audio_data = util.move_to_device(self.audio_data, device)
        return self

    def float(self):
        """Calls ``.float()`` on ``self.audio_data``.

        Returns
        -------
        AudioSignal
        """
        self.audio_data = self.audio_data.astype("float32")
        return self

    def cpu(self):
        """Moves AudioSignal to cpu.

        Returns
        -------
        AudioSignal
        """
        return self.to("cpu")

    def cuda(self):
        """Moves AudioSignal to cuda.

        Returns
        -------
        AudioSignal
        """
        return self.to("gpu")

    def numpy(self):
        """Detaches ``self.audio_data``, moves to cpu, and converts to numpy.

        Returns
        -------
        np.ndarray
            Audio data as a numpy array.
        """
        return self.audio_data.detach().cpu().numpy()

    def zero_pad(self, before: int, after: int):
        """Zero pads the audio_data tensor before and after.

        Parameters
        ----------
        before : int
            How many zeros to prepend to audio.
        after : int
            How many zeros to append to audio.

        Returns
        -------
        AudioSignal
            AudioSignal with padding applied.
        """
        self.audio_data = paddle.nn.functional.pad(
            self.audio_data, (before, after), data_format="NCL")
        return self

    def zero_pad_to(self, length: int, mode: str="after"):
        """Pad with zeros to a specified length, either before or after
        the audio data.

        Parameters
        ----------
        length : int
            Length to pad to
        mode : str, optional
            Whether to prepend or append zeros to signal, by default "after"

        Returns
        -------
        AudioSignal
            AudioSignal with padding applied.
        """
        if mode == "before":
            self.zero_pad(max(length - self.signal_length, 0), 0)
        elif mode == "after":
            self.zero_pad(0, max(length - self.signal_length, 0))
        return self

    def trim(self, before: int, after: int):
        """Trims the audio_data tensor before and after.

        Parameters
        ----------
        before : int
            How many samples to trim from beginning.
        after : int
            How many samples to trim from end.

        Returns
        -------
        AudioSignal
            AudioSignal with trimming applied.
        """
        if after == 0:
            self.audio_data = self.audio_data[..., before:]
        else:
            self.audio_data = self.audio_data[..., before:-after]
        return self

    def truncate_samples(self, length_in_samples: int):
        """Truncate signal to specified length.

        Parameters
        ----------
        length_in_samples : int
            Truncate to this many samples.

        Returns
        -------
        AudioSignal
            AudioSignal with truncation applied.
        """
        self.audio_data = self.audio_data[..., :length_in_samples]
        return self

    @property
    def device(self):
        """Get device that AudioSignal is on.

        Returns
        -------
        paddle.device
            Device that AudioSignal is on.
        """
        if self.audio_data is not None:
            device = self.audio_data.place
        elif self.stft_data is not None:
            device = self.stft_data.place
        return device

    # Properties
    @property
    def audio_data(self):
        """Returns the audio data tensor in the object.

        Audio data is always of the shape
        (batch_size, num_channels, num_samples). If value has less
        than 3 dims (e.g. is (num_channels, num_samples)), then it will
        be reshaped to (1, num_channels, num_samples) - a batch size of 1.

        Parameters
        ----------
        data : typing.Union[paddle.Tensor, np.ndarray]
            Audio data to set.

        Returns
        -------
        paddle.Tensor
            Audio samples.
        """
        return self._audio_data

    @audio_data.setter
    def audio_data(self, data: typing.Union[paddle.Tensor, np.ndarray]):
        if data is not None:
            assert paddle.is_tensor(data), "audio_data should be paddle.Tensor"
            assert data.ndim == 3, "audio_data should be 3-dim (B, C, T)"
        self._audio_data = data
        # Old loudness value not guaranteed to be right, reset it.
        self._loudness = None
        return

    # alias for audio_data
    samples = audio_data

    @property
    def stft_data(self):
        """Returns the STFT data inside the signal. Shape is
        (batch, channels, frequencies, time).

        Returns
        -------
        paddle.Tensor
            Complex spectrogram data.
        """
        return self._stft_data

    @stft_data.setter
    def stft_data(self, data: typing.Union[paddle.Tensor, np.ndarray]):
        if data is not None:
            assert paddle.is_tensor(data) and paddle.is_complex(data)
            if self.stft_data is not None and self.stft_data.shape != data.shape:
                warnings.warn("stft_data changed shape")
        self._stft_data = data
        return

    @property
    def batch_size(self):
        """Batch size of audio signal.

        Returns
        -------
        int
            Batch size of signal.
        """
        return self.audio_data.shape[0]

    @property
    def signal_length(self):
        """Length of audio signal.

        Returns
        -------
        int
            Length of signal in samples.
        """
        return self.audio_data.shape[-1]

    # alias for signal_length
    length = signal_length

    @property
    def shape(self):
        """Shape of audio data.

        Returns
        -------
        tuple
            Shape of audio data.
        """
        return self.audio_data.shape

    @property
    def signal_duration(self):
        """Length of audio signal in seconds.

        Returns
        -------
        float
            Length of signal in seconds.
        """
        return self.signal_length / self.sample_rate

    # alias for signal_duration
    duration = signal_duration

    @property
    def num_channels(self):
        """Number of audio channels.

        Returns
        -------
        int
            Number of audio channels.
        """
        return self.audio_data.shape[1]

    # STFT
    @staticmethod
    @functools.lru_cache(None)
    def get_window(window_type: str, window_length: int, device: str=None):
        """Wrapper around scipy.signal.get_window so one can also get the
        popular sqrt-hann window. This function caches for efficiency
        using functools.lru\_cache.

        Parameters
        ----------
        window_type : str
            Type of window to get
        window_length : int
            Length of the window
        device : str
            Device to put window onto.

        Returns
        -------
        paddle.Tensor
            Window returned by scipy.signal.get_window, as a tensor.
        """
        from scipy import signal

        if window_type == "average":
            window = np.ones(window_length) / window_length
        elif window_type == "sqrt_hann":
            window = np.sqrt(signal.get_window("hann", window_length))
        else:
            window = signal.get_window(window_type, window_length)
        window = paddle.to_tensor(window).astype("float32")
        return window

    @property
    def stft_params(self):
        """Returns STFTParams object, which can be re-used to other
        AudioSignals.

        This property can be set as well. If values are not defined in STFTParams,
        they are inferred automatically from the signal properties. The default is to use
        32ms windows, with 8ms hop length, and the square root of the hann window.

        Returns
        -------
        STFTParams
            STFT parameters for the AudioSignal.

        Examples
        --------
        >>> stft_params = STFTParams(128, 32)
        >>> signal1 = AudioSignal(paddle.randn([44100]), 44100, stft_params=stft_params)
        >>> signal2 = AudioSignal(paddle.randn([44100]), 44100, stft_params=signal1.stft_params)
        >>> signal1.stft_params = STFTParams() # Defaults
        """
        return self._stft_params

    @stft_params.setter
    def stft_params(self, value: STFTParams):
        # 
        default_win_len = int(2**(np.ceil(np.log2(0.032 * self.sample_rate))))
        default_hop_len = default_win_len // 4
        default_win_type = "hann"
        default_match_stride = False
        default_padding_type = "reflect"

        default_stft_params = STFTParams(
            window_length=default_win_len,
            hop_length=default_hop_len,
            window_type=default_win_type,
            match_stride=default_match_stride,
            padding_type=default_padding_type, )._asdict()

        value = value._asdict() if value else default_stft_params

        for key in default_stft_params:
            if value[key] is None:
                value[key] = default_stft_params[key]

        self._stft_params = STFTParams(**value)
        self.stft_data = None

    def compute_stft_padding(self,
                             window_length: int,
                             hop_length: int,
                             match_stride: bool):
        """Compute how the STFT should be padded, based on match\_stride.

        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_length : int
            Hop length of STFT.
        match_stride : bool
            Whether or not to match stride, making the STFT have the same alignment as
            convolutional layers.

        Returns
        -------
        tuple
            Amount to pad on either side of audio.
        """
        length = self.signal_length

        if match_stride:
            assert hop_length == window_length // 4, "For match_stride, hop must equal n_fft // 4"
            right_pad = math.ceil(length / hop_length) * hop_length - length
            pad = (window_length - hop_length) // 2
        else:
            right_pad = 0
            pad = 0

        return right_pad, pad

    def stft(
            self,
            window_length: int=None,
            hop_length: int=None,
            window_type: str=None,
            match_stride: bool=None,
            padding_type: str=None, ):
        """Computes the short-time Fourier transform of the audio data,
        with specified STFT parameters.

        Parameters
        ----------
        window_length : int, optional
            Window length of STFT, by default ``0.032 * self.sample_rate``.
        hop_length : int, optional
            Hop length of STFT, by default ``window_length // 4``.
        window_type : str, optional
            Type of window to use, by default ``sqrt\_hann``.
        match_stride : bool, optional
            Whether to match the stride of convolutional layers, by default False
        padding_type : str, optional
            Type of padding to use, by default 'reflect'

        Returns
        -------
        paddle.Tensor
            STFT of audio data.

        Examples
        --------
        Compute the STFT of an AudioSignal:

        >>> signal = AudioSignal(paddle.randn([44100]), 44100)
        >>> signal.stft()

        Vary the window and hop length:

        >>> stft_params_list = [STFTParams(128, 32), STFTParams(512, 128)]
        >>> for stft_params in stft_params_list:
        >>>     signal.stft_params = stft_params
        >>>     signal.stft()

        """
        window_length = self.stft_params.window_length if window_length is None else int(
            window_length)
        hop_length = self.stft_params.hop_length if hop_length is None else int(
            hop_length)
        window_type = self.stft_params.window_type if window_type is None else window_type
        match_stride = self.stft_params.match_stride if match_stride is None else match_stride
        padding_type = self.stft_params.padding_type if padding_type is None else padding_type

        window = self.get_window(window_type, window_length)

        audio_data = self.audio_data
        right_pad, pad = self.compute_stft_padding(window_length, hop_length,
                                                   match_stride)
        audio_data = paddle.nn.functional.pad(
            x=audio_data,
            pad=[pad, pad + right_pad],
            mode="reflect",
            data_format="NCL", )
        stft_data = paddle.signal.stft(
            audio_data.reshape([-1, audio_data.shape[-1]]).astype("float32"),
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            # return_complex=True,
            center=True, )
        _, nf, nt = stft_data.shape
        stft_data = stft_data.reshape(
            [self.batch_size, self.num_channels, nf, nt])

        if match_stride:
            # Drop first two and last two frames, which are added
            # because of padding. Now num_frames * hop_length = num_samples.
            stft_data = stft_data[..., 2:-2]
        self.stft_data = stft_data

        return stft_data

    def istft(
            self,
            window_length: int=None,
            hop_length: int=None,
            window_type: str=None,
            match_stride: bool=None,
            length: int=None, ):
        """Computes inverse STFT and sets it to audio\_data.

        Parameters
        ----------
        window_length : int, optional
            Window length of STFT, by default ``0.032 * self.sample_rate``.
        hop_length : int, optional
            Hop length of STFT, by default ``window_length // 4``.
        window_type : str, optional
            Type of window to use, by default ``sqrt\_hann``.
        match_stride : bool, optional
            Whether to match the stride of convolutional layers, by default False
        length : int, optional
            Original length of signal, by default None

        Returns
        -------
        AudioSignal
            AudioSignal with istft applied.

        Raises
        ------
        RuntimeError
            Raises an error if stft was not called prior to istft on the signal,
            or if stft_data is not set.
        """
        if self.stft_data is None:
            raise RuntimeError("Cannot do inverse STFT without self.stft_data!")

        window_length = self.stft_params.window_length if window_length is None else int(
            window_length)
        hop_length = self.stft_params.hop_length if hop_length is None else int(
            hop_length)
        window_type = self.stft_params.window_type if window_type is None else window_type
        match_stride = self.stft_params.match_stride if match_stride is None else match_stride

        window = self.get_window(window_type, window_length,
                                 self.stft_data.place)

        nb, nch, nf, nt = self.stft_data.shape
        stft_data = self.stft_data.reshape([nb * nch, nf, nt])
        right_pad, pad = self.compute_stft_padding(window_length, hop_length,
                                                   match_stride)

        if length is None:
            length = self.original_signal_length
            length = length + 2 * pad + right_pad

        if match_stride:
            # Zero-pad the STFT on either side, putting back the frames that were
            # dropped in stft().
            stft_data = paddle.nn.functional.pad(
                stft_data, pad=(2, 2), data_format="NCL")

        audio_data = paddle.signal.istft(
            stft_data,
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            length=length,
            center=True, )
        audio_data = audio_data.reshape([nb, nch, -1])
        if match_stride:
            audio_data = audio_data[..., pad:-(pad + right_pad)]
        self.audio_data = audio_data

        return self

    @staticmethod
    @functools.lru_cache(None)
    def get_mel_filters(sr: int,
                        n_fft: int,
                        n_mels: int,
                        fmin: float=0.0,
                        fmax: float=None):
        """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins.

        Parameters
        ----------
        sr : int
            Sample rate of audio
        n_fft : int
            Number of FFT bins
        n_mels : int
            Number of mels
        fmin : float, optional
            Lowest frequency, in Hz, by default 0.0
        fmax : float, optional
            Highest frequency, by default None

        Returns
        -------
        np.ndarray [shape=(n_mels, 1 + n_fft/2)]
            Mel transform matrix
        """
        from librosa.filters import mel as librosa_mel_fn

        return librosa_mel_fn(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax, )

    def mel_spectrogram(
            self,
            n_mels: int=80,
            mel_fmin: float=0.0,
            mel_fmax: float=None,
            **kwargs, ):
        """Computes a Mel spectrogram.

        Parameters
        ----------
        n_mels : int, optional
            Number of mels, by default 80
        mel_fmin : float, optional
            Lowest frequency, in Hz, by default 0.0
        mel_fmax : float, optional
            Highest frequency, by default None
        kwargs : dict, optional
            Keyword arguments to self.stft().

        Returns
        -------
        paddle.Tensor [shape=(batch, channels, mels, time)]
            Mel spectrogram.
        """
        # from paddle.audio.compliance.librosa import melspectrogram
        # # from ..compliance.librosa import melspectrogram
        # return melspectrogram(
        #     x=self.audio_data,
        #     sr=self.sample_rate,
        #     window_size: int=512,
        #     hop_length: int=320,
        #     n_mels: int=64,
        #     fmin: float=50.0,
        #     fmax: Optional[float]=None,
        #     window: str='hann',
        #     center: bool=True,
        #     pad_mode: str='reflect',
        #     power: float=2.0,
        #     to_db: bool=True,
        #     ref: float=1.0,
        #     amin: float=1e-10,
        #     top_db: Optional[float]=None
        # )

        stft = self.stft(**kwargs)
        magnitude = paddle.abs(stft)

        nf = magnitude.shape[2]
        mel_basis = self.get_mel_filters(
            sr=self.sample_rate,
            n_fft=2 * (nf - 1),
            n_mels=n_mels,
            fmin=mel_fmin,
            fmax=mel_fmax, )
        mel_basis = paddle.to_tensor(mel_basis)

        mel_spectrogram = magnitude.transpose([0, 1, 3, 2]) @ mel_basis.T
        mel_spectrogram = mel_spectrogram.transpose([0, 1, 3, 2])
        return mel_spectrogram

    @staticmethod
    @functools.lru_cache(None)
    def get_dct(n_mfcc: int, n_mels: int, norm: str="ortho", device: str=None):
        """Create a discrete cosine transform (DCT) transformation matrix with shape (``n_mels``, ``n_mfcc``),
        it can be normalized depending on norm. For more information about dct:
        http://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II

        Parameters
        ----------
        n_mfcc : int
            Number of mfccs
        n_mels : int
            Number of mels
        norm   : str
            Use "ortho" to get a orthogonal matrix or None, by default "ortho"
        device : str, optional
            Device to load the transformation matrix on, by default None

        Returns
        -------
        paddle.Tensor [shape=(n_mels, n_mfcc)] T
            The dct transformation matrix.
        """

        return create_dct(n_mfcc, n_mels, norm)

    def mfcc(
            self,
            n_mfcc: int=40,
            n_mels: int=80,
            log_offset: float=1e-6,
            **kwargs, ):
        """Computes mel-frequency cepstral coefficients (MFCCs).

        Parameters
        ----------
        n_mfcc : int, optional
            Number of mels, by default 40
        n_mels : int, optional
            Number of mels, by default 80
        log_offset: float, optional
            Small value to prevent numerical issues when trying to compute log(0), by default 1e-6
        kwargs : dict, optional
            Keyword arguments to self.mel_spectrogram(), note that some of them will be used for self.stft()

        Returns
        -------
        paddle.Tensor [shape=(batch, channels, mfccs, time)]
            MFCCs.
        """

        # from paddle.audio.compliance.librosa import mfcc
        # return mfcc(self.audio_data, self.sample_rate, n_mfcc=n_mfcc, n_mels=n_mels)

        mel_spectrogram = self.mel_spectrogram(n_mels, **kwargs)
        mel_spectrogram = paddle.log(mel_spectrogram + log_offset)
        dct_mat = self.get_dct(n_mfcc, n_mels, "ortho", self.device)

        mfcc = mel_spectrogram.transpose([0, 1, 3, 2]) @ dct_mat
        mfcc = mfcc.transpose([0, 1, 3, 2])
        return mfcc

    @property
    def magnitude(self):
        """Computes and returns the absolute value of the STFT, which
        is the magnitude. This value can also be set to some tensor.
        When set, ``self.stft_data`` is manipulated so that its magnitude
        matches what this is set to, and modulated by the phase.

        Returns
        -------
        paddle.Tensor
            Magnitude of STFT.

        Examples
        --------
        >>> signal = AudioSignal(paddle.randn([44100]), 44100)
        >>> magnitude = signal.magnitude # Computes stft if not computed
        >>> magnitude[magnitude < magnitude.mean()] = 0
        >>> signal.magnitude = magnitude
        >>> signal.istft()
        """
        if self.stft_data is None:
            self.stft()
        return paddle.abs(self.stft_data)

    @magnitude.setter
    def magnitude(self, value):
        self.stft_data = value * util.exp_compat(1j * self.phase)
        return

    def log_magnitude(self,
                      ref_value: float=1.0,
                      amin: float=1e-5,
                      top_db: float=80.0):
        """Computes the log-magnitude of the spectrogram.

        Parameters
        ----------
        ref_value : float, optional
            The magnitude is scaled relative to ``ref``: ``20 * log10(S / ref)``.
            Zeros in the output correspond to positions where ``S == ref``,
            by default 1.0
        amin : float, optional
            Minimum threshold for ``S`` and ``ref``, by default 1e-5
        top_db : float, optional
            Threshold the output at ``top_db`` below the peak:
            ``max(10 * log10(S/ref)) - top_db``, by default -80.0

        Returns
        -------
        paddle.Tensor
            Log-magnitude spectrogram
        """
        magnitude = self.magnitude

        amin = amin**2
        log_spec = 10.0 * paddle.log10(magnitude.pow(2).clip(min=amin))
        if paddle.is_tensor(ref_value):
            ref_value = ref_value.item()
        log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

        if top_db is not None:
            log_spec = paddle.maximum(log_spec, log_spec.max() - top_db)
        return log_spec

    @property
    def phase(self):
        """Computes and returns the phase of the STFT.
        This value can also be set to some tensor.
        When set, ``self.stft_data`` is manipulated so that its phase
        matches what this is set to, we original magnitudeith th.

        Returns
        -------
        paddle.Tensor
            Phase of STFT.

        Examples
        --------
        >>> signal = AudioSignal(paddle.randn([44100]), 44100)
        >>> phase = signal.phase # Computes stft if not computed
        >>> phase[phase < phase.mean()] = 0
        >>> signal.phase = phase
        >>> signal.istft()
        """
        if self.stft_data is None:
            self.stft()
        return paddle.angle(self.stft_data)

    @phase.setter
    def phase(self, value):
        # 
        self.stft_data = self.magnitude * util.exp_compat(1j * value)
        return

    # Operator overloading
    def __add__(self, other):
        new_signal = self.clone()
        new_signal.audio_data += util._get_value(other)
        return new_signal

    def __iadd__(self, other):
        self.audio_data += util._get_value(other)
        return self

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        new_signal = self.clone()
        new_signal.audio_data -= util._get_value(other)
        return new_signal

    def __isub__(self, other):
        self.audio_data -= util._get_value(other)
        return self

    def __mul__(self, other):
        new_signal = self.clone()
        new_signal.audio_data *= util._get_value(other)
        return new_signal

    def __imul__(self, other):
        self.audio_data *= util._get_value(other)
        return self

    def __rmul__(self, other):
        return self * other

    # Representation
    def _info(self):
        # 
        dur = f"{self.signal_duration:0.3f}" if self.signal_duration else "[unknown]"
        info = {
            "duration":
            f"{dur} seconds",
            "batch_size":
            self.batch_size,
            "path":
            self.path_to_file if self.path_to_file else "path unknown",
            "sample_rate":
            self.sample_rate,
            "num_channels": (self.num_channels
                             if self.num_channels else "[unknown]"),
            "audio_data.shape":
            self.audio_data.shape,
            "stft_params":
            self.stft_params,
            "device":
            self.device,
        }

        return info

    def markdown(self):
        """Produces a markdown representation of AudioSignal, in a markdown table.

        Returns
        -------
        str
            Markdown representation of AudioSignal.

        Examples
        --------
        >>> signal = AudioSignal(paddle.randn([44100]), 44100)
        >>> print(signal.markdown())
        | Key | Value
        |---|---
        | duration | 1.000 seconds |
        | batch_size | 1 |
        | path | path unknown |
        | sample_rate | 44100 |
        | num_channels | 1 |
        | audio_data.shape | paddle.Size([1, 1, 44100]) |
        | stft_params | STFTParams(window_length=2048, hop_length=512, window_type='sqrt_hann', match_stride=False) |
        | device | cpu |
        """
        info = self._info()

        FORMAT = "| Key | Value \n" "|---|--- \n"
        for k, v in info.items():
            row = f"| {k} | {v} |\n"
            FORMAT += row
        return FORMAT

    def __str__(self):
        info = self._info()

        desc = ""
        for k, v in info.items():
            desc += f"{k}: {v}\n"
        return desc

    def __rich__(self):
        from rich.table import Table

        info = self._info()

        table = Table(title=f"{self.__class__.__name__}")
        table.add_column("Key", style="green")
        table.add_column("Value", style="cyan")

        for k, v in info.items():
            table.add_row(k, str(v))
        return table

    # Comparison
    def __eq__(self, other):
        for k, v in list(self.__dict__.items()):
            if paddle.is_tensor(v):

                if paddle.is_complex(v):
                    if not np.allclose(
                            v.cpu().numpy(),
                            other.__dict__[k].cpu().numpy(),
                            atol=1e-6):
                        max_error = (v - other.__dict__[k]).abs().max()
                        print(f"Max abs error for {k}: {max_error}")
                        return False
                else:
                    if not paddle.allclose(v, other.__dict__[k], atol=1e-6):
                        max_error = (v - other.__dict__[k]).abs().max()
                        print(f"Max abs error for {k}: {max_error}")
                        return False
        return True

    # Indexing
    def __getitem__(self, key):
        if paddle.is_tensor(key) and key.ndim == 0 and key.item() is True:
            assert self.batch_size == 1
            audio_data = self.audio_data
            _loudness = self._loudness
            stft_data = self.stft_data

        elif isinstance(key, (bool, int, list, slice, tuple)) or (
                paddle.is_tensor(key) and key.ndim <= 1):
            # Indexing only on the batch dimension.
            # Then let's copy over relevant stuff.
            # Future work: make this work for time-indexing
            # as well, using the hop length.
            audio_data = self.audio_data[key]
            _loudness = self._loudness[
                key] if self._loudness is not None else None
            # stft_data = self.stft_data[
            #     key] if self.stft_data is not None else None
            stft_data = util.bool_index_compat(
                self.stft_data, key) if self.stft_data is not None else None

        sources = None

        copy = type(self)(
            audio_data, self.sample_rate, stft_params=self.stft_params)
        copy._loudness = _loudness
        copy._stft_data = stft_data
        copy.sources = sources

        return copy

    def __setitem__(self, key, value):
        if not isinstance(value, type(self)):
            self.audio_data[key] = value
            return

        if paddle.is_tensor(key) and key.ndim == 0 and key.item() is True:
            assert self.batch_size == 1
            self.audio_data = value.audio_data
            self._loudness = value._loudness
            self.stft_data = value.stft_data
            return

        elif isinstance(key, (bool, int, list, slice, tuple)) or (
                paddle.is_tensor(key) and key.ndim <= 1):
            if self.audio_data is not None and value.audio_data is not None:
                self.audio_data[key] = value.audio_data
            if self._loudness is not None and value._loudness is not None:
                if paddle.is_tensor(key) and key.dtype == paddle.bool:
                    # FOR Paddle BOOL Index
                    _key_no_bool = paddle.nonzero(key).flatten()
                    self._loudness[_key_no_bool] = value._loudness
                else:
                    self._loudness[key] = value._loudness
            if self.stft_data is not None and value.stft_data is not None:
                # self.stft_data[key] = value.stft_data
                self.stft_data = util.bool_setitem_compat(self.stft_data, key,
                                                          value.stft_data)
            return

    def __ne__(self, other):
        return not self == other
