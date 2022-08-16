import os
from typing import List, Optional, Tuple
import paddle
import numpy

from paddlespeech.audio._internal import module_utils as _mod_utils
from paddlespeech.audio.utils.sox_utils import list_effects
from paddlespeech.audio import _paddleaudio as paddleaudio

#code is from: https://github.com/pytorch/audio/blob/main/torchaudio/sox_effects/sox_effects.py

@_mod_utils.requires_sox()
def init_sox_effects():
    """Initialize resources required to use sox effects.

    Note:
        You do not need to call this function manually. It is called automatically.

    Once initialized, you do not need to call this function again across the multiple uses of
    sox effects though it is safe to do so as long as :func:`shutdown_sox_effects` is not called yet.
    Once :func:`shutdown_sox_effects` is called, you can no longer use SoX effects and initializing
    again will result in error.
    """
    paddleaudio.sox_effects_initialize_sox_effects()


@_mod_utils.requires_sox()
def shutdown_sox_effects():
    """Clean up resources required to use sox effects.

    Note:
        You do not need to call this function manually. It is called automatically.

    It is safe to call this function multiple times.
    Once :py:func:`shutdown_sox_effects` is called, you can no longer use SoX effects and
    initializing again will result in error.
    """
    paddleaudio.sox_effects_shutdown_sox_effects()


@_mod_utils.requires_sox()
def effect_names() -> List[str]:
    """Gets list of valid sox effect names

    Returns:
        List[str]: list of available effect names.

    Example
        >>> paddleaudio.sox_effects.effect_names()
        ['allpass', 'band', 'bandpass', ... ]
    """
    return list(list_effects().keys())


@_mod_utils.requires_sox()
def apply_effects_tensor(
    tensor: paddle.Tensor,
    sample_rate: int,
    effects: List[List[str]],
    channels_first: bool = True,
) -> Tuple[paddle.Tensor, int]:
    """Apply sox effects to given Tensor

    .. devices:: CPU

    Note:
        This function only works on CPU Tensors.
        This function works in the way very similar to ``sox`` command, however there are slight
        differences. For example, ``sox`` command adds certain effects automatically (such as
        ``rate`` effect after ``speed`` and ``pitch`` and other effects), but this function does
        only applies the given effects. (Therefore, to actually apply ``speed`` effect, you also
        need to give ``rate`` effect with desired sampling rate.).

    Args:
        tensor (paddle.Tensor): Input 2D CPU Tensor.
        sample_rate (int): Sample rate
        effects (List[List[str]]): List of effects.
        channels_first (bool, optional): Indicates if the input Tensor's dimension is
            `[channels, time]` or `[time, channels]`

    Returns:
        (Tensor, int): Resulting Tensor and sample rate.
        The resulting Tensor has the same ``dtype`` as the input Tensor, and
        the same channels order. The shape of the Tensor can be different based on the
        effects applied. Sample rate can also be different based on the effects applied.

    Example - Basic usage
        >>>
        >>> # Defines the effects to apply
        >>> effects = [
        ...     ['gain', '-n'],  # normalises to 0dB
        ...     ['pitch', '5'],  # 5 cent pitch shift
        ...     ['rate', '8000'],  # resample to 8000 Hz
        ... ]
        >>>
        >>> # Generate pseudo wave:
        >>> # normalized, channels first, 2ch, sampling rate 16000, 1 second
        >>> sample_rate = 16000
        >>> waveform = 2 * paddle.rand([2, sample_rate * 1]) - 1
        >>> waveform.shape
        paddle.Size([2, 16000])
        >>> waveform
        tensor([[ 0.3138,  0.7620, -0.9019,  ..., -0.7495, -0.4935,  0.5442],
                [-0.0832,  0.0061,  0.8233,  ..., -0.5176, -0.9140, -0.2434]])
        >>>
        >>> # Apply effects
        >>> waveform, sample_rate = apply_effects_tensor(
        ...     wave_form, sample_rate, effects, channels_first=True)
        >>>
        >>> # Check the result
        >>> # The new waveform is sampling rate 8000, 1 second.
        >>> # normalization and channel order are preserved
        >>> waveform.shape
        paddle.Size([2, 8000])
        >>> waveform
        tensor([[ 0.5054, -0.5518, -0.4800,  ..., -0.0076,  0.0096, -0.0110],
                [ 0.1331,  0.0436, -0.3783,  ..., -0.0035,  0.0012,  0.0008]])
        >>> sample_rate
        8000

    """
    tensor_np = tensor.numpy()
    ret = paddleaudio.sox_effects_apply_effects_tensor(tensor_np, sample_rate, effects, channels_first)
    if ret is not None:
       return (paddle.to_tensor(ret[0]), ret[1])
    raise RuntimeError("Failed to apply sox effect")


@_mod_utils.requires_sox()
def apply_effects_file(
    path: str,
    effects: List[List[str]],
    normalize: bool = True,
    channels_first: bool = True,
    format: Optional[str] = None,
) -> Tuple[paddle.Tensor, int]:
    """Apply sox effects to the audio file and load the resulting data as Tensor

    Note:
        This function works in the way very similar to ``sox`` command, however there are slight
        differences. For example, ``sox`` commnad adds certain effects automatically (such as
        ``rate`` effect after ``speed``, ``pitch`` etc), but this function only applies the given
        effects. Therefore, to actually apply ``speed`` effect, you also need to give ``rate``
        effect with desired sampling rate, because internally, ``speed`` effects only alter sampling
        rate and leave samples untouched.

    Args:
        path (path-like object or file-like object):
        effects (List[List[str]]): List of effects.
        normalize (bool, optional):
            When ``True``, this function always return ``float32``, and sample values are
            normalized to ``[-1.0, 1.0]``.
            If input file is integer WAV, giving ``False`` will change the resulting Tensor type to
            integer type. This argument has no effect for formats other
            than integer WAV type.
        channels_first (bool, optional): When True, the returned Tensor has dimension `[channel, time]`.
            Otherwise, the returned Tensor's dimension is `[time, channel]`.
        format (str or None, optional):
            Override the format detection with the given format.
            Providing the argument might help when libsox can not infer the format
            from header or extension,

    Returns:
        (Tensor, int): Resulting Tensor and sample rate.
        If ``normalize=True``, the resulting Tensor is always ``float32`` type.
        If ``normalize=False`` and the input audio file is of integer WAV file, then the
        resulting Tensor has corresponding integer type. (Note 24 bit integer type is not supported)
        If ``channels_first=True``, the resulting Tensor has dimension `[channel, time]`,
        otherwise `[time, channel]`.

    Example - Basic usage
        >>>
        >>> # Defines the effects to apply
        >>> effects = [
        ...     ['gain', '-n'],  # normalises to 0dB
        ...     ['pitch', '5'],  # 5 cent pitch shift
        ...     ['rate', '8000'],  # resample to 8000 Hz
        ... ]
        >>>
        >>> # Apply effects and load data with channels_first=True
        >>> waveform, sample_rate = apply_effects_file("data.wav", effects, channels_first=True)
        >>>
        >>> # Check the result
        >>> waveform.shape
        paddle.Size([2, 8000])
        >>> waveform
        tensor([[ 5.1151e-03,  1.8073e-02,  2.2188e-02,  ...,  1.0431e-07,
                 -1.4761e-07,  1.8114e-07],
                [-2.6924e-03,  2.1860e-03,  1.0650e-02,  ...,  6.4122e-07,
                 -5.6159e-07,  4.8103e-07]])
        >>> sample_rate
        8000

    Example - Apply random speed perturbation to dataset
        >>>
        >>> # Load data from file, apply random speed perturbation
        >>> class RandomPerturbationFile(paddle.utils.data.Dataset):
        ...     \"\"\"Given flist, apply random speed perturbation
        ...
        ...     Suppose all the input files are at least one second long.
        ...     \"\"\"
        ...     def __init__(self, flist: List[str], sample_rate: int):
        ...         super().__init__()
        ...         self.flist = flist
        ...         self.sample_rate = sample_rate
        ...
        ...     def __getitem__(self, index):
        ...         speed = 0.5 + 1.5 * random.randn()
        ...         effects = [
        ...             ['gain', '-n', '-10'],  # apply 10 db attenuation
        ...             ['remix', '-'],  # merge all the channels
        ...             ['speed', f'{speed:.5f}'],  # duration is now 0.5 ~ 2.0 seconds.
        ...             ['rate', f'{self.sample_rate}'],
        ...             ['pad', '0', '1.5'],  # add 1.5 seconds silence at the end
        ...             ['trim', '0', '2'],  # get the first 2 seconds
        ...         ]
        ...         waveform, _ = paddleaudio.sox_effects.apply_effects_file(
        ...             self.flist[index], effects)
        ...         return waveform
        ...
        ...     def __len__(self):
        ...         return len(self.flist)
        ...
        >>> dataset = RandomPerturbationFile(file_list, sample_rate=8000)
        >>> loader = paddle.utils.data.DataLoader(dataset, batch_size=32)
        >>> for batch in loader:
        >>>     pass
    """
    if hasattr(path, "read"):
        ret = paddleaudio.apply_effects_fileobj(path, effects, normalize, channels_first, format)
        if ret is None:
            raise RuntimeError("Failed to load audio from {}".format(path))
        return (paddle.to_tensor(ret[0]), ret[1])
    path = os.fspath(path)
    ret = paddleaudio.sox_effects_apply_effects_file(path, effects, normalize, channels_first, format)
    if ret is not None:
        return (paddle.to_tensor(ret[0]), ret[1])
    raise RuntimeError("Failed to load audio from {}".format(path))