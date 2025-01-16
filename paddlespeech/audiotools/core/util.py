# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/audiotools/core/util.py)
import collections
import csv
import glob
import math
import numbers
import os
import random
import typing
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import ffmpeg
import librosa
import numpy as np
import paddle
import soundfile
from flatten_dict import flatten
from flatten_dict import unflatten

from .audio_signal import AudioSignal
from paddlespeech.utils import satisfy_paddle_version
from paddlespeech.vector.training.seeding import seed_everything

__all__ = [
    "exp_compat",
    "bool_index_compat",
    "bool_setitem_compat",
    "Info",
    "info",
    "ensure_tensor",
    "random_state",
    "find_audio",
    "read_sources",
    "choose_from_list_of_lists",
    "chdir",
    "move_to_device",
    "prepare_batch",
    "sample_from_dist",
    "format_figure",
    "default_collate",
    "collate",
    "hz_to_bin",
    "generate_chord_dataset",
]


def exp_compat(x):
    """
    Compute the exponential of the input tensor `x`.

    This function is designed to handle compatibility issues with PaddlePaddle versions below 2.6,
    which do not support the `exp` operation for complex tensors. In such cases, the computation
    is offloaded to NumPy.

    Args:
        x (paddle.Tensor): The input tensor for which to compute the exponential.

    Returns:
        paddle.Tensor: The result of the exponential operation, as a PaddlePaddle tensor.

    Notes:
        - If the PaddlePaddle version is 2.6 or above, the function uses `paddle.exp` directly.
        - For versions below 2.6, the tensor is first converted to a NumPy array, the exponential
          is computed using `np.exp`, and the result is then converted back to a PaddlePaddle tensor.
    """
    if satisfy_paddle_version("2.6"):
        return paddle.exp(x)
    else:
        x_np = x.cpu().numpy()
        return paddle.to_tensor(np.exp(x_np))


def bool_index_compat(x, mask):
    """
    Perform boolean indexing on the input tensor `x` using the provided `mask`.

    This function ensures compatibility with PaddlePaddle versions below 2.6, where boolean indexing
    may not be fully supported. For older versions, the operation is performed using NumPy.

    Args:
        x (paddle.Tensor): The input tensor to be indexed.
        mask (paddle.Tensor or int): The boolean mask or integer index used for indexing.

    Returns:
        paddle.Tensor: The result of the boolean indexing operation, as a PaddlePaddle tensor.

    Notes:
        - If the PaddlePaddle version is 2.6 or above, or if `mask` is an integer, the function uses
          Paddle's native indexing directly.
        - For versions below 2.6, the tensor and mask are converted to NumPy arrays, the indexing
          operation is performed using NumPy, and the result is converted back to a PaddlePaddle tensor.
    """
    if satisfy_paddle_version("2.6") or isinstance(mask, (int, list, slice)):
        return x[mask]
    else:
        x_np = x.cpu().numpy()[mask.cpu().numpy()]
        return paddle.to_tensor(x_np)


def bool_setitem_compat(x, mask, y):
    """
    Perform boolean assignment on the input tensor `x` using the provided `mask` and values `y`.

    This function ensures compatibility with PaddlePaddle versions below 2.6, where boolean assignment
    may not be fully supported. For older versions, the operation is performed using NumPy.

    Args:
        x (paddle.Tensor): The input tensor to be modified.
        mask (paddle.Tensor): The boolean mask used for assignment.
        y (paddle.Tensor): The values to assign to the selected elements of `x`.

    Returns:
        paddle.Tensor: The modified tensor after the assignment operation.

    Notes:
        - If the PaddlePaddle version is 2.6 or above, the function uses Paddle's native assignment directly.
        - For versions below 2.6, the tensor, mask, and values are converted to NumPy arrays, the assignment
          operation is performed using NumPy, and the result is converted back to a PaddlePaddle tensor.
    """
    if satisfy_paddle_version("2.6"):

        x[mask] = y
        return x
    else:
        x_np = x.cpu().numpy()
        x_np[mask.cpu().numpy()] = y.cpu().numpy()

        return paddle.to_tensor(x_np)


@dataclass
class Info:

    sample_rate: float
    num_frames: int

    @property
    def duration(self) -> float:
        return self.num_frames / self.sample_rate


def info_ffmpeg(audio_path: str):
    """
    Parameters
    ----------
    audio_path : str
        Path to audio file.
    """
    probe = ffmpeg.probe(audio_path)
    audio_streams = [
        stream for stream in probe['streams'] if stream['codec_type'] == 'audio'
    ]
    if not audio_streams:
        raise ValueError("No audio stream found in the file.")
    audio_stream = audio_streams[0]

    sample_rate = int(audio_stream['sample_rate'])
    duration = float(audio_stream['duration'])

    num_frames = int(duration * sample_rate)

    info = Info(sample_rate=sample_rate, num_frames=num_frames)
    return info


def info(audio_path: str):
    """

    Parameters
    ----------
    audio_path : str
        Path to audio file.
    """
    try:
        info = soundfile.info(str(audio_path))
        info = Info(sample_rate=info.samplerate, num_frames=info.frames)
    except:
        info = info_ffmpeg(str(audio_path))

    return info


def ensure_tensor(
        x: typing.Union[np.ndarray, paddle.Tensor, float, int],
        ndim: int=None,
        batch_size: int=None, ):
    """Ensures that the input ``x`` is a tensor of specified
    dimensions and batch size.

    Parameters
    ----------
    x : typing.Union[np.ndarray, paddle.Tensor, float, int]
        Data that will become a tensor on its way out.
    ndim : int, optional
        How many dimensions should be in the output, by default None
    batch_size : int, optional
        The batch size of the output, by default None

    Returns
    -------
    paddle.Tensor
        Modified version of ``x`` as a tensor.
    """
    if not paddle.is_tensor(x):
        x = paddle.to_tensor(x)
    if ndim is not None:
        assert x.ndim <= ndim
        while x.ndim < ndim:
            x = x.unsqueeze(-1)
    if batch_size is not None:
        if x.shape[0] != batch_size:
            shape = list(x.shape)
            shape[0] = batch_size
            x = paddle.expand(x, shape)
    return x


def _get_value(other):
    # 
    from . import AudioSignal

    if isinstance(other, AudioSignal):
        return other.audio_data
    return other


def random_state(seed: typing.Union[int, np.random.RandomState]):
    """
    Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : typing.Union[int, np.random.RandomState] or None
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    np.random.RandomState
        Random state object.

    Raises
    ------
    ValueError
        If seed is not valid, an error is thrown.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    elif isinstance(seed, (numbers.Integral, np.integer, int)):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    else:
        raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                         " instance" % seed)


@contextmanager
def _close_temp_files(tmpfiles: list):
    """Utility function for creating a context and closing all temporary files
    once the context is exited. For correct functionality, all temporary file
    handles created inside the context must be appended to the ```tmpfiles```
    list.

    This function is taken wholesale from Scaper.

    Parameters
    ----------
    tmpfiles : list
        List of temporary file handles
    """

    def _close():
        for t in tmpfiles:
            try:
                t.close()
                os.unlink(t.name)
            except:
                pass

    try:
        yield
    except:
        _close()
        raise
    _close()


AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3"]


def find_audio(folder: str, ext: List[str]=AUDIO_EXTENSIONS):
    """Finds all audio files in a directory recursively.
    Returns a list.

    Parameters
    ----------
    folder : str
        Folder to look for audio files in, recursively.
    ext : List[str], optional
        Extensions to look for without the ., by default
        ``['.wav', '.flac', '.mp3', '.mp4']``.
    """
    folder = Path(folder)
    # Take care of case where user has passed in an audio file directly
    # into one of the calling functions.
    if str(folder).endswith(tuple(ext)):
        # if, however, there's a glob in the path, we need to
        # return the glob, not the file.
        if "*" in str(folder):
            return glob.glob(str(folder), recursive=("**" in str(folder)))
        else:
            return [folder]

    files = []
    for x in ext:
        files += folder.glob(f"**/*{x}")
    return files


def read_sources(
        sources: List[str],
        remove_empty: bool=True,
        relative_path: str="",
        ext: List[str]=AUDIO_EXTENSIONS, ):
    """Reads audio sources that can either be folders
    full of audio files, or CSV files that contain paths
    to audio files. CSV files that adhere to the expected
    format can be generated by
    :py:func:`audiotools.data.preprocess.create_csv`.

    Parameters
    ----------
    sources : List[str]
        List of audio sources to be converted into a
        list of lists of audio files.
    remove_empty : bool, optional
        Whether or not to remove rows with an empty "path"
        from each CSV file, by default True.

    Returns
    -------
    list
        List of lists of rows of CSV files.
    """
    files = []
    relative_path = Path(relative_path)
    for source in sources:
        source = str(source)
        _files = []
        if source.endswith(".csv"):
            with open(source, "r") as f:
                reader = csv.DictReader(f)
                for x in reader:
                    if remove_empty and x["path"] == "":
                        continue
                    if x["path"] != "":
                        x["path"] = str(relative_path / x["path"])
                    _files.append(x)
        else:
            for x in find_audio(source, ext=ext):
                x = str(relative_path / x)
                _files.append({"path": x})
        files.append(sorted(_files, key=lambda x: x["path"]))
    return files


def choose_from_list_of_lists(state: np.random.RandomState,
                              list_of_lists: list,
                              p: float=None):
    """Choose a single item from a list of lists.

    Parameters
    ----------
    state : np.random.RandomState
        Random state to use when choosing an item.
    list_of_lists : list
        A list of lists from which items will be drawn.
    p : float, optional
        Probabilities of each list, by default None

    Returns
    -------
    typing.Any
        An item from the list of lists.
    """
    source_idx = state.choice(list(range(len(list_of_lists))), p=p)
    item_idx = state.randint(len(list_of_lists[source_idx]))
    return list_of_lists[source_idx][item_idx], source_idx, item_idx


@contextmanager
def chdir(newdir: typing.Union[Path, str]):
    """
    Context manager for switching directories to run a
    function. Useful for when you want to use relative
    paths to different runs.

    Parameters
    ----------
    newdir : typing.Union[Path, str]
        Directory to switch to.
    """
    curdir = os.getcwd()
    try:
        os.chdir(newdir)
        yield
    finally:
        os.chdir(curdir)


def move_to_device(data, device):
    if device is None or device == "":
        return data
    elif device == 'cpu':
        return paddle.to_tensor(data, place=paddle.CPUPlace())
    elif device in ('gpu', 'cuda'):
        return paddle.to_tensor(data, place=paddle.CUDAPlace())
    else:
        device = device.replace("cuda", "gpu") if "cuda" in device else device
        return data.to(device)


def prepare_batch(batch: typing.Union[dict, list, paddle.Tensor],
                  device: str="cpu"):
    """Moves items in a batch (typically generated by a DataLoader as a list
    or a dict) to the specified device. This works even if dictionaries
    are nested.

    Parameters
    ----------
    batch : typing.Union[dict, list, paddle.Tensor]
        Batch, typically generated by a dataloader, that will be moved to
        the device.
    device : str, optional
        Device to move batch to, by default "cpu"

    Returns
    -------
    typing.Union[dict, list, paddle.Tensor]
        Batch with all values moved to the specified device.
    """
    device = device.replace("cuda", "gpu")
    if isinstance(batch, dict):
        batch = flatten(batch)
        for key, val in batch.items():
            try:
                # batch[key] = val.to(device)
                batch[key] = move_to_device(val, device)
            except:
                pass
        batch = unflatten(batch)
    elif paddle.is_tensor(batch):
        # batch = batch.to(device)
        batch = move_to_device(batch, device)
    elif isinstance(batch, list):
        for i in range(len(batch)):
            try:
                batch[i] = batch[i].to(device)
            except:
                pass
    return batch


def sample_from_dist(dist_tuple: tuple, state: np.random.RandomState=None):
    """Samples from a distribution defined by a tuple. The first
    item in the tuple is the distribution type, and the rest of the
    items are arguments to that distribution. The distribution function
    is gotten from the ``np.random.RandomState`` object.

    Parameters
    ----------
    dist_tuple : tuple
        Distribution tuple
    state : np.random.RandomState, optional
        Random state, or seed to use, by default None

    Returns
    -------
    typing.Union[float, int, str]
        Draw from the distribution.

    Examples
    --------
    Sample from a uniform distribution:

    >>> dist_tuple = ("uniform", 0, 1)
    >>> sample_from_dist(dist_tuple)

    Sample from a constant distribution:

    >>> dist_tuple = ("const", 0)
    >>> sample_from_dist(dist_tuple)

    Sample from a normal distribution:

    >>> dist_tuple = ("normal", 0, 0.5)
    >>> sample_from_dist(dist_tuple)

    """
    if dist_tuple[0] == "const":
        return dist_tuple[1]
    state = random_state(state)
    dist_fn = getattr(state, dist_tuple[0])
    return dist_fn(*dist_tuple[1:])


BASE_SIZE = 864
DEFAULT_FIG_SIZE = (9, 3)


def format_figure(
        fig_size: tuple=None,
        title: str=None,
        fig=None,
        format_axes: bool=True,
        format: bool=True,
        font_color: str="white", ):
    """Prettifies the spectrogram and waveform plots. A title
    can be inset into the top right corner, and the axes can be
    inset into the figure, allowing the data to take up the entire
    image. Used in

    - :py:func:`audiotools.core.display.DisplayMixin.specshow`
    - :py:func:`audiotools.core.display.DisplayMixin.waveplot`
    - :py:func:`audiotools.core.display.DisplayMixin.wavespec`

    Parameters
    ----------
    fig_size : tuple, optional
        Size of figure, by default (9, 3)
    title : str, optional
        Title to inset in top right, by default None
    fig : matplotlib.figure.Figure, optional
        Figure object, if None ``plt.gcf()`` will be used, by default None
    format_axes : bool, optional
        Format the axes to be inside the figure, by default True
    format : bool, optional
        This formatting can be skipped entirely by passing ``format=False``
        to any of the plotting functions that use this formater, by default True
    font_color : str, optional
        Color of font of axes, by default "white"
    """
    import matplotlib
    import matplotlib.pyplot as plt

    if fig_size is None:
        fig_size = DEFAULT_FIG_SIZE
    if not format:
        return
    if fig is None:
        fig = plt.gcf()
    fig.set_size_inches(*fig_size)
    axs = fig.axes

    pixels = (fig.get_size_inches() * fig.dpi)[0]
    font_scale = pixels / BASE_SIZE

    if format_axes:
        axs = fig.axes

        for ax in axs:
            ymin, _ = ax.get_ylim()
            xmin, _ = ax.get_xlim()

            ticks = ax.get_yticks()
            for t in ticks[2:-1]:
                t = axs[0].annotate(
                    f"{(t / 1000):2.1f}k",
                    xy=(xmin, t),
                    xycoords="data",
                    xytext=(5, -5),
                    textcoords="offset points",
                    ha="left",
                    va="top",
                    color=font_color,
                    fontsize=12 * font_scale,
                    alpha=0.75, )

            ticks = ax.get_xticks()[2:]
            for t in ticks[:-1]:
                t = axs[0].annotate(
                    f"{t:2.1f}s",
                    xy=(t, ymin),
                    xycoords="data",
                    xytext=(5, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color=font_color,
                    fontsize=12 * font_scale,
                    alpha=0.75, )

            ax.margins(0, 0)
            ax.set_axis_off()
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    if title is not None:
        t = axs[0].annotate(
            title,
            xy=(1, 1),
            xycoords="axes fraction",
            fontsize=20 * font_scale,
            xytext=(-5, -5),
            textcoords="offset points",
            ha="right",
            va="top",
            color="white", )
        t.set_bbox(dict(facecolor="black", alpha=0.5, edgecolor="black"))


_default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def collate_tensor_fn(
        batch,
        *,
        collate_fn_map: Optional[Dict[Union[type, Tuple[type, ...]],
                                      Callable]]=None, ):
    out = paddle.stack(batch, axis=0)
    return out


def collate_float_fn(
        batch,
        *,
        collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]],
                                      Callable]]=None, ):
    return paddle.to_tensor(batch, dtype=paddle.float64)


def collate_int_fn(
        batch,
        *,
        collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]],
                                      Callable]]=None, ):
    return paddle.to_tensor(batch)


def collate_str_fn(
        batch,
        *,
        collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]],
                                      Callable]]=None, ):
    return batch


default_collate_fn_map: Dict[Union[Type, Tuple[Type, ...]], Callable] = {
    paddle.Tensor: collate_tensor_fn
}
default_collate_fn_map[float] = collate_float_fn
default_collate_fn_map[int] = collate_int_fn
default_collate_fn_map[str] = collate_str_fn
default_collate_fn_map[bytes] = collate_str_fn


def default_collate(batch,
                    *,
                    collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]],
                                                  Callable]]=None):
    r"""
    General collate function that handles collection type of element within each batch.

    The function also opens function registry to deal with specific element types. `default_collate_fn_map`
    provides default collate functions for tensors, numpy arrays, numbers and strings.

    Args:
        batch: a single batch to be collated
        collate_fn_map: Optional dictionary mapping from element type to the corresponding collate function.
            If the element type isn't present in this dictionary,
            this function will go through each key of the dictionary in the insertion order to
            invoke the corresponding collate function if the element type is a subclass of the key.
    Note:
        Each collate function requires a positional argument for batch and a keyword argument
        for the dictionary of collate functions as `collate_fn_map`.
    """
    elem = batch[0]
    elem_type = type(elem)

    if collate_fn_map is not None:
        if elem_type in collate_fn_map:
            return collate_fn_map[elem_type](
                batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](
                    batch, collate_fn_map=collate_fn_map)

    if isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({
                key: default_collate(
                    [d[key] for d in batch], collate_fn_map=collate_fn_map)
                for key in elem
            })
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {
                key: default_collate(
                    [d[key] for d in batch], collate_fn_map=collate_fn_map)
                for key in elem
            }
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(default_collate(
            samples, collate_fn_map=collate_fn_map) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                "each element in list of batch should be of equal size")
        transposed = list(
            zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [
                default_collate(samples, collate_fn_map=collate_fn_map)
                for samples in transposed
            ]  # Backwards compatibility.
        else:
            try:
                return elem_type([
                    default_collate(samples, collate_fn_map=collate_fn_map)
                    for samples in transposed
                ])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [
                    default_collate(samples, collate_fn_map=collate_fn_map)
                    for samples in transposed
                ]

    raise TypeError(_default_collate_err_msg_format.format(elem_type))


def collate(list_of_dicts: list, n_splits: int=None):
    """Collates a list of dictionaries (e.g. as returned by a
    dataloader) into a dictionary with batched values. This routine
    uses the default torch collate function for everything
    except AudioSignal objects, which are handled by the
    :py:func:`audiotools.core.audio_signal.AudioSignal.batch`
    function.

    This function takes n_splits to enable splitting a batch
    into multiple sub-batches for the purposes of gradient accumulation,
    etc.

    Parameters
    ----------
    list_of_dicts : list
        List of dictionaries to be collated.
    n_splits : int
        Number of splits to make when creating the batches (split into
        sub-batches). Useful for things like gradient accumulation.

    Returns
    -------
    dict
        Dictionary containing batched data.
    """

    batches = []
    list_len = len(list_of_dicts)

    return_list = False if n_splits is None else True
    n_splits = 1 if n_splits is None else n_splits
    n_items = int(math.ceil(list_len / n_splits))

    for i in range(0, list_len, n_items):
        # Flatten the dictionaries to avoid recursion.
        list_of_dicts_ = [flatten(d) for d in list_of_dicts[i:i + n_items]]
        dict_of_lists = {
            k: [dic[k] for dic in list_of_dicts_]
            for k in list_of_dicts_[0]
        }

        batch = {}
        for k, v in dict_of_lists.items():
            if isinstance(v, list):
                if all(isinstance(s, AudioSignal) for s in v):
                    batch[k] = AudioSignal.batch(v, pad_signals=True)
                else:
                    batch[k] = default_collate(
                        v, collate_fn_map=default_collate_fn_map)
        batches.append(unflatten(batch))

    batches = batches[0] if not return_list else batches
    return batches


def hz_to_bin(hz: paddle.Tensor, n_fft: int, sample_rate: int):
    """Closest frequency bin given a frequency, number
    of bins, and a sampling rate.

    Parameters
    ----------
    hz : paddle.Tensor
       Tensor of frequencies in Hz.
    n_fft : int
        Number of FFT bins.
    sample_rate : int
        Sample rate of audio.

    Returns
    -------
    paddle.Tensor
        Closest bins to the data.
    """
    shape = hz.shape
    hz = hz.reshape([-1])
    freqs = paddle.linspace(0, sample_rate / 2, 2 + n_fft // 2)
    hz = paddle.clip(hz, max=sample_rate / 2).astype(freqs.dtype)

    closest = (hz[None, :] - freqs[:, None]).abs()
    closest_bins = closest.argmin(axis=0)

    return closest_bins.reshape(shape)


def generate_chord_dataset(
        max_voices: int=8,
        sample_rate: int=44100,
        num_items: int=5,
        duration: float=1.0,
        min_note: str="C2",
        max_note: str="C6",
        output_dir: Path="chords", ):
    """
    Generates a toy multitrack dataset of chords, synthesized from sine waves.


    Parameters
    ----------
    max_voices : int, optional
        Maximum number of voices in a chord, by default 8
    sample_rate : int, optional
        Sample rate of audio, by default 44100
    num_items : int, optional
        Number of items to generate, by default 5
    duration : float, optional
        Duration of each item, by default 1.0
    min_note : str, optional
        Minimum note in the dataset, by default "C2"
    max_note : str, optional
        Maximum note in the dataset, by default "C6"
    output_dir : Path, optional
        Directory to save the dataset, by default "chords"

    """
    import librosa
    from . import AudioSignal
    from ..data.preprocess import create_csv

    min_midi = librosa.note_to_midi(min_note)
    max_midi = librosa.note_to_midi(max_note)

    tracks = []
    for idx in range(num_items):
        track = {}
        # figure out how many voices to put in this track
        num_voices = random.randint(1, max_voices)
        for voice_idx in range(num_voices):
            # choose some random params
            midinote = random.randint(min_midi, max_midi)
            dur = random.uniform(0.85 * duration, duration)

            sig = AudioSignal.wave(
                frequency=librosa.midi_to_hz(midinote),
                duration=dur,
                sample_rate=sample_rate,
                shape="sine", )
            track[f"voice_{voice_idx}"] = sig
        tracks.append(track)

    # save the tracks to disk
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    for idx, track in enumerate(tracks):
        track_dir = output_dir / f"track_{idx}"
        track_dir.mkdir(exist_ok=True)
        for voice_name, sig in track.items():
            sig.write(track_dir / f"{voice_name}.wav")

    all_voices = list(set([k for track in tracks for k in track.keys()]))
    voice_lists = {voice: [] for voice in all_voices}
    for track in tracks:
        for voice_name in all_voices:
            if voice_name in track:
                voice_lists[voice_name].append(track[voice_name].path_to_file)
            else:
                voice_lists[voice_name].append("")

    for voice_name, paths in voice_lists.items():
        create_csv(paths, output_dir / f"{voice_name}.csv", loudness=True)

    return output_dir
