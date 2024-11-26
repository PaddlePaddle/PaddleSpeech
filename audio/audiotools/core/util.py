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
from typing import Dict
from typing import List
from typing import Optional, Union, Type, Any, Callable, Tuple, NamedTuple, Iterable
import collections
import librosa
import numpy as np
import paddle
import soundfile
from audio_signal import AudioSignal
from flatten_dict import flatten
from flatten_dict import unflatten

from ..data.preprocess import create_csv


@dataclass
class Info:

    sample_rate: float
    num_frames: int

    @property
    def duration(self) -> float:
        return self.num_frames / self.sample_rate


def info(audio_path: str):
    """✅

    Parameters
    ----------
    audio_path : str
        Path to audio file.
    """
    info = soundfile.info(str(audio_path))
    info = Info(sample_rate=info.samplerate, num_frames=info.frames)

    return info


def ensure_tensor(
    x: typing.Union[np.ndarray, paddle.Tensor, float, int],
    ndim: int = None,
    batch_size: int = None,
):
    """✅Ensures that the input ``x`` is a tensor of specified
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
    # ✅
    # from . import AudioSignal
    from audio_signal import AudioSignal

    if isinstance(other, AudioSignal):
        return other.audio_data
    return other


def random_state(seed: typing.Union[int, np.random.RandomState]):
    """✅
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
        raise ValueError("%r cannot be used to seed a numpy.random.RandomState" " instance" % seed)


def seed(random_seed):
    """✅
    Seeds all random states with the same random seed
    for reproducibility. Seeds ``numpy``, ``random`` and ``paddle``
    random generators.

    Args:
        random_seed (int): integer corresponding to random seed to
        use.
    """

    paddle.seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


@contextmanager
def _close_temp_files(tmpfiles: list):
    """✅Utility function for creating a context and closing all temporary files
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
    except:  # pragma: no cover
        _close()
        raise
    _close()


AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".mp4"]


def find_audio(folder: str, ext: List[str] = AUDIO_EXTENSIONS):
    """✅Finds all audio files in a directory recursively.
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
    remove_empty: bool = True,
    relative_path: str = "",
    ext: List[str] = AUDIO_EXTENSIONS,
):
    """✅Reads audio sources that can either be folders
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


def choose_from_list_of_lists(state: np.random.RandomState, list_of_lists: list, p: float = None):
    """✅Choose a single item from a list of lists.

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
    """✅
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


def prepare_batch(batch: typing.Union[dict, list, paddle.Tensor], device: str = "cpu"):
    """✅Moves items in a batch (typically generated by a DataLoader as a list
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
                batch[key] = val.to(device)
            except:
                pass
        batch = unflatten(batch)
    elif paddle.is_tensor(batch):
        batch = batch.to(device)
    elif isinstance(batch, list):
        for i in range(len(batch)):
            try:
                batch[i] = batch[i].to(device)
            except:
                pass
    return batch


def sample_from_dist(dist_tuple: tuple, state: np.random.RandomState = None):
    """✅Samples from a distribution defined by a tuple. The first
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
    fig_size: tuple = None,
    title: str = None,
    fig=None,
    format_axes: bool = True,
    format: bool = True,
    font_color: str = "white",
):
    """✅Prettifies the spectrogram and waveform plots. A title
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
                    alpha=0.75,
                )

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
                    alpha=0.75,
                )

            ax.margins(0, 0)
            ax.set_axis_off()
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

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
            color="white",
        )
        t.set_bbox(dict(facecolor="black", alpha=0.5, edgecolor="black"))


_default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, " "dicts or lists; found {}"
)


def default_collate(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
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
            return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

        for collate_type in collate_fn_map:
            if isinstance(elem, collate_type):
                return collate_fn_map[collate_type](batch, collate_fn_map=collate_fn_map)

    if isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate(samples, collate_fn_map=collate_fn_map) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [
                collate(samples, collate_fn_map=collate_fn_map) for samples in transposed
            ]  # Backwards compatibility.
        else:
            try:
                return elem_type([collate(samples, collate_fn_map=collate_fn_map) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]

    raise TypeError(_default_collate_err_msg_format.format(elem_type))


def collate(list_of_dicts: list, n_splits: int = None):
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
        list_of_dicts_ = [flatten(d) for d in list_of_dicts[i : i + n_items]]
        dict_of_lists = {k: [dic[k] for dic in list_of_dicts_] for k in list_of_dicts_[0]}

        batch = {}
        for k, v in dict_of_lists.items():
            if isinstance(v, list):
                if all(isinstance(s, AudioSignal) for s in v):
                    batch[k] = AudioSignal.batch(v, pad_signals=True)
                else:
                    # Borrow the default collate fn from torch.
                    batch[k] = default_collate(v)
        batches.append(unflatten(batch))

    batches = batches[0] if not return_list else batches
    return batches
