# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/audiotools/data/datasets.py)
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import paddle
from paddle.io import DistributedBatchSampler
from paddle.io import SequenceSampler

from ..core import AudioSignal
from ..core import util

__all__ = [
    "AudioLoader", "AudioDataset", "ConcatDataset",
    "ResumableDistributedSampler", "ResumableSequentialSampler"
]


class AudioLoader:
    """Loads audio endlessly from a list of audio sources
    containing paths to audio files. Audio sources can be
    folders full of audio files (which are found via file
    extension) or by providing a CSV file which contains paths
    to audio files.

    Parameters
    ----------
    sources : List[str], optional
        Sources containing folders, or CSVs with
        paths to audio files, by default None
    weights : List[float], optional
        Weights to sample audio files from each source, by default None
    relative_path : str, optional
        Path audio should be loaded relative to, by default ""
    transform : Callable, optional
        Transform to instantiate alongside audio sample,
        by default None
    ext : List[str]
        List of extensions to find audio within each source by. Can
        also be a file name (e.g. "vocals.wav"). by default
        ``['.wav', '.flac', '.mp3', '.mp4']``.
    shuffle: bool
        Whether to shuffle the files within the dataloader. Defaults to True.
    shuffle_state: int
        State to use to seed the shuffle of the files.
    """

    def __init__(
            self,
            sources: List[str]=None,
            weights: List[float]=None,
            transform: Callable=None,
            relative_path: str="",
            ext: List[str]=util.AUDIO_EXTENSIONS,
            shuffle: bool=True,
            shuffle_state: int=0, ):
        self.audio_lists = util.read_sources(
            sources, relative_path=relative_path, ext=ext)

        self.audio_indices = [(src_idx, item_idx)
                              for src_idx, src in enumerate(self.audio_lists)
                              for item_idx in range(len(src))]
        if shuffle:
            state = util.random_state(shuffle_state)
            state.shuffle(self.audio_indices)

        self.sources = sources
        self.weights = weights
        self.transform = transform

    def __call__(
            self,
            state,
            sample_rate: int,
            duration: float,
            loudness_cutoff: float=-40,
            num_channels: int=1,
            offset: float=None,
            source_idx: int=None,
            item_idx: int=None,
            global_idx: int=None, ):
        if source_idx is not None and item_idx is not None:
            try:
                audio_info = self.audio_lists[source_idx][item_idx]
            except:
                audio_info = {"path": "none"}
        elif global_idx is not None:
            source_idx, item_idx = self.audio_indices[global_idx %
                                                      len(self.audio_indices)]
            audio_info = self.audio_lists[source_idx][item_idx]
        else:
            audio_info, source_idx, item_idx = util.choose_from_list_of_lists(
                state, self.audio_lists, p=self.weights)

        path = audio_info["path"]
        signal = AudioSignal.zeros(duration, sample_rate, num_channels)

        if path != "none":
            if offset is None:
                signal = AudioSignal.salient_excerpt(
                    path,
                    duration=duration,
                    state=state,
                    loudness_cutoff=loudness_cutoff, )
            else:
                signal = AudioSignal(
                    path,
                    offset=offset,
                    duration=duration, )

        if num_channels == 1:
            signal = signal.to_mono()
        signal = signal.resample(sample_rate)

        if signal.duration < duration:
            signal = signal.zero_pad_to(int(duration * sample_rate))

        for k, v in audio_info.items():
            signal.metadata[k] = v

        item = {
            "signal": signal,
            "source_idx": source_idx,
            "item_idx": item_idx,
            "source": str(self.sources[source_idx]),
            "path": str(path),
        }
        if self.transform is not None:
            item["transform_args"] = self.transform.instantiate(
                state, signal=signal)
        return item


def default_matcher(x, y):
    return Path(x).parent == Path(y).parent


def align_lists(lists, matcher: Callable=default_matcher):
    longest_list = lists[np.argmax([len(l) for l in lists])]
    for i, x in enumerate(longest_list):
        for l in lists:
            if i >= len(l):
                l.append({"path": "none"})
            elif not matcher(l[i]["path"], x["path"]):
                l.insert(i, {"path": "none"})
    return lists


class AudioDataset:
    """Loads audio from multiple loaders (with associated transforms)
    for a specified number of samples. Excerpts are drawn randomly
    of the specified duration, above a specified loudness threshold
    and are resampled on the fly to the desired sample rate
    (if it is different from the audio source sample rate).

    This takes either a single AudioLoader object,
    a dictionary of AudioLoader objects, or a dictionary of AudioLoader
    objects. Each AudioLoader is called by the dataset, and the
    result is placed in the output dictionary. A transform can also be
    specified for the entire dataset, rather than for each specific
    loader. This transform can be applied to the output of all the
    loaders if desired.

    AudioLoader objects can be specified as aligned, which means the
    loaders correspond to multitrack audio (e.g. a vocals, bass,
    drums, and other loader for multitrack music mixtures).


    Parameters
    ----------
    loaders : Union[AudioLoader, List[AudioLoader], Dict[str, AudioLoader]]
        AudioLoaders to sample audio from.
    sample_rate : int
        Desired sample rate.
    n_examples : int, optional
        Number of examples (length of dataset), by default 1000
    duration : float, optional
        Duration of audio samples, by default 0.5
    loudness_cutoff : float, optional
        Loudness cutoff threshold for audio samples, by default -40
    num_channels : int, optional
        Number of channels in output audio, by default 1
    transform : Callable, optional
        Transform to instantiate alongside each dataset item, by default None
    aligned : bool, optional
        Whether the loaders should be sampled in an aligned manner (e.g. same
        offset, duration, and matched file name), by default False
    shuffle_loaders : bool, optional
        Whether to shuffle the loaders before sampling from them, by default False
    matcher : Callable
        How to match files from adjacent audio lists (e.g. for a multitrack audio loader),
        by default uses the parent directory of each file.
    without_replacement : bool
        Whether to choose files with or without replacement, by default True.


    Examples
    --------
    >>> from audio.audiotools.data.datasets import AudioLoader
    >>> from audio.audiotools.data.datasets import AudioDataset
    >>> from audio.audiotools import transforms as tfm
    >>> import numpy as np
    >>>
    >>> loaders = [
    >>>     AudioLoader(
    >>>         sources=[f"tests/audiotools/audio/spk"],
    >>>         transform=tfm.Equalizer(),
    >>>         ext=["wav"],
    >>>     )
    >>>     for i in range(5)
    >>> ]
    >>>
    >>> dataset = AudioDataset(
    >>>     loaders = loaders,
    >>>     sample_rate = 44100,
    >>>     duration = 1.0,
    >>>     transform = tfm.RescaleAudio(),
    >>> )
    >>>
    >>> item = dataset[np.random.randint(len(dataset))]
    >>>
    >>> for i in range(len(loaders)):
    >>>     item[i]["signal"] = loaders[i].transform(
    >>>         item[i]["signal"], **item[i]["transform_args"]
    >>>     )
    >>>     item[i]["signal"].widget(i)
    >>>
    >>> mix = sum([item[i]["signal"] for i in range(len(loaders))])
    >>> mix = dataset.transform(mix, **item["transform_args"])
    >>> mix.widget("mix")

    Below is an example of how one could load MUSDB multitrack data:

    >>> from audio import audiotools as at
    >>> from pathlib import Path
    >>> from audio.audiotools import transforms as tfm
    >>> import numpy as np
    >>> import torch
    >>>
    >>> def build_dataset(
    >>>     sample_rate: int = 44100,
    >>>     duration: float = 5.0,
    >>>     musdb_path: str = "~/.data/musdb/",
    >>> ):
    >>>     musdb_path = Path(musdb_path).expanduser()
    >>>     loaders = {
    >>>         src: at.datasets.AudioLoader(
    >>>             sources=[musdb_path],
    >>>             transform=tfm.Compose(
    >>>                 tfm.VolumeNorm(("uniform", -20, -10)),
    >>>                 tfm.Silence(prob=0.1),
    >>>             ),
    >>>             ext=[f"{src}.wav"],
    >>>         )
    >>>         for src in ["vocals", "bass", "drums", "other"]
    >>>     }
    >>>
    >>>     dataset = at.datasets.AudioDataset(
    >>>         loaders=loaders,
    >>>         sample_rate=sample_rate,
    >>>         duration=duration,
    >>>         num_channels=1,
    >>>         aligned=True,
    >>>         transform=tfm.RescaleAudio(),
    >>>         shuffle_loaders=True,
    >>>     )
    >>>     return dataset, list(loaders.keys())
    >>>
    >>> train_data, sources = build_dataset()
    >>> dataloader = torch.utils.data.DataLoader(
    >>>     train_data,
    >>>     batch_size=16,
    >>>     num_workers=0,
    >>>     collate_fn=train_data.collate,
    >>> )
    >>> batch = next(iter(dataloader))
    >>>
    >>> for k in sources:
    >>>     src = batch[k]
    >>>     src["transformed"] = train_data.loaders[k].transform(
    >>>         src["signal"].clone(), **src["transform_args"]
    >>>     )
    >>>
    >>> mixture = sum(batch[k]["transformed"] for k in sources)
    >>> mixture = train_data.transform(mixture, **batch["transform_args"])
    >>>
    >>> # Say a model takes the mix and gives back (n_batch, n_src, n_time).
    >>> # Construct the targets:
    >>> targets = at.AudioSignal.batch([batch[k]["transformed"] for k in sources], dim=1)

    Similarly, here's example code for loading Slakh data:

    >>> from audio import audiotools as at
    >>> from pathlib import Path
    >>> from audio.audiotools import transforms as tfm
    >>> import numpy as np
    >>> import torch
    >>> import glob
    >>>
    >>> def build_dataset(
    >>>     sample_rate: int = 16000,
    >>>     duration: float = 10.0,
    >>>     slakh_path: str = "~/.data/slakh/",
    >>> ):
    >>>     slakh_path = Path(slakh_path).expanduser()
    >>>
    >>>     # Find the max number of sources in Slakh
    >>>     src_names = [x.name for x in list(slakh_path.glob("**/*.wav"))  if "S" in str(x.name)]
    >>>     n_sources = len(list(set(src_names)))
    >>>
    >>>     loaders = {
    >>>         f"S{i:02d}": at.datasets.AudioLoader(
    >>>             sources=[slakh_path],
    >>>             transform=tfm.Compose(
    >>>                 tfm.VolumeNorm(("uniform", -20, -10)),
    >>>                 tfm.Silence(prob=0.1),
    >>>             ),
    >>>             ext=[f"S{i:02d}.wav"],
    >>>         )
    >>>         for i in range(n_sources)
    >>>     }
    >>>     dataset = at.datasets.AudioDataset(
    >>>         loaders=loaders,
    >>>         sample_rate=sample_rate,
    >>>         duration=duration,
    >>>         num_channels=1,
    >>>         aligned=True,
    >>>         transform=tfm.RescaleAudio(),
    >>>         shuffle_loaders=False,
    >>>     )
    >>>
    >>>     return dataset, list(loaders.keys())
    >>>
    >>> train_data, sources = build_dataset()
    >>> dataloader = torch.utils.data.DataLoader(
    >>>     train_data,
    >>>     batch_size=16,
    >>>     num_workers=0,
    >>>     collate_fn=train_data.collate,
    >>> )
    >>> batch = next(iter(dataloader))
    >>>
    >>> for k in sources:
    >>>     src = batch[k]
    >>>     src["transformed"] = train_data.loaders[k].transform(
    >>>         src["signal"].clone(), **src["transform_args"]
    >>>     )
    >>>
    >>> mixture = sum(batch[k]["transformed"] for k in sources)
    >>> mixture = train_data.transform(mixture, **batch["transform_args"])

    """

    def __init__(
            self,
            loaders: Union[AudioLoader, List[AudioLoader], Dict[str,
                                                                AudioLoader]],
            sample_rate: int,
            n_examples: int=1000,
            duration: float=0.5,
            offset: float=None,
            loudness_cutoff: float=-40,
            num_channels: int=1,
            transform: Callable=None,
            aligned: bool=False,
            shuffle_loaders: bool=False,
            matcher: Callable=default_matcher,
            without_replacement: bool=True, ):
        # Internally we convert loaders to a dictionary
        if isinstance(loaders, list):
            loaders = {i: l for i, l in enumerate(loaders)}
        elif isinstance(loaders, AudioLoader):
            loaders = {0: loaders}

        self.loaders = loaders
        self.loudness_cutoff = loudness_cutoff
        self.num_channels = num_channels

        self.length = n_examples
        self.transform = transform
        self.sample_rate = sample_rate
        self.duration = duration
        self.offset = offset
        self.aligned = aligned
        self.shuffle_loaders = shuffle_loaders
        self.without_replacement = without_replacement

        if aligned:
            loaders_list = list(loaders.values())
            for i in range(len(loaders_list[0].audio_lists)):
                input_lists = [l.audio_lists[i] for l in loaders_list]
                # Alignment happens in-place
                align_lists(input_lists, matcher)

    def __getitem__(self, idx):
        state = util.random_state(idx)
        offset = None if self.offset is None else self.offset
        item = {}

        keys = list(self.loaders.keys())
        if self.shuffle_loaders:
            state.shuffle(keys)

        loader_kwargs = {
            "state": state,
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "loudness_cutoff": self.loudness_cutoff,
            "num_channels": self.num_channels,
            "global_idx": idx if self.without_replacement else None,
        }

        # Draw item from first loader
        loader = self.loaders[keys[0]]
        item[keys[0]] = loader(**loader_kwargs)

        for key in keys[1:]:
            loader = self.loaders[key]
            if self.aligned:
                # Path mapper takes the current loader + everything
                # returned by the first loader.
                offset = item[keys[0]]["signal"].metadata["offset"]
                loader_kwargs.update({
                    "offset": offset,
                    "source_idx": item[keys[0]]["source_idx"],
                    "item_idx": item[keys[0]]["item_idx"],
                })
            item[key] = loader(**loader_kwargs)

        # Sort dictionary back into original order
        keys = list(self.loaders.keys())
        item = {k: item[k] for k in keys}

        item["idx"] = idx
        if self.transform is not None:
            item["transform_args"] = self.transform.instantiate(
                state=state, signal=item[keys[0]]["signal"])

        # If there's only one loader, pop it up
        # to the main dictionary, instead of keeping it
        # nested.
        if len(keys) == 1:
            item.update(item.pop(keys[0]))

        return item

    def __len__(self):
        return self.length

    @staticmethod
    def collate(list_of_dicts: Union[list, dict], n_splits: int=None):
        """Collates items drawn from this dataset. Uses
        :py:func:`audiotools.core.util.collate`.

        Parameters
        ----------
        list_of_dicts : typing.Union[list, dict]
            Data drawn from each item.
        n_splits : int
            Number of splits to make when creating the batches (split into
            sub-batches). Useful for things like gradient accumulation.

        Returns
        -------
        dict
            Dictionary of batched data.
        """
        return util.collate(list_of_dicts, n_splits=n_splits)


class ConcatDataset(AudioDataset):
    # 
    def __init__(self, datasets: list):
        self.datasets = datasets

    def __len__(self):
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, idx):
        dataset = self.datasets[idx % len(self.datasets)]
        return dataset[idx // len(self.datasets)]


class ResumableDistributedSampler(DistributedBatchSampler):
    """Distributed sampler that can be resumed from a given start index."""

    def __init__(self,
                 dataset,
                 batch_size,
                 start_idx: int=None,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 drop_last=False):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last, )
        # Start index, allows to resume an experiment at the index it was
        if start_idx is not None:
            self.start_idx = start_idx // self.num_replicas
        else:
            self.start_idx = 0
        # 重新计算样本总数，因为 DistributedBatchSampler 的 __len__ 方法是基于 shuffle 后的样本总数计算的
        self.total_size = len(self.dataset) if not shuffle else len(
            self.indices)

    def __iter__(self):
        # 由于 Paddle 的 DistributedBatchSampler 直接返回 batch，我们需要将其展开为单个索引
        indices_iter = iter(super().__iter__())
        # 跳过前面的 start_idx 个 batch
        for _ in range(self.start_idx):
            next(indices_iter)

        current_idx = 0
        while True:
            batch_indices = next(indices_iter, None)
            if batch_indices is None:
                break
            for idx in batch_indices:
                if current_idx >= self.start_idx * self.batch_size:  # 调整判断条件，确保从 start_idx 开始
                    yield idx
                current_idx += 1
        self.start_idx = 0  # set the index back to 0 so for the next epoch


class ResumableSequentialSampler(SequenceSampler):
    """Sequential sampler that can be resumed from a given start index."""

    def __init__(self, dataset, start_idx: int=None, **kwargs):
        super().__init__(dataset, **kwargs)
        # Start index, allows to resume an experiment at the index it was
        self.start_idx = start_idx if start_idx is not None else 0

    def __iter__(self):
        for i, idx in enumerate(super().__iter__()):
            if i >= self.start_idx:
                yield idx
        self.start_idx = 0  # set the index back to 0 so for the next epoch
