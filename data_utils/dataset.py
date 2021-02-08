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

import math
import random
import tarfile
import numpy as np
import paddle
from paddle.io import Dataset
from paddle.io import DataLoader
from paddle.io import BatchSampler
from paddle.io import DistributedBatchSampler
from collections import namedtuple
from functools import partial

from data_utils.utility import read_manifest
from data_utils.augmentor.augmentation import AugmentationPipeline
from data_utils.featurizer.speech_featurizer import SpeechFeaturizer
from data_utils.speech import SpeechSegment
from data_utils.normalizer import FeatureNormalizer


class DeepSpeech2Dataset(Dataset):
    def __init__(self,
                 manifest_path,
                 vocab_filepath,
                 mean_std_filepath,
                 augmentation_config='{}',
                 max_duration=float('inf'),
                 min_duration=0.0,
                 stride_ms=10.0,
                 window_ms=20.0,
                 max_freq=None,
                 specgram_type='linear',
                 use_dB_normalization=True,
                 random_seed=0,
                 keep_transcription_text=False):
        super().__init__()

        self._max_duration = max_duration
        self._min_duration = min_duration
        self._normalizer = FeatureNormalizer(mean_std_filepath)
        self._augmentation_pipeline = AugmentationPipeline(
            augmentation_config=augmentation_config, random_seed=random_seed)
        self._speech_featurizer = SpeechFeaturizer(
            vocab_filepath=vocab_filepath,
            specgram_type=specgram_type,
            stride_ms=stride_ms,
            window_ms=window_ms,
            max_freq=max_freq,
            use_dB_normalization=use_dB_normalization)
        self._rng = random.Random(random_seed)
        self._keep_transcription_text = keep_transcription_text
        # for caching tar files info
        self._local_data = namedtuple('local_data', ['tar2info', 'tar2object'])
        self._local_data.tar2info = {}
        self._local_data.tar2object = {}

        # read manifest
        self._manifest = read_manifest(
            manifest_path=manifest_path,
            max_duration=self._max_duration,
            min_duration=self._min_duration)
        self._manifest.sort(key=lambda x: x["duration"])

    @property
    def manifest(self):
        return self._manifest

    @property
    def vocab_size(self):
        """Return the vocabulary size.

        :return: Vocabulary size.
        :rtype: int
        """
        return self._speech_featurizer.vocab_size

    @property
    def vocab_list(self):
        """Return the vocabulary in list.

        :return: Vocabulary in list.
        :rtype: list
        """
        return self._speech_featurizer.vocab_list

    def _parse_tar(self, file):
        """Parse a tar file to get a tarfile object
        and a map containing tarinfoes
        """
        result = {}
        f = tarfile.open(file)
        for tarinfo in f.getmembers():
            result[tarinfo.name] = tarinfo
        return f, result

    def _subfile_from_tar(self, file):
        """Get subfile object from tar.

        It will return a subfile object from tar file
        and cached tar file info for next reading request.
        """
        tarpath, filename = file.split(':', 1)[1].split('#', 1)
        if 'tar2info' not in self._local_data.__dict__:
            self._local_data.tar2info = {}
        if 'tar2object' not in self._local_data.__dict__:
            self._local_data.tar2object = {}
        if tarpath not in self._local_data.tar2info:
            object, infoes = self._parse_tar(tarpath)
            self._local_data.tar2info[tarpath] = infoes
            self._local_data.tar2object[tarpath] = object
        return self._local_data.tar2object[tarpath].extractfile(
            self._local_data.tar2info[tarpath][filename])

    def process_utterance(self, audio_file, transcript):
        """Load, augment, featurize and normalize for speech data.

        :param audio_file: Filepath or file object of audio file.
        :type audio_file: str | file
        :param transcript: Transcription text.
        :type transcript: str
        :return: Tuple of audio feature tensor and data of transcription part,
                 where transcription part could be token ids or text.
        :rtype: tuple of (2darray, list)
        """
        if isinstance(audio_file, str) and audio_file.startswith('tar:'):
            speech_segment = SpeechSegment.from_file(
                self._subfile_from_tar(audio_file), transcript)
        else:
            speech_segment = SpeechSegment.from_file(audio_file, transcript)
        self._augmentation_pipeline.transform_audio(speech_segment)
        specgram, transcript_part = self._speech_featurizer.featurize(
            speech_segment, self._keep_transcription_text)
        specgram = self._normalizer.apply(specgram)
        return specgram, transcript_part

    def _instance_reader_creator(self, manifest):
        """
        Instance reader creator. Create a callable function to produce
        instances of data.

        Instance: a tuple of ndarray of audio spectrogram and a list of
        token indices for transcript.
        """

        def reader():
            for instance in manifest:
                inst = self.process_utterance(instance["audio_filepath"],
                                              instance["text"])
                yield inst

        return reader

    def __len__(self):
        return len(self._manifest)

    def __getitem__(self, idx):
        instance = self._manifest[idx]
        return self.process_utterance(instance["audio_filepath"],
                                      instance["text"])


class DeepSpeech2DistributedBatchSampler(DistributedBatchSampler):
    def __init__(self,
                 dataset,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 drop_last=False,
                 sortagrad=False,
                 shuffle_method="batch_shuffle"):
        super().__init__(dataset, batch_size, num_replicas, rank, shuffle,
                         drop_last)
        self._sortagrad = sortagrad
        self._shuffle_method = shuffle_method

    def _batch_shuffle(self, manifest, batch_size, clipped=False):
        """Put similarly-sized instances into minibatches for better efficiency
        and make a batch-wise shuffle.

        1. Sort the audio clips by duration.
        2. Generate a random number `k`, k in [0, batch_size).
        3. Randomly shift `k` instances in order to create different batches
           for different epochs. Create minibatches.
        4. Shuffle the minibatches.

        :param manifest: Manifest contents. List of dict.
        :type manifest: list
        :param batch_size: Batch size. This size is also used for generate
                           a random number for batch shuffle.
        :type batch_size: int
        :param clipped: Whether to clip the heading (small shift) and trailing
                        (incomplete batch) instances.
        :type clipped: bool
        :return: Batch shuffled mainifest.
        :rtype: list
        """
        rng = np.random.RandomState(self.epoch)
        manifest.sort(key=lambda x: x["duration"])
        shift_len = rng.randint(0, batch_size - 1)
        batch_manifest = list(zip(*[iter(manifest[shift_len:])] * batch_size))
        rng.shuffle(batch_manifest)
        batch_manifest = [item for batch in batch_manifest for item in batch]
        if not clipped:
            res_len = len(manifest) - shift_len - len(batch_manifest)
            batch_manifest.extend(manifest[-res_len:])
            batch_manifest.extend(manifest[0:shift_len])
        return batch_manifest

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # sort (by duration) or batch-wise shuffle the manifest
        if self.shuffle:
            if self.epoch == 0 and self.sortagrad:
                pass
            else:
                if self._shuffle_method == "batch_shuffle":
                    indices = self._batch_shuffle(
                        indices, self.batch_size, clipped=False)
                elif self._shuffle_method == "instance_shuffle":
                    np.random.RandomState(self.epoch).shuffle(indices)
                else:
                    raise ValueError("Unknown shuffle method %s." %
                                     self._shuffle_method)
        assert len(indices) == self.total_size
        self.epoch += 1

        # subsample
        def _get_indices_by_batch_size(indices):
            subsampled_indices = []
            last_batch_size = self.total_size % (self.batch_size * self.nranks)
            assert last_batch_size % self.nranks == 0
            last_local_batch_size = last_batch_size // self.nranks

            for i in range(self.local_rank * self.batch_size,
                           len(indices) - last_batch_size,
                           self.batch_size * self.nranks):
                subsampled_indices.extend(indices[i:i + self.batch_size])

            indices = indices[len(indices) - last_batch_size:]
            subsampled_indices.extend(
                indices[self.local_rank * last_local_batch_size:(
                    self.local_rank + 1) * last_local_batch_size])
            return subsampled_indices

        if self.nranks > 1:
            indices = _get_indices_by_batch_size(indices)

        assert len(indices) == self.num_samples
        _sample_iter = iter(indices)

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        num_samples = self.num_samples
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size


class DeepSpeech2BatchSampler(BatchSampler):
    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle=False,
                 drop_last=False,
                 sortagrad=False,
                 shuffle_method="batch_shuffle",
                 num_replicas=1,
                 rank=0):
        self.dataset = dataset

        assert isinstance(batch_size, int) and batch_size > 0, \
                "batch_size should be a positive integer"
        self.batch_size = batch_size
        assert isinstance(shuffle, bool), \
                "shuffle should be a boolean value"
        self.shuffle = shuffle
        assert isinstance(drop_last, bool), \
                "drop_last should be a boolean number"

        if num_replicas is not None:
            assert isinstance(num_replicas, int) and num_replicas > 0, \
                    "num_replicas should be a positive integer"
            self.nranks = num_replicas
        else:
            self.nranks = num_replicas

        if rank is not None:
            assert isinstance(rank, int) and rank >= 0, \
                    "rank should be a non-negative integer"
            self.local_rank = rank
        else:
            self.local_rank = rank

        self.drop_last = drop_last
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.nranks))
        self.total_size = self.num_samples * self.nranks
        self._sortagrad = sortagrad
        self._shuffle_method = shuffle_method

    def _batch_shuffle(self, manifest, batch_size, clipped=False):
        """Put similarly-sized instances into minibatches for better efficiency
        and make a batch-wise shuffle.

        1. Sort the audio clips by duration.
        2. Generate a random number `k`, k in [0, batch_size).
        3. Randomly shift `k` instances in order to create different batches
           for different epochs. Create minibatches.
        4. Shuffle the minibatches.

        :param manifest: Manifest contents. List of dict.
        :type manifest: list
        :param batch_size: Batch size. This size is also used for generate
                           a random number for batch shuffle.
        :type batch_size: int
        :param clipped: Whether to clip the heading (small shift) and trailing
                        (incomplete batch) instances.
        :type clipped: bool
        :return: Batch shuffled mainifest.
        :rtype: list
        """
        rng = np.random.RandomState(self.epoch)
        manifest.sort(key=lambda x: x["duration"])
        shift_len = rng.randint(0, batch_size - 1)
        batch_manifest = list(zip(*[iter(manifest[shift_len:])] * batch_size))
        rng.shuffle(batch_manifest)
        batch_manifest = [item for batch in batch_manifest for item in batch]
        if not clipped:
            res_len = len(manifest) - shift_len - len(batch_manifest)
            batch_manifest.extend(manifest[-res_len:])
            batch_manifest.extend(manifest[0:shift_len])
        return batch_manifest

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # sort (by duration) or batch-wise shuffle the manifest
        if self.shuffle:
            if self.epoch == 0 and self.sortagrad:
                pass
            else:
                if self._shuffle_method == "batch_shuffle":
                    indices = self._batch_shuffle(
                        indices, self.batch_size, clipped=False)
                elif self._shuffle_method == "instance_shuffle":
                    np.random.RandomState(self.epoch).shuffle(indices)
                else:
                    raise ValueError("Unknown shuffle method %s." %
                                     self._shuffle_method)
        assert len(indices) == self.total_size
        self.epoch += 1

        # subsample
        def _get_indices_by_batch_size(indices):
            subsampled_indices = []
            last_batch_size = self.total_size % (self.batch_size * self.nranks)
            assert last_batch_size % self.nranks == 0
            last_local_batch_size = last_batch_size // self.nranks

            for i in range(self.local_rank * self.batch_size,
                           len(indices) - last_batch_size,
                           self.batch_size * self.nranks):
                subsampled_indices.extend(indices[i:i + self.batch_size])

            indices = indices[len(indices) - last_batch_size:]
            subsampled_indices.extend(
                indices[self.local_rank * last_local_batch_size:(
                    self.local_rank + 1) * last_local_batch_size])
            return subsampled_indices

        if self.nranks > 1:
            indices = _get_indices_by_batch_size(indices)

        assert len(indices) == self.num_samples
        _sample_iter = iter(indices)

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        num_samples = self.num_samples
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size

    def set_epoch(self, epoch):
        """
        Sets the epoch number. When :attr:`shuffle=True`, this number is used
        as seeds of random numbers. By default, users may not set this, all
        replicas (workers) use a different random ordering for each epoch.
        If set same number at each epoch, this sampler will yield the same
        ordering at all epoches.
        Arguments:
            epoch (int): Epoch number.
        Examples:
            .. code-block:: python
    
                import numpy as np
    
                from paddle.io import Dataset, DistributedBatchSampler
    
                # init with dataset
                class RandomDataset(Dataset):
                    def __init__(self, num_samples):
                        self.num_samples = num_samples
                
                    def __getitem__(self, idx):
                        image = np.random.random([784]).astype('float32')
                        label = np.random.randint(0, 9, (1, )).astype('int64')
                        return image, label
                    
                    def __len__(self):
                        return self.num_samples
      
                dataset = RandomDataset(100)
                sampler = DistributedBatchSampler(dataset, batch_size=64)
    
                for epoch in range(10):
                    sampler.set_epoch(epoch)
        """
        self.epoch = epoch


def create_dataloader(manifest_path,
                      vocab_filepath,
                      mean_std_filepath,
                      augmentation_config='{}',
                      max_duration=float('inf'),
                      min_duration=0.0,
                      stride_ms=10.0,
                      window_ms=20.0,
                      max_freq=None,
                      specgram_type='linear',
                      use_dB_normalization=True,
                      random_seed=0,
                      keep_transcription_text=False,
                      is_training=False,
                      batch_size=1,
                      num_workers=0,
                      sortagrad=False,
                      shuffle_method=None,
                      dist=False):

    dataset = DeepSpeech2Dataset(
        manifest_path,
        vocab_filepath,
        mean_std_filepath,
        augmentation_config=augmentation_config,
        max_duration=max_duration,
        min_duration=min_duration,
        stride_ms=stride_ms,
        window_ms=window_ms,
        max_freq=max_freq,
        specgram_type=specgram_type,
        use_dB_normalization=use_dB_normalization,
        random_seed=random_seed,
        keep_transcription_text=keep_transcription_text)

    if dist:
        batch_sampler = DeepSpeech2DistributedBatchSampler(
            dataset,
            batch_size,
            num_replicas=None,
            rank=None,
            shuffle=is_training,
            drop_last=is_training,
            sortagrad=is_training,
            shuffle_method=shuffle_method)
    else:
        batch_sampler = DeepSpeech2BatchSampler(
            dataset,
            shuffle=is_training,
            batch_size=batch_size,
            drop_last=is_training,
            sortagrad=is_training,
            shuffle_method=shuffle_method)

    def padding_batch(batch, padding_to=-1, flatten=False, is_training=True):
        """
        Padding audio features with zeros to make them have the same shape (or
        a user-defined shape) within one bach.

        If ``padding_to`` is -1, the maximun shape in the batch will be used
        as the target shape for padding. Otherwise, `padding_to` will be the
        target shape (only refers to the second axis).

        If `flatten` is True, features will be flatten to 1darray.
        """
        new_batch = []
        # get target shape
        max_length = max([audio.shape[1] for audio, text in batch])
        if padding_to != -1:
            if padding_to < max_length:
                raise ValueError("If padding_to is not -1, it should be larger "
                                 "than any instance's shape in the batch")
            max_length = padding_to
        max_text_length = max([len(text) for audio, text in batch])
        # padding
        padded_audios = []
        audio_lens = []
        texts, text_lens = [], []
        for audio, text in batch:
            padded_audio = np.zeros([audio.shape[0], max_length])
            padded_audio[:, :audio.shape[1]] = audio
            if flatten:
                padded_audio = padded_audio.flatten()
            padded_audios.append(padded_audio)
            audio_lens.append(audio.shape[1])
            padded_text = np.zeros([max_text_length])
            padded_text[:len(text)] = text
            texts.append(padded_text)
            text_lens.append(len(text))

        padded_audios = np.array(padded_audios).astype('float32')
        audio_lens = np.array(audio_lens).astype('int64')
        texts = np.array(texts).astype('int32')
        text_lens = np.array(text_lens).astype('int64')
        return padded_audios, texts, audio_lens, text_lens

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=partial(padding_batch, is_training=is_training),
        num_workers=num_workers, )
    return loader
