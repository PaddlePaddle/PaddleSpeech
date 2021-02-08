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
"""Contains data generator for orgnaizing various audio data preprocessing
pipeline and offering data reader interface of PaddlePaddle requirements.
"""

import random
import tarfile
import multiprocessing
import numpy as np
import paddle.fluid as fluid
from threading import local
from data_utils.utility import read_manifest
from data_utils.augmentor.augmentation import AugmentationPipeline
from data_utils.featurizer.speech_featurizer import SpeechFeaturizer
from data_utils.speech import SpeechSegment
from data_utils.normalizer import FeatureNormalizer

__all__ = ['DataGenerator']


class DataGenerator():
    """
    DataGenerator provides basic audio data preprocessing pipeline, and offers
    data reader interfaces of PaddlePaddle requirements.

    :param vocab_filepath: Vocabulary filepath for indexing tokenized
                           transcripts.
    :type vocab_filepath: str
    :param mean_std_filepath: File containing the pre-computed mean and stddev.
    :type mean_std_filepath: None|str
    :param augmentation_config: Augmentation configuration in json string.
                                Details see AugmentationPipeline.__doc__.
    :type augmentation_config: str
    :param max_duration: Audio with duration (in seconds) greater than
                         this will be discarded.
    :type max_duration: float
    :param min_duration: Audio with duration (in seconds) smaller than
                         this will be discarded.
    :type min_duration: float
    :param stride_ms: Striding size (in milliseconds) for generating frames.
    :type stride_ms: float
    :param window_ms: Window size (in milliseconds) for generating frames.
    :type window_ms: float
    :param max_freq: Used when specgram_type is 'linear', only FFT bins
                     corresponding to frequencies between [0, max_freq] are
                     returned.
    :types max_freq: None|float
    :param specgram_type: Specgram feature type. Options: 'linear'.
    :type specgram_type: str
    :param use_dB_normalization: Whether to normalize the audio to -20 dB
                                before extracting the features.
    :type use_dB_normalization: bool
    :param random_seed: Random seed.
    :type random_seed: int
    :param keep_transcription_text: If set to True, transcription text will
                                    be passed forward directly without
                                    converting to index sequence.
    :type keep_transcription_text: bool
    :param place: The place to run the program.
    :type place: CPUPlace or CUDAPlace
    :param is_training: If set to True, generate text data for training, 
                        otherwise,  generate text data for infer.
    :type is_training: bool 
    """

    def __init__(self,
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
                 place=fluid.CPUPlace(),
                 is_training=True):
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
        self._epoch = 0
        self._is_training = is_training
        # for caching tar files info
        self._local_data = local()
        self._local_data.tar2info = {}
        self._local_data.tar2object = {}
        self._place = place

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

    def batch_reader_creator(self,
                             manifest_path,
                             batch_size,
                             padding_to=-1,
                             flatten=False,
                             sortagrad=False,
                             shuffle_method="batch_shuffle"):
        """
        Batch data reader creator for audio data. Return a callable generator
        function to produce batches of data.

        Audio features within one batch will be padded with zeros to have the
        same shape, or a user-defined shape.

        :param manifest_path: Filepath of manifest for audio files.
        :type manifest_path: str
        :param batch_size: Number of instances in a batch.
        :type batch_size: int
        :param padding_to:  If set -1, the maximun shape in the batch
                            will be used as the target shape for padding.
                            Otherwise, `padding_to` will be the target shape.
        :type padding_to: int
        :param flatten: If set True, audio features will be flatten to 1darray.
        :type flatten: bool
        :param sortagrad: If set True, sort the instances by audio duration
                          in the first epoch for speed up training.
        :type sortagrad: bool
        :param shuffle_method: Shuffle method. Options:
                                '' or None: no shuffle.
                                'instance_shuffle': instance-wise shuffle.
                                'batch_shuffle': similarly-sized instances are
                                                 put into batches, and then
                                                 batch-wise shuffle the batches.
                                                 For more details, please see
                                                 ``_batch_shuffle.__doc__``.
                                'batch_shuffle_clipped': 'batch_shuffle' with
                                                         head shift and tail
                                                         clipping. For more
                                                         details, please see
                                                         ``_batch_shuffle``.
                              If sortagrad is True, shuffle is disabled
                              for the first epoch.
        :type shuffle_method: None|str
        :return: Batch reader function, producing batches of data when called.
        :rtype: callable
        """

        def batch_reader():
            # read manifest
            manifest = read_manifest(
                manifest_path=manifest_path,
                max_duration=self._max_duration,
                min_duration=self._min_duration)

            # sort (by duration) or batch-wise shuffle the manifest
            if self._epoch == 0 and sortagrad:
                manifest.sort(key=lambda x: x["duration"])

            else:
                if shuffle_method == "batch_shuffle":
                    manifest = self._batch_shuffle(
                        manifest, batch_size, clipped=False)
                elif shuffle_method == "batch_shuffle_clipped":
                    manifest = self._batch_shuffle(
                        manifest, batch_size, clipped=True)
                elif shuffle_method == "instance_shuffle":
                    self._rng.shuffle(manifest)
                elif shuffle_method == None:
                    pass
                else:
                    raise ValueError("Unknown shuffle method %s." %
                                     shuffle_method)
            # prepare batches
            batch = []
            instance_reader = self._instance_reader_creator(manifest)

            for instance in instance_reader():
                batch.append(instance)
                if len(batch) == batch_size:
                    yield self._padding_batch(batch, padding_to, flatten)
                    batch = []
            if len(batch) >= 1:
                yield self._padding_batch(batch, padding_to, flatten)
            self._epoch += 1

        return batch_reader

    @property
    def feeding(self):
        """Returns data reader's feeding dict.

        :return: Data feeding dict.
        :rtype: dict
        """
        feeding_dict = {"audio_spectrogram": 0, "transcript_text": 1}
        return feeding_dict

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

    def _padding_batch(self, batch, padding_to=-1, flatten=False):
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
            if self._is_training:
                padded_text = np.zeros([max_text_length])
                padded_text[:len(text)] = text
                texts.append(padded_text)
            else:
                texts.append(text)
            text_lens.append(len(text))

        padded_audios = np.array(padded_audios).astype('float32')
        audio_lens = np.array(audio_lens).astype('int64')
        if self._is_training:
            texts = np.array(texts).astype('int32')
            text_lens = np.array(text_lens).astype('int64')
        return padded_audios, texts, audio_lens, text_lens

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
        manifest.sort(key=lambda x: x["duration"])
        shift_len = self._rng.randint(0, batch_size - 1)
        batch_manifest = list(zip(*[iter(manifest[shift_len:])] * batch_size))
        self._rng.shuffle(batch_manifest)
        batch_manifest = [item for batch in batch_manifest for item in batch]
        if not clipped:
            res_len = len(manifest) - shift_len - len(batch_manifest)
            batch_manifest.extend(manifest[-res_len:])
            batch_manifest.extend(manifest[0:shift_len])
        return batch_manifest
