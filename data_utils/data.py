"""Contains data generator for orgnaizing various audio data preprocessing
pipeline and offering data reader interface of PaddlePaddle requirements.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import paddle.v2 as paddle
from data_utils import utils
from data_utils.augmentor.augmentation import AugmentationPipeline
from data_utils.featurizer.speech_featurizer import SpeechFeaturizer
from data_utils.speech import SpeechSegment
from data_utils.normalizer import FeatureNormalizer


class DataGenerator(object):
    """
    DataGenerator provides basic audio data preprocessing pipeline, and offers
    data reader interfaces of PaddlePaddle requirements.

    :param vocab_filepath: Vocabulary filepath for indexing tokenized
                           transcripts.
    :type vocab_filepath: basestring
    :param mean_std_filepath: File containing the pre-computed mean and stddev.
    :type mean_std_filepath: None|basestring
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
    :param random_seed: Random seed.
    :type random_seed: int
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
                 random_seed=0):
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
            max_freq=max_freq)
        self._rng = random.Random(random_seed)
        self._epoch = 0

    def batch_reader_creator(self,
                             manifest_path,
                             batch_size,
                             min_batch_size=1,
                             padding_to=-1,
                             flatten=False,
                             sortagrad=False,
                             batch_shuffle=False):
        """
        Batch data reader creator for audio data. Return a callable generator
        function to produce batches of data.
        
        Audio features within one batch will be padded with zeros to have the
        same shape, or a user-defined shape.

        :param manifest_path: Filepath of manifest for audio files.
        :type manifest_path: basestring
        :param batch_size: Number of instances in a batch.
        :type batch_size: int
        :param min_batch_size: Any batch with batch size smaller than this will
                               be discarded. (To be deprecated in the future.)
        :type min_batch_size: int
        :param padding_to:  If set -1, the maximun shape in the batch
                            will be used as the target shape for padding.
                            Otherwise, `padding_to` will be the target shape.
        :type padding_to: int
        :param flatten: If set True, audio features will be flatten to 1darray.
        :type flatten: bool
        :param sortagrad: If set True, sort the instances by audio duration
                          in the first epoch for speed up training.
        :type sortagrad: bool
        :param batch_shuffle: If set True, instances are batch-wise shuffled.
                              For more details, please see 
                              ``_batch_shuffle.__doc__``.
                              If sortagrad is True, batch_shuffle is disabled
                              for the first epoch.
        :type batch_shuffle: bool
        :return: Batch reader function, producing batches of data when called.
        :rtype: callable
        """

        def batch_reader():
            # read manifest
            manifest = utils.read_manifest(
                manifest_path=manifest_path,
                max_duration=self._max_duration,
                min_duration=self._min_duration)
            # sort (by duration) or batch-wise shuffle the manifest
            if self._epoch == 0 and sortagrad:
                manifest.sort(key=lambda x: x["duration"])
            elif batch_shuffle:
                manifest = self._batch_shuffle(manifest, batch_size)
            # prepare batches
            instance_reader = self._instance_reader_creator(manifest)
            batch = []
            for instance in instance_reader():
                batch.append(instance)
                if len(batch) == batch_size:
                    yield self._padding_batch(batch, padding_to, flatten)
                    batch = []
            if len(batch) >= min_batch_size:
                yield self._padding_batch(batch, padding_to, flatten)
            self._epoch += 1

        return batch_reader

    @property
    def feeding(self):
        """Returns data reader's feeding dict.
        
        :return: Data feeding dict.
        :rtype: dict 
        """
        return {"audio_spectrogram": 0, "transcript_text": 1}

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

    def _process_utterance(self, filename, transcript):
        """Load, augment, featurize and normalize for speech data."""
        speech_segment = SpeechSegment.from_file(filename, transcript)
        self._augmentation_pipeline.transform_audio(speech_segment)
        specgram, text_ids = self._speech_featurizer.featurize(speech_segment)
        specgram = self._normalizer.apply(specgram)
        return specgram, text_ids

    def _instance_reader_creator(self, manifest):
        """
        Instance reader creator. Create a callable function to produce
        instances of data.

        Instance: a tuple of ndarray of audio spectrogram and a list of
        token indices for transcript.
        """

        def reader():
            for instance in manifest:
                yield self._process_utterance(instance["audio_filepath"],
                                              instance["text"])

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
        # padding
        for audio, text in batch:
            padded_audio = np.zeros([audio.shape[0], max_length])
            padded_audio[:, :audio.shape[1]] = audio
            if flatten:
                padded_audio = padded_audio.flatten()
            new_batch.append((padded_audio, text))
        return new_batch

    def _batch_shuffle(self, manifest, batch_size):
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
        :return: Batch shuffled mainifest.
        :rtype: list
        """
        manifest.sort(key=lambda x: x["duration"])
        shift_len = self._rng.randint(0, batch_size - 1)
        batch_manifest = zip(*[iter(manifest[shift_len:])] * batch_size)
        self._rng.shuffle(batch_manifest)
        batch_manifest = list(sum(batch_manifest, ()))
        res_len = len(manifest) - shift_len - len(batch_manifest)
        batch_manifest.extend(manifest[-res_len:])
        batch_manifest.extend(manifest[0:shift_len])
        return batch_manifest
