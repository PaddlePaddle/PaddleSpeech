"""
    Providing basic audio data preprocessing pipeline, and offering
    both instance-level and batch-level data reader interfaces.
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
from data_utils.audio import SpeechSegment
from data_utils.normalizer import FeatureNormalizer


class DataGenerator(object):
    """
    DataGenerator provides basic audio data preprocessing pipeline, and offers
    both instance-level and batch-level data reader interfaces.
    Normalized FFT are used as audio features here.

    :param vocab_filepath: Vocabulary file path for indexing tokenized
                           transcriptions.
    :type vocab_filepath: basestring
    :param normalizer_manifest_path: Manifest filepath for collecting feature
                                     normalization statistics, e.g. mean, std.
    :type normalizer_manifest_path: basestring
    :param normalizer_num_samples: Number of instances sampled for collecting
                                   feature normalization statistics.
                                   Default is 100.
    :type normalizer_num_samples: int
    :param max_duration: Audio clips with duration (in seconds) greater than
                         this will be discarded. Default is 20.0.
    :type max_duration: float
    :param min_duration: Audio clips with duration (in seconds) smaller than
                         this will be discarded. Default is 0.0.
    :type min_duration: float
    :param stride_ms: Striding size (in milliseconds) for generating frames.
                      Default is 10.0. 
    :type stride_ms: float
    :param window_ms: Window size (in milliseconds) for frames. Default is 20.0.
    :type window_ms: float
    :param max_frequency: Maximun frequency for FFT features. FFT features of
                          frequency larger than this will be discarded.
                          If set None, all features will be kept.
                          Default is None.
    :type max_frequency: float
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
                 random_seed=0):
        self._max_duration = max_duration
        self._min_duration = min_duration
        self._normalizer = FeatureNormalizer(mean_std_filepath)
        self._augmentation_pipeline = AugmentationPipeline(
            augmentation_config=augmentation_config, random_seed=random_seed)
        self._speech_featurizer = SpeechFeaturizer(
            vocab_filepath=vocab_filepath,
            stride_ms=stride_ms,
            window_ms=window_ms,
            max_freq=max_freq,
            random_seed=random_seed)
        self._rng = random.Random(random_seed)
        self._epoch = 0

    def batch_reader_creator(self,
                             manifest_path,
                             batch_size,
                             padding_to=-1,
                             flatten=False,
                             sortagrad=False,
                             batch_shuffle=False):
        """
        Batch data reader creator for audio data. Creat a callable function to
        produce batches of data.
        
        Audio features will be padded with zeros to make each instance in the
        batch to share the same audio feature shape.

        :param manifest_path: Filepath of manifest for audio clip files.
        :type manifest_path: basestring
        :param batch_size: Instance number in a batch.
        :type batch_size: int
        :param padding_to:  If set -1, the maximun column numbers in the batch
                            will be used as the target size for padding.
                            Otherwise, `padding_to` will be the target size.
                            Default is -1.
        :type padding_to: int
        :param flatten: If set True, audio data will be flatten to be a 1-dim
                        ndarray. Otherwise, 2-dim ndarray. Default is False.
        :type flatten: bool
        :param sortagrad: Sort the audio clips by duration in the first epoc
                          if set True.
        :type sortagrad: bool
        :param batch_shuffle: Shuffle the audio clips if set True. It is
                              not a thorough instance-wise shuffle, but a
                              specific batch-wise shuffle. For more details,
                              please see `_batch_shuffle` function.
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
            if len(batch) > 0:
                yield self._padding_batch(batch, padding_to, flatten)
            self._epoch += 1

        return batch_reader

    @property
    def feeding(self):
        """Returns data_reader's feeding dict."""
        return {"audio_spectrogram": 0, "transcript_text": 1}

    @property
    def vocab_size(self):
        """Returns vocabulary size."""
        return self._speech_featurizer.vocab_size

    @property
    def vocab_list(self):
        """Returns vocabulary list."""
        return self._speech_featurizer.vocab_list

    def _process_utterance(self, filename, transcript):
        speech_segment = SpeechSegment.from_file(filename, transcript)
        self._augmentation_pipeline.transform_audio(speech_segment)
        specgram, text_ids = self._speech_featurizer.featurize(speech_segment)
        specgram = self._normalizer.apply(specgram)
        return specgram, text_ids

    def _instance_reader_creator(self, manifest):
        """
        Instance reader creator for audio data. Creat a callable function to
        produce instances of data.

        Instance: a tuple of a numpy ndarray of audio spectrogram and a list of
        tokenized and indexed transcription text.

        :param manifest: Filepath of manifest for audio clip files.
        :type manifest: basestring
        :return: Data reader function.
        :rtype: callable
        """

        def reader():
            for instance in manifest:
                yield self._process_utterance(instance["audio_filepath"],
                                              instance["text"])

        return reader

    def _padding_batch(self, batch, padding_to=-1, flatten=False):
        """
        Padding audio part of features (only in the time axis -- column axis)
        with zeros, to make each instance in the batch share the same
        audio feature shape.

        If `padding_to` is set -1, the maximun column numbers in the batch will
        be used as the target size. Otherwise, `padding_to` will be the target
        size. Default is -1.

        If `flatten` is set True, audio data will be flatten to be a 1-dim
        ndarray. Default is False.
        """
        new_batch = []
        # get target shape
        max_length = max([audio.shape[1] for audio, text in batch])
        if padding_to != -1:
            if padding_to < max_length:
                raise ValueError("If padding_to is not -1, it should be greater"
                                 " or equal to the original instance length.")
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
        """
        The instances have different lengths and they cannot be
        combined into a single matrix multiplication. It usually
        sorts the training examples by length and combines only
        similarly-sized instances into minibatches, pads with
        silence when necessary so that all instances in a batch
        have the same length. This batch shuffle fuction is used
        to make similarly-sized instances into minibatches and
        make a batch-wise shuffle.

        1. Sort the audio clips by duration.
        2. Generate a random number `k`, k in [0, batch_size).
        3. Randomly remove `k` instances in order to make different mini-batches,
           then make minibatches and each minibatch size is batch_size.
        4. Shuffle the minibatches.

        :param manifest: manifest file.
        :type manifest: list
        :param batch_size: Batch size. This size is also used for generate
                           a random number for batch shuffle.
        :type batch_size: int
        :return: batch shuffled mainifest.
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
