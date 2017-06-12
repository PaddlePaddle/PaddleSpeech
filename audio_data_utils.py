"""
    Providing basic audio data preprocessing pipeline, and offering
    both instance-level and batch-level data reader interfaces.
"""
import paddle.v2 as paddle
import logging
import json
import random
import soundfile
import numpy as np
import itertools
import os

RANDOM_SEED = 0
logger = logging.getLogger(__name__)


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
                 normalizer_manifest_path,
                 normalizer_num_samples=100,
                 max_duration=20.0,
                 min_duration=0.0,
                 stride_ms=10.0,
                 window_ms=20.0,
                 max_frequency=None):
        self.__max_duration__ = max_duration
        self.__min_duration__ = min_duration
        self.__stride_ms__ = stride_ms
        self.__window_ms__ = window_ms
        self.__max_frequency__ = max_frequency
        self.__epoc__ = 0
        self.__random__ = random.Random(RANDOM_SEED)
        # load vocabulary (dictionary)
        self.__vocab_dict__, self.__vocab_list__ = \
            self.__load_vocabulary_from_file__(vocab_filepath)
        # collect normalizer statistics
        self.__mean__, self.__std__ = self.__collect_normalizer_statistics__(
            manifest_path=normalizer_manifest_path,
            num_samples=normalizer_num_samples)

    def __audio_featurize__(self, audio_filename):
        """
        Preprocess audio data, including feature extraction, normalization etc..
        """
        features = self.__audio_basic_featurize__(audio_filename)
        return self.__normalize__(features)

    def __text_featurize__(self, text):
        """
        Preprocess text data, including tokenizing and token indexing etc..
        """
        return self.__convert_text_to_char_index__(
            text=text, vocabulary=self.__vocab_dict__)

    def __audio_basic_featurize__(self, audio_filename):
        """
        Compute basic (without normalization etc.) features for audio data.
        """
        return self.__spectrogram_from_file__(
            filename=audio_filename,
            stride_ms=self.__stride_ms__,
            window_ms=self.__window_ms__,
            max_freq=self.__max_frequency__)

    def __collect_normalizer_statistics__(self, manifest_path, num_samples=100):
        """
        Compute feature normalization statistics, i.e. mean and stddev.
        """
        # read manifest
        manifest = self.__read_manifest__(
            manifest_path=manifest_path,
            max_duration=self.__max_duration__,
            min_duration=self.__min_duration__)
        # sample for statistics
        sampled_manifest = self.__random__.sample(manifest, num_samples)
        # extract spectrogram feature
        features = []
        for instance in sampled_manifest:
            spectrogram = self.__audio_basic_featurize__(
                instance["audio_filepath"])
            features.append(spectrogram)
        features = np.hstack(features)
        mean = np.mean(features, axis=1).reshape([-1, 1])
        std = np.std(features, axis=1).reshape([-1, 1])
        return mean, std

    def __normalize__(self, features, eps=1e-14):
        """
        Normalize features to be of zero mean and unit stddev.
        """
        return (features - self.__mean__) / (self.__std__ + eps)

    def __spectrogram_from_file__(self,
                                  filename,
                                  stride_ms=10.0,
                                  window_ms=20.0,
                                  max_freq=None,
                                  eps=1e-14):
        """
        Laod audio data and calculate the log of spectrogram by FFT.
        Refer to utils.py in https://github.com/baidu-research/ba-dls-deepspeech
        """
        audio, sample_rate = soundfile.read(filename)
        if audio.ndim >= 2:
            audio = np.mean(audio, 1)
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must be greater than half of "
                             "sample rate.")
        if stride_ms > window_ms:
            raise ValueError("Stride size must not be greater than "
                             "window size.")
        stride_size = int(0.001 * sample_rate * stride_ms)
        window_size = int(0.001 * sample_rate * window_ms)
        spectrogram, freqs = self.__extract_spectrogram__(
            audio,
            window_size=window_size,
            stride_size=stride_size,
            sample_rate=sample_rate)
        ind = np.where(freqs <= max_freq)[0][-1] + 1
        return np.log(spectrogram[:ind, :] + eps)

    def __extract_spectrogram__(self, samples, window_size, stride_size,
                                sample_rate):
        """
        Compute the spectrogram by FFT for a discrete real signal.
        Refer to utils.py in https://github.com/baidu-research/ba-dls-deepspeech
        """
        # extract strided windows
        truncate_size = (len(samples) - window_size) % stride_size
        samples = samples[:len(samples) - truncate_size]
        nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
        nstrides = (samples.strides[0], samples.strides[0] * stride_size)
        windows = np.lib.stride_tricks.as_strided(
            samples, shape=nshape, strides=nstrides)
        assert np.all(
            windows[:, 1] == samples[stride_size:(stride_size + window_size)])
        # window weighting, squared Fast Fourier Transform (fft), scaling
        weighting = np.hanning(window_size)[:, None]
        fft = np.fft.rfft(windows * weighting, axis=0)
        fft = np.absolute(fft)**2
        scale = np.sum(weighting**2) * sample_rate
        fft[1:-1, :] *= (2.0 / scale)
        fft[(0, -1), :] /= scale
        # prepare fft frequency list
        freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
        return fft, freqs

    def __load_vocabulary_from_file__(self, vocabulary_path):
        """
        Load vocabulary from file.
        """
        if not os.path.exists(vocabulary_path):
            raise ValueError("Vocabulary file %s not found.", vocabulary_path)
        vocab_lines = []
        with open(vocabulary_path, 'r') as file:
            vocab_lines.extend(file.readlines())
        vocab_list = [line[:-1] for line in vocab_lines]
        vocab_dict = dict(
            [(token, id) for (id, token) in enumerate(vocab_list)])
        return vocab_dict, vocab_list

    def __convert_text_to_char_index__(self, text, vocabulary):
        """
        Convert text string to a list of character index integers.
        """
        return [vocabulary[w] for w in text]

    def __read_manifest__(self, manifest_path, max_duration, min_duration):
        """
        Load and parse manifest file.
        """
        manifest = []
        for json_line in open(manifest_path):
            try:
                json_data = json.loads(json_line)
            except Exception as e:
                raise ValueError("Error reading manifest: %s" % str(e))
            if (json_data["duration"] <= max_duration and
                    json_data["duration"] >= min_duration):
                manifest.append(json_data)
        return manifest

    def __padding_batch__(self, batch, padding_to=-1, flatten=False):
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

    def __batch_shuffle__(self, manifest, batch_shuffle_size):
        """
        1. Sort the audio clips by duration.
        2. Generate a random number `k`, k in [0, batch_shuffle_size).
        3. Randomly remove `k` instances in order to make different mini-batches,
           then make minibatches and each minibatch size is batch_shuffle_size.
        4. Shuffle the minibatches.

        :param manifest: manifest file.
        :type manifest: list
        :param batch_shuffle_size: This size is uesed to generate a random number,
                                   it usually equals to batch size.
        :type batch_shuffle_size: int
        :return: batch shuffled mainifest.
        :rtype: list
        """
        manifest.sort(key=lambda x: x["duration"])
        shift_len = self.__random__.randint(0, batch_shuffle_size - 1)
        batch_manifest = zip(*[iter(manifest[shift_len:])] * batch_shuffle_size)
        self.__random__.shuffle(batch_manifest)
        batch_manifest = list(sum(batch_manifest, ()))
        res_len = len(manifest) - shift_len - len(batch_manifest)
        batch_manifest.extend(manifest[-res_len:])
        batch_manifest.extend(manifest[0:shift_len])
        return batch_manifest

    def instance_reader_creator(self, manifest):
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
            # extract spectrogram feature
            for instance in manifest:
                spectrogram = self.__audio_featurize__(
                    instance["audio_filepath"])
                transcript = self.__text_featurize__(instance["text"])
                yield (spectrogram, transcript)

        return reader

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
                              not a thorough instance-wise shuffle,
                              but a specific batch-wise shuffle.
        :type batch_shuffle: bool
        :return: Batch reader function, producing batches of data when called.
        :rtype: callable
        """

        def batch_reader():
            # read manifest
            manifest = self.__read_manifest__(
                manifest_path=manifest_path,
                max_duration=self.__max_duration__,
                min_duration=self.__min_duration__)

            # sort (by duration) or shuffle manifest
            if self.__epoc__ == 0 and sortagrad:
                manifest.sort(key=lambda x: x["duration"])
            elif batch_shuffle:
                manifest = self.__batch_shuffle__(manifest, batch_size)

            instance_reader = self.instance_reader_creator(manifest)
            batch = []
            for instance in instance_reader():
                batch.append(instance)
                if len(batch) == batch_size:
                    yield self.__padding_batch__(batch, padding_to, flatten)
                    batch = []
            if len(batch) > 0:
                yield self.__padding_batch__(batch, padding_to, flatten)
            self.__epoc__ += 1

        return batch_reader

    def vocabulary_size(self):
        """
        Get vocabulary size.

        :return: Vocabulary size.
        :rtype: int
        """
        return len(self.__vocab_list__)

    def vocabulary_dict(self):
        """
        Get vocabulary in dict.

        :return: Vocabulary in dict.
        :rtype: dict
        """
        return self.__vocab_dict__

    def vocabulary_list(self):
        """
        Get vocabulary in list.

        :return: Vocabulary in list
        :rtype: list
        """
        return self.__vocab_list__

    def data_name_feeding(self):
        """
        Get feeddings (data field name and corresponding field id).

        :return: Feeding dict.
        :rtype: dict
        """
        feeding = {
            "audio_spectrogram": 0,
            "transcript_text": 1,
        }
        return feeding
