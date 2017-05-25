"""
   Audio data preprocessing tools and reader creators.
"""
import paddle.v2 as paddle
import logging
import json
import random
import soundfile
import numpy as np
import os

# TODO: add z-score normalization.

ENGLISH_CHAR_VOCAB_FILEPATH = "eng_vocab.txt"

logger = logging.getLogger(__name__)


def spectrogram_from_file(filename,
                          stride_ms=10,
                          window_ms=20,
                          max_freq=None,
                          eps=1e-14):
    """
    Calculate the log of linear spectrogram from FFT energy
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
        raise ValueError("Stride size must not be greater than window size.")
    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)
    spectrogram, freqs = extract_spectrogram(
        audio,
        window_size=window_size,
        stride_size=stride_size,
        sample_rate=sample_rate)
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    return np.log(spectrogram[:ind, :] + eps)


def extract_spectrogram(samples, window_size, stride_size, sample_rate):
    """
    Compute the spectrogram for a real discrete signal.
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
    # window weighting, compute squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)**2
    scale = np.sum(weighting**2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    # prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
    return fft, freqs


def vocabulary_from_file(vocabulary_path):
    """
    Load vocabulary from file.
    """
    if os.path.exists(vocabulary_path):
        vocab_lines = []
        with open(vocabulary_path, 'r') as file:
            vocab_lines.extend(file.readlines())
        vocab_list = [line[:-1] for line in vocab_lines]
        vocab_dict = dict(
            [(token, id) for (id, token) in enumerate(vocab_list)])
        return vocab_dict, vocab_list
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def get_vocabulary_size():
    """
    Get vocabulary size.
    """
    vocab_dict, _ = vocabulary_from_file(ENGLISH_CHAR_VOCAB_FILEPATH)
    return len(vocab_dict)


def get_vocabulary():
    """
    Get vocabulary.
    """
    return vocabulary_from_file(ENGLISH_CHAR_VOCAB_FILEPATH)


def parse_transcript(text, vocabulary):
    """
    Convert the transcript text string to list of token index integers.
    """
    return [vocabulary[w] for w in text]


def reader_creator(manifest_path,
                   sort_by_duration=True,
                   shuffle=False,
                   max_duration=10.0,
                   min_duration=0.0):
    """
    Audio data reader creator.

    Instance: a tuple of a numpy ndarray of audio spectrogram and a list of
    tokenized transcription text.

    :param manifest_path: Filepath for Manifest of audio clip files.
    :type manifest_path: basestring
    :param sort_by_duration: Sort the audio clips by duration if set True.
                             For SortaGrad.
    :type sort_by_duration: bool
    :param shuffle: Shuffle the audio clips if set True.
    :type shuffle: bool
    :param max_duration: Audio clips with duration (in seconds) greater than
                         this will be discarded.
    :type max_duration: float
    :param min_duration: Audio clips with duration (in seconds) smaller than
                         this will be discarded.
    :type min_duration: float
    :return: Data reader function.
    :rtype: callable
    """
    if sort_by_duration and shuffle:
        sort_by_duration = False
        logger.warn("When shuffle set to true, "
                    "sort_by_duration is forced to set False.")
    vocab_dict, _ = vocabulary_from_file(ENGLISH_CHAR_VOCAB_FILEPATH)

    def reader():
        # read manifest
        manifest_data = []
        for json_line in open(manifest_path):
            try:
                json_data = json.loads(json_line)
            except Exception as e:
                raise ValueError("Error reading manifest: %s" % str(e))
            if (json_data["duration"] <= max_duration and
                    json_data["duration"] >= min_duration):
                manifest_data.append(json_data)
        # sort (by duration) or shuffle manifest
        if sort_by_duration:
            manifest_data.sort(key=lambda x: x["duration"])
        if shuffle:
            random.shuffle(manifest_data)
        # extract spectrogram feature
        for instance in manifest_data:
            spectrogram = spectrogram_from_file(instance["audio_filepath"])
            text = parse_transcript(instance["text"], vocab_dict)
            yield (spectrogram, text)

    return reader


def padding_batch_reader(batch_reader, padding=[-1, -1], flatten=True):
    """
    Padding for batches. Return a batch reader.

    Each instance in a batch will be padded to be of a same target shape.
    The target shape is the largest shape among all the batch instances and
    'padding' argument. Therefore, if padding is set [-1, -1], instance will be
    padded to have the same shape just within each batch and the shape will
    be different across batches; if padding is set
    [VERY_LARGE_NUM, VERY_LARGE_NUM], instances in all batches will be padded to
    have the same shape of [VERY_LARGE_NUM, VERY_LARGE_NUM].

    :param batch_reader: Input batch reader.
    :type batch_reader: callable
    :param padding: Padding pattern. Details please refer to the above.
    :type padding: list
    :param flatten: Flatten the tensor to be one dimension.
    :type flatten: bool
    :return: Batch reader function.
    :rtype: callable
    """

    def padding_batch(batch):
        new_batch = []
        # get target shape within batch
        nshape_list = [padding]
        for audio, text in batch:
            nshape_list.append(audio.shape)
        target_shape = np.array(nshape_list).max(axis=0)
        # padding
        for audio, text in batch:
            pad_shape = target_shape - audio.shape
            assert np.all(pad_shape >= 0)
            padded_audio = np.pad(
                audio, [(0, pad_shape[0]), (0, pad_shape[1])], mode="constant")
            if flatten:
                padded_audio = padded_audio.flatten()
            new_batch.append((padded_audio, text))
        return new_batch

    def new_batch_reader():
        for batch in batch_reader():
            yield padding_batch(batch)

    return new_batch_reader
