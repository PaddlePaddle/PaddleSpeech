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
"""Contains the speech featurizer class."""
from deepspeech.frontend.featurizer.audio_featurizer import AudioFeaturizer
from deepspeech.frontend.featurizer.text_featurizer import TextFeaturizer


class SpeechFeaturizer(object):
    """Speech featurizer, for extracting features from both audio and transcript
    contents of SpeechSegment.

    Currently, for audio parts, it supports feature types of linear
    spectrogram and mfcc; for transcript parts, it only supports char-level
    tokenizing and conversion into a list of token indices. Note that the
    token indexing order follows the given vocabulary file.

    :param vocab_filepath: Filepath to load vocabulary for token indices
                           conversion.
    :type specgram_type: str
    :param specgram_type: Specgram feature type. Options: 'linear', 'mfcc'.
    :type specgram_type: str
    :param stride_ms: Striding size (in milliseconds) for generating frames.
    :type stride_ms: float
    :param window_ms: Window size (in milliseconds) for generating frames.
    :type window_ms: float
    :param max_freq: When specgram_type is 'linear', only FFT bins
                     corresponding to frequencies between [0, max_freq] are
                     returned; when specgram_type is 'mfcc', max_freq is the
                     highest band edge of mel filters.
    :types max_freq: None|float
    :param target_sample_rate: Speech are resampled (if upsampling or
                               downsampling is allowed) to this before
                               extracting spectrogram features.
    :type target_sample_rate: float
    :param use_dB_normalization: Whether to normalize the audio to a certain
                                 decibels before extracting the features.
    :type use_dB_normalization: bool
    :param target_dB: Target audio decibels for normalization.
    :type target_dB: float
    """

    def __init__(self,
                 unit_type,
                 vocab_filepath,
                 spm_model_prefix=None,
                 specgram_type='linear',
                 feat_dim=None,
                 delta_delta=False,
                 stride_ms=10.0,
                 window_ms=20.0,
                 n_fft=None,
                 max_freq=None,
                 target_sample_rate=16000,
                 use_dB_normalization=True,
                 target_dB=-20,
                 dither=1.0):
        self._audio_featurizer = AudioFeaturizer(
            specgram_type=specgram_type,
            feat_dim=feat_dim,
            delta_delta=delta_delta,
            stride_ms=stride_ms,
            window_ms=window_ms,
            n_fft=n_fft,
            max_freq=max_freq,
            target_sample_rate=target_sample_rate,
            use_dB_normalization=use_dB_normalization,
            target_dB=target_dB,
            dither=dither)
        self._text_featurizer = TextFeaturizer(unit_type, vocab_filepath,
                                               spm_model_prefix)

    def featurize(self, speech_segment, keep_transcription_text):
        """Extract features for speech segment.

        1. For audio parts, extract the audio features.
        2. For transcript parts, keep the original text or convert text string
           to a list of token indices in char-level.

        Args:
            speech_segment (SpeechSegment): Speech segment to extract features from.
            keep_transcription_text (bool): True, keep transcript text, False, token ids

        Returns:
            tuple: 1) spectrogram audio feature in 2darray, 2) list oftoken indices.
        """
        spec_feature = self._audio_featurizer.featurize(speech_segment)
        if keep_transcription_text:
            return spec_feature, speech_segment.transcript
        if speech_segment.has_token:
            text_ids = speech_segment.token_ids
        else:
            text_ids = self._text_featurizer.featurize(
                speech_segment.transcript)
        return spec_feature, text_ids

    @property
    def vocab_size(self):
        """Return the vocabulary size.
        Returns:
            int: Vocabulary size.
        """
        return self._text_featurizer.vocab_size

    @property
    def vocab_list(self):
        """Return the vocabulary in list.
        Returns:
            List[str]: 
        """
        return self._text_featurizer.vocab_list

    @property
    def vocab_dict(self):
        """Return the vocabulary in dict.
        Returns:
            Dict[str, int]: 
        """
        return self._text_featurizer.vocab_dict

    @property
    def feature_size(self):
        """Return the audio feature size.
        Returns:
            int: audio feature size.
        """
        return self._audio_featurizer.feature_size

    @property
    def stride_ms(self):
        """time length in `ms` unit per frame
        Returns:
            float: time(ms)/frame
        """
        return self._audio_featurizer.stride_ms

    @property
    def text_feature(self):
        """Return the text feature object.
        Returns:
            TextFeaturizer: object.
        """
        return self._text_featurizer
