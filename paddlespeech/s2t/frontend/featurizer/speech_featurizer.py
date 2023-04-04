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
from paddlespeech.s2t.frontend.featurizer.audio_featurizer import AudioFeaturizer
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer


class SpeechFeaturizer():
    """Speech and Text feature extraction.
    """
    def __init__(self,
                 unit_type,
                 vocab_filepath,
                 spm_model_prefix=None,
                 spectrum_type='linear',
                 feat_dim=None,
                 delta_delta=False,
                 stride_ms=10.0,
                 window_ms=20.0,
                 n_fft=None,
                 max_freq=None,
                 target_sample_rate=16000,
                 use_dB_normalization=True,
                 target_dB=-20,
                 dither=1.0,
                 maskctc=False):
        self.stride_ms = stride_ms
        self.window_ms = window_ms

        self.audio_feature = AudioFeaturizer(
            spectrum_type=spectrum_type,
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
        self.feature_size = self.audio_feature.feature_size

        self.text_feature = TextFeaturizer(unit_type=unit_type,
                                           vocab=vocab_filepath,
                                           spm_model_prefix=spm_model_prefix,
                                           maskctc=maskctc)
        self.vocab_size = self.text_feature.vocab_size

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
        spec_feature = self.audio_feature.featurize(speech_segment)

        if keep_transcription_text:
            return spec_feature, speech_segment.transcript

        if speech_segment.has_token:
            text_ids = speech_segment.token_ids
        else:
            text_ids = self.text_feature.featurize(speech_segment.transcript)
        return spec_feature, text_ids

    def text_featurize(self, text, keep_transcription_text):
        """Extract features for speech segment.

        1. For audio parts, extract the audio features.
        2. For transcript parts, keep the original text or convert text string
           to a list of token indices in char-level.

        Args:
            text (str): text.
            keep_transcription_text (bool): True, keep transcript text, False, token ids

        Returns:
            (str|List[int]): text, or list of token indices.
        """
        if keep_transcription_text:
            return text

        text_ids = self.text_feature.featurize(text)
        return text_ids
