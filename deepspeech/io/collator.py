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

import logging
import numpy as np
from collections import namedtuple

logger = logging.getLogger(__name__)

__all__ = [
    "SpeechCollator",
]


class SpeechCollator():
    def __init__(self, padding_to=-1, is_training=True):
        """
        Padding audio features with zeros to make them have the same shape (or
        a user-defined shape) within one bach.

        If ``padding_to`` is -1, the maximun shape in the batch will be used
        as the target shape for padding. Otherwise, `padding_to` will be the
        target shape (only refers to the second axis).
        """
        self._padding_to = padding_to
        self._is_training = is_training

    def __call__(self, batch):
        new_batch = []
        # get target shape
        max_length = max([audio.shape[1] for audio, _ in batch])
        if self._padding_to != -1:
            if self._padding_to < max_length:
                raise ValueError("If padding_to is not -1, it should be larger "
                                 "than any instance's shape in the batch")
            max_length = self._padding_to
        max_text_length = max([len(text) for _, text in batch])
        # padding
        padded_audios = []
        audio_lens = []
        texts, text_lens = [], []
        for audio, text in batch:
            # audio
            padded_audio = np.zeros([audio.shape[0], max_length])
            padded_audio[:, :audio.shape[1]] = audio
            padded_audios.append(padded_audio)
            audio_lens.append(audio.shape[1])
            # text
            padded_text = np.zeros([max_text_length])
            if self._is_training:
                padded_text[:len(text)] = text  # token ids
            else:
                padded_text[:len(text)] = [ord(t)
                                           for t in text]  # string, unicode ord
            texts.append(padded_text)
            text_lens.append(len(text))

        padded_audios = np.array(padded_audios).astype('float32')
        audio_lens = np.array(audio_lens).astype('int64')
        texts = np.array(texts).astype('int32')
        text_lens = np.array(text_lens).astype('int64')
        return padded_audios, texts, audio_lens, text_lens
