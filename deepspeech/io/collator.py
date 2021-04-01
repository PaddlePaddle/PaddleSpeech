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

from deepspeech.io.utility import pad_sequence

logger = logging.getLogger(__name__)

__all__ = ["SpeechCollator"]


class SpeechCollator():
    def __init__(self, is_training=True):
        """
        Padding audio features with zeros to make them have the same shape (or
        a user-defined shape) within one bach.

        If ``padding_to`` is -1, the maximun shape in the batch will be used
        as the target shape for padding. Otherwise, `padding_to` will be the
        target shape (only refers to the second axis).

        if ``is_training`` is True, text is token ids else is raw string.
        """
        self._is_training = is_training

    def __call__(self, batch):
        """batch examples

        Args:
            batch ([List]): batch is (audio, text)
                audio (np.ndarray) shape (D, T)
                text (List[int] or str): shape (U,)

        Returns:
            tuple(audio, text, audio_lens, text_lens): batched data.
                audio : (B, Tmax, D)
                text : (B, Umax)
                audio_lens: (B)
                text_lens: (B)
        """
        audios = []
        audio_lens = []
        texts = []
        text_lens = []
        for audio, text in batch:
            # audio
            audios.append(audio.T)  # [T, D]
            audio_lens.append(audio.shape[1])
            # text
            # for training, text is token ids
            # else text is string, convert to unicode ord
            tokens = []
            if self._is_training:
                tokens = text  # token ids
            else:
                assert isinstance(text, str)
                tokens = [ord(t) for t in text]
            tokens = tokens if isinstance(tokens, np.ndarray) else np.array(
                tokens, dtype=np.int64)
            texts.append(tokens)
            text_lens.append(len(text))

        padded_audios = pad_sequence(
            audios, padding_value=0.0).astype(np.float32)  #[B, T, D]
        padded_texts = pad_sequence(texts, padding_value=-1).astype(np.int32)
        audio_lens = np.array(audio_lens).astype(np.int64)
        text_lens = np.array(text_lens).astype(np.int64)
        return padded_audios, padded_texts, audio_lens, text_lens
