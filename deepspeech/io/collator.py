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
import numpy as np

from deepspeech.frontend.utility import IGNORE_ID
from deepspeech.io.utility import pad_sequence
from deepspeech.utils.log import Log

__all__ = ["SpeechCollator"]

logger = Log(__name__).getlog()


class SpeechCollator():
    def __init__(self, keep_transcription_text=True):
        """
        Padding audio features with zeros to make them have the same shape (or
        a user-defined shape) within one bach.

        if ``keep_transcription_text`` is False, text is token ids else is raw string.
        """
        self._keep_transcription_text = keep_transcription_text

    def __call__(self, batch):
        """batch examples

        Args:
            batch ([List]): batch is (audio, text)
                audio (np.ndarray) shape (D, T)
                text (List[int] or str): shape (U,)

        Returns:
            tuple(audio, text, audio_lens, text_lens): batched data.
                audio : (B, Tmax, D)
                audio_lens: (B)
                text : (B, Umax)
                text_lens: (B)
        """
        audios = []
        audio_lens = []
        texts = []
        text_lens = []
        utts = []
        for utt, audio, text in batch:
            #utt
            utts.append(utt)
            # audio
            audios.append(audio.T)  # [T, D]
            audio_lens.append(audio.shape[1])
            # text
            # for training, text is token ids
            # else text is string, convert to unicode ord
            tokens = []
            if self._keep_transcription_text:
                assert isinstance(text, str), (type(text), text)
                tokens = [ord(t) for t in text]
            else:
                tokens = text  # token ids
            tokens = tokens if isinstance(tokens, np.ndarray) else np.array(
                tokens, dtype=np.int64)
            texts.append(tokens)
            text_lens.append(tokens.shape[0])

        padded_audios = pad_sequence(
            audios, padding_value=0.0).astype(np.float32)  #[B, T, D]
        audio_lens = np.array(audio_lens).astype(np.int64)
        padded_texts = pad_sequence(
            texts, padding_value=IGNORE_ID).astype(np.int64)
        text_lens = np.array(text_lens).astype(np.int64)
        return utts, padded_audios, audio_lens, padded_texts, text_lens
