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
"""Contains the speech segment class."""
import numpy as np

from paddlespeech.s2t.frontend.audio import AudioSegment


class SpeechSegment(AudioSegment):
    """Speech Segment with Text

    Args:
        AudioSegment (AudioSegment): Audio Segment
    """

    def __init__(self,
                 samples,
                 sample_rate,
                 transcript,
                 tokens=None,
                 token_ids=None):
        """Speech segment abstraction, a subclass of AudioSegment,
            with an additional transcript.

        Args:
            samples (ndarray.float32): Audio samples [num_samples x num_channels].
            sample_rate (int): Audio sample rate.
            transcript (str): Transcript text for the speech.
            tokens (List[str], optinal): Transcript tokens for the speech.
            token_ids (List[int], optional): Transcript token ids for the speech.
        """
        AudioSegment.__init__(self, samples, sample_rate)
        self._transcript = transcript
        # must init `tokens` with `token_ids` at the same time
        self._tokens = tokens
        self._token_ids = token_ids

    def __eq__(self, other):
        """Return whether two objects are equal.

        Returns:
            bool: True, when equal to other
        """
        if not AudioSegment.__eq__(self, other):
            return False
        if self._transcript != other._transcript:
            return False
        if self.has_token and other.has_token:
            if self._tokens != other._tokens:
                return False
            if self._token_ids != other._token_ids:
                return False
        return True

    def __ne__(self, other):
        """Return whether two objects are unequal."""
        return not self.__eq__(other)

    @classmethod
    def from_file(cls,
                  filepath,
                  transcript,
                  tokens=None,
                  token_ids=None,
                  infos=None):
        """Create speech segment from audio file and corresponding transcript.

        Args:
            filepath (str|file): Filepath or file object to audio file.
            transcript (str): Transcript text for the speech.
            tokens (List[str], optional): text tokens. Defaults to None.
            token_ids (List[int], optional): text token ids. Defaults to None.
            infos (TarLocalData, optional): tar2obj and tar2infos. Defaults to None.

        Returns:
            SpeechSegment: Speech segment instance.
        """
        audio = AudioSegment.from_file(filepath, infos)
        return cls(audio.samples, audio.sample_rate, transcript, tokens,
                   token_ids)

    @classmethod
    def from_bytes(cls, bytes, transcript, tokens=None, token_ids=None):
        """Create speech segment from a byte string and corresponding

        Args:
            filepath (str|file): Filepath or file object to audio file.
            transcript (str): Transcript text for the speech.
            tokens (List[str], optional): text tokens. Defaults to None.
            token_ids (List[int], optional): text token ids. Defaults to None.

        Returns:
            SpeechSegment: Speech segment instance.
        """
        audio = AudioSegment.from_bytes(bytes)
        return cls(audio.samples, audio.sample_rate, transcript, tokens,
                   token_ids)

    @classmethod
    def concatenate(cls, *segments):
        """Concatenate an arbitrary number of speech segments together, both
        audio and transcript will be concatenated.

        :param *segments: Input speech segments to be concatenated.
        :type *segments: tuple of SpeechSegment
        :return: Speech segment instance.
        :rtype: SpeechSegment
        :raises ValueError: If the number of segments is zero, or if the 
                            sample_rate of any two segments does not match.
        :raises TypeError: If any segment is not SpeechSegment instance.
        """
        if len(segments) == 0:
            raise ValueError("No speech segments are given to concatenate.")
        sample_rate = segments[0]._sample_rate
        transcripts = ""
        tokens = []
        token_ids = []
        for seg in segments:
            if sample_rate != seg._sample_rate:
                raise ValueError("Can't concatenate segments with "
                                 "different sample rates")
            if type(seg) is not cls:
                raise TypeError("Only speech segments of the same type "
                                "instance can be concatenated.")
            transcripts += seg._transcript
            if self.has_token:
                tokens += seg._tokens
                token_ids += seg._token_ids
        samples = np.concatenate([seg.samples for seg in segments])
        return cls(samples, sample_rate, transcripts, tokens, token_ids)

    @classmethod
    def slice_from_file(cls,
                        filepath,
                        transcript,
                        tokens=None,
                        token_ids=None,
                        start=None,
                        end=None):
        """Loads a small section of an speech without having to load
        the entire file into the memory which can be incredibly wasteful.

        :param filepath: Filepath or file object to audio file.
        :type filepath: str|file
        :param start: Start time in seconds. If start is negative, it wraps
                      around from the end. If not provided, this function 
                      reads from the very beginning.
        :type start: float
        :param end: End time in seconds. If end is negative, it wraps around
                    from the end. If not provided, the default behvaior is
                    to read to the end of the file.
        :type end: float
        :param transcript: Transcript text for the speech. if not provided, 
                           the defaults is an empty string.
        :type transript: str
        :return: SpeechSegment instance of the specified slice of the input
                 speech file.
        :rtype: SpeechSegment
        """
        audio = AudioSegment.slice_from_file(filepath, start, end)
        return cls(audio.samples, audio.sample_rate, transcript, tokens,
                   token_ids)

    @classmethod
    def make_silence(cls, duration, sample_rate):
        """Creates a silent speech segment of the given duration and
        sample rate, transcript will be an empty string.

        Args:
            duration (float): Length of silence in seconds.
            sample_rate (float): Sample rate.

        Returns:
            SpeechSegment: Silence of the given duration.
        """
        audio = AudioSegment.make_silence(duration, sample_rate)
        return cls(audio.samples, audio.sample_rate, "")

    @property
    def has_token(self):
        if self._tokens and self._token_ids:
            return True
        return False

    @property
    def transcript(self):
        """Return the transcript text.

        Returns:
            str: Transcript text for the speech.
        """

        return self._transcript

    @property
    def tokens(self):
        """Return the transcript text tokens.

        Returns:
            List[str]: text tokens.
        """
        return self._tokens

    @property
    def token_ids(self):
        """Return the transcript text token ids.

        Returns:
            List[int]: text token ids.
        """
        return self._token_ids
