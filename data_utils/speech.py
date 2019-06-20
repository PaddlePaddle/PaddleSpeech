"""Contains the speech segment class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from data_utils.audio import AudioSegment


class SpeechSegment(AudioSegment):
    """Speech segment abstraction, a subclass of AudioSegment,
    with an additional transcript.

    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :param transcript: Transcript text for the speech.
    :type transript: basestring
    :raises TypeError: If the sample data type is not float or int.
    """

    def __init__(self, samples, sample_rate, transcript):
        AudioSegment.__init__(self, samples, sample_rate)
        self._transcript = transcript

    def __eq__(self, other):
        """Return whether two objects are equal.
        """
        if not AudioSegment.__eq__(self, other):
            return False
        if self._transcript != other._transcript:
            return False
        return True

    def __ne__(self, other):
        """Return whether two objects are unequal."""
        return not self.__eq__(other)

    @classmethod
    def from_file(cls, filepath, transcript):
        """Create speech segment from audio file and corresponding transcript.
        
        :param filepath: Filepath or file object to audio file.
        :type filepath: basestring|file
        :param transcript: Transcript text for the speech.
        :type transript: basestring
        :return: Speech segment instance.
        :rtype: SpeechSegment
        """
        audio = AudioSegment.from_file(filepath)
        return cls(audio.samples, audio.sample_rate, transcript)

    @classmethod
    def from_bytes(cls, bytes, transcript):
        """Create speech segment from a byte string and corresponding
        transcript.
        
        :param bytes: Byte string containing audio samples.
        :type bytes: str
        :param transcript: Transcript text for the speech.
        :type transript: basestring
        :return: Speech segment instance.
        :rtype: Speech Segment
        """
        audio = AudioSegment.from_bytes(bytes)
        return cls(audio.samples, audio.sample_rate, transcript)

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
        for seg in segments:
            if sample_rate != seg._sample_rate:
                raise ValueError("Can't concatenate segments with "
                                 "different sample rates")
            if type(seg) is not cls:
                raise TypeError("Only speech segments of the same type "
                                "instance can be concatenated.")
            transcripts += seg._transcript
        samples = np.concatenate([seg.samples for seg in segments])
        return cls(samples, sample_rate, transcripts)

    @classmethod
    def slice_from_file(cls, filepath, transcript, start=None, end=None):
        """Loads a small section of an speech without having to load
        the entire file into the memory which can be incredibly wasteful.

        :param filepath: Filepath or file object to audio file.
        :type filepath: basestring|file
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
        :type transript: basestring
        :return: SpeechSegment instance of the specified slice of the input
                 speech file.
        :rtype: SpeechSegment
        """
        audio = AudioSegment.slice_from_file(filepath, start, end)
        return cls(audio.samples, audio.sample_rate, transcript)

    @classmethod
    def make_silence(cls, duration, sample_rate):
        """Creates a silent speech segment of the given duration and
        sample rate, transcript will be an empty string.

        :param duration: Length of silence in seconds.
        :type duration: float
        :param sample_rate: Sample rate.
        :type sample_rate: float
        :return: Silence of the given duration.
        :rtype: SpeechSegment
        """
        audio = AudioSegment.make_silence(duration, sample_rate)
        return cls(audio.samples, audio.sample_rate, "")

    @property
    def transcript(self):
        """Return the transcript text.

        :return: Transcript text for the speech.
        :rtype: basestring
        """
        return self._transcript
