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
"""Contains the audio segment class."""
import copy
import io
import random
import re
import struct

import numpy as np
import resampy
import soundfile
from scipy import signal

from .utility import convert_samples_from_float32
from .utility import convert_samples_to_float32
from .utility import subfile_from_tar


class AudioSegment():
    """Monaural audio segment abstraction.

    :param samples: Audio samples [num_samples x num_channels].
    :type samples: ndarray.float32
    :param sample_rate: Audio sample rate.
    :type sample_rate: int
    :raises TypeError: If the sample data type is not float or int.
    """
    def __init__(self, samples, sample_rate):
        """Create audio segment from samples.

        Samples are convert float32 internally, with int scaled to [-1, 1].
        """
        self._samples = self._convert_samples_to_float32(samples)
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

    def __eq__(self, other):
        """Return whether two objects are equal."""
        if type(other) is not type(self):
            return False
        if self._sample_rate != other._sample_rate:
            return False
        if self._samples.shape != other._samples.shape:
            return False
        if np.any(self.samples != other._samples):
            return False
        return True

    def __ne__(self, other):
        """Return whether two objects are unequal."""
        return not self.__eq__(other)

    def __str__(self):
        """Return human-readable representation of segment."""
        return ("%s: num_samples=%d, sample_rate=%d, duration=%.2fsec, "
                "rms=%.2fdB" % (type(self), self.num_samples, self.sample_rate,
                                self.duration, self.rms_db))

    @classmethod
    def from_file(cls, file, infos=None):
        """Create audio segment from audio file.

        Args:
            filepath (str|file): Filepath or file object to audio file.
            infos (TarLocalData, optional): tar2obj and tar2infos. Defaults to None.

        Returns:
            AudioSegment: Audio segment instance.
        """
        if isinstance(file, str) and re.findall(r".seqbin_\d+$", file):
            return cls.from_sequence_file(file)
        elif isinstance(file, str) and file.startswith('tar:'):
            return cls.from_file(subfile_from_tar(file, infos))
        else:
            samples, sample_rate = soundfile.read(file, dtype='float32')
            return cls(samples, sample_rate)

    @classmethod
    def slice_from_file(cls, file, start=None, end=None):
        """Loads a small section of an audio without having to load
        the entire file into the memory which can be incredibly wasteful.

        :param file: Input audio filepath or file object.
        :type file: str|file
        :param start: Start time in seconds. If start is negative, it wraps
                      around from the end. If not provided, this function
                      reads from the very beginning.
        :type start: float
        :param end: End time in seconds. If end is negative, it wraps around
                    from the end. If not provided, the default behvaior is
                    to read to the end of the file.
        :type end: float
        :return: AudioSegment instance of the specified slice of the input
                 audio file.
        :rtype: AudioSegment
        :raise ValueError: If start or end is incorrectly set, e.g. out of
                           bounds in time.
        """
        sndfile = soundfile.SoundFile(file)
        sample_rate = sndfile.samplerate
        duration = float(len(sndfile)) / sample_rate
        start = 0. if start is None else start
        end = duration if end is None else end
        if start < 0.0:
            start += duration
        if end < 0.0:
            end += duration
        if start < 0.0:
            raise ValueError("The slice start position (%f s) is out of "
                             "bounds." % start)
        if end < 0.0:
            raise ValueError("The slice end position (%f s) is out of bounds." %
                             end)
        if start > end:
            raise ValueError("The slice start position (%f s) is later than "
                             "the slice end position (%f s)." % (start, end))
        if end > duration:
            raise ValueError("The slice end position (%f s) is out of bounds "
                             "(> %f s)" % (end, duration))
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)
        sndfile.seek(start_frame)
        data = sndfile.read(frames=end_frame - start_frame, dtype='float32')
        return cls(data, sample_rate)

    @classmethod
    def from_sequence_file(cls, filepath):
        """Create audio segment from sequence file. Sequence file is a binary
        file containing a collection of multiple audio files, with several
        header bytes in the head indicating the offsets of each audio byte data
        chunk.

        The format is:

            4 bytes (int, version),
            4 bytes (int, num of utterance),
            4 bytes (int, bytes per header),
            [bytes_per_header*(num_utterance+1)] bytes (offsets for each audio),
            audio_bytes_data_of_1st_utterance,
            audio_bytes_data_of_2nd_utterance,
            ......

        Sequence file name must end with ".seqbin". And the filename of the 5th
        utterance's audio file in sequence file "xxx.seqbin" must be
        "xxx.seqbin_5", with "5" indicating the utterance index within this
        sequence file (starting from 1).

        :param filepath: Filepath of sequence file.
        :type filepath: str
        :return: Audio segment instance.
        :rtype: AudioSegment
        """
        # parse filepath
        matches = re.match(r"(.+\.seqbin)_(\d+)", filepath)
        if matches is None:
            raise IOError("File type of %s is not supported" % filepath)
        filename = matches.group(1)
        fileno = int(matches.group(2))

        # read headers
        f = io.open(filename, mode='rb', encoding='utf8')
        version = f.read(4)
        num_utterances = struct.unpack("i", f.read(4))[0]
        bytes_per_header = struct.unpack("i", f.read(4))[0]
        header_bytes = f.read(bytes_per_header * (num_utterances + 1))
        header = [
            struct.unpack(
                "i", header_bytes[bytes_per_header * i:bytes_per_header *
                                  (i + 1)])[0]
            for i in range(num_utterances + 1)
        ]

        # read audio bytes
        f.seek(header[fileno - 1])
        audio_bytes = f.read(header[fileno] - header[fileno - 1])
        f.close()

        # create audio segment
        try:
            return cls.from_bytes(audio_bytes)
        except Exception as e:
            samples = np.frombuffer(audio_bytes, dtype='int16')
            return cls(samples=samples, sample_rate=8000)

    @classmethod
    def from_bytes(cls, bytes):
        """Create audio segment from a byte string containing audio samples.

        :param bytes: Byte string containing audio samples.
        :type bytes: str
        :return: Audio segment instance.
        :rtype: AudioSegment
        """
        samples, sample_rate = soundfile.read(io.BytesIO(bytes),
                                              dtype='float32')
        return cls(samples, sample_rate)

    @classmethod
    def from_pcm(cls, samples, sample_rate):
        """Create audio segment from a byte string containing audio samples.
        :param samples: Audio samples [num_samples x num_channels].
        :type samples: numpy.ndarray
        :param sample_rate: Audio sample rate.
        :type sample_rate: int
        :return: Audio segment instance.
        :rtype: AudioSegment
        """
        return cls(samples, sample_rate)

    @classmethod
    def concatenate(cls, *segments):
        """Concatenate an arbitrary number of audio segments together.

        :param *segments: Input audio segments to be concatenated.
        :type *segments: tuple of AudioSegment
        :return: Audio segment instance as concatenating results.
        :rtype: AudioSegment
        :raises ValueError: If the number of segments is zero, or if the
                            sample_rate of any segments does not match.
        :raises TypeError: If any segment is not AudioSegment instance.
        """
        # Perform basic sanity-checks.
        if len(segments) == 0:
            raise ValueError("No audio segments are given to concatenate.")
        sample_rate = segments[0]._sample_rate
        for seg in segments:
            if sample_rate != seg._sample_rate:
                raise ValueError("Can't concatenate segments with "
                                 "different sample rates")
            if type(seg) is not cls:
                raise TypeError("Only audio segments of the same type "
                                "can be concatenated.")
        samples = np.concatenate([seg.samples for seg in segments])
        return cls(samples, sample_rate)

    @classmethod
    def make_silence(cls, duration, sample_rate):
        """Creates a silent audio segment of the given duration and sample rate.

        :param duration: Length of silence in seconds.
        :type duration: float
        :param sample_rate: Sample rate.
        :type sample_rate: float
        :return: Silent AudioSegment instance of the given duration.
        :rtype: AudioSegment
        """
        samples = np.zeros(int(duration * sample_rate))
        return cls(samples, sample_rate)

    def to_wav_file(self, filepath, dtype='float32'):
        """Save audio segment to disk as wav file.

        :param filepath: WAV filepath or file object to save the
                         audio segment.
        :type filepath: str|file
        :param dtype: Subtype for audio file. Options: 'int16', 'int32',
                      'float32', 'float64'. Default is 'float32'.
        :type dtype: str
        :raises TypeError: If dtype is not supported.
        """
        samples = self._convert_samples_from_float32(self._samples, dtype)
        subtype_map = {
            'int16': 'PCM_16',
            'int32': 'PCM_32',
            'float32': 'FLOAT',
            'float64': 'DOUBLE'
        }
        soundfile.write(filepath,
                        samples,
                        self._sample_rate,
                        format='WAV',
                        subtype=subtype_map[dtype])

    def superimpose(self, other):
        """Add samples from another segment to those of this segment
        (sample-wise addition, not segment concatenation).

        Note that this is an in-place transformation.

        :param other: Segment containing samples to be added in.
        :type other: AudioSegments
        :raise TypeError: If type of two segments don't match.
        :raise ValueError: If the sample rates of the two segments are not
                           equal, or if the lengths of segments don't match.
        """
        if isinstance(other, type(self)):
            raise TypeError("Cannot add segments of different types: %s "
                            "and %s." % (type(self), type(other)))
        if self._sample_rate != other._sample_rate:
            raise ValueError("Sample rates must match to add segments.")
        if len(self._samples) != len(other._samples):
            raise ValueError("Segment lengths must match to add segments.")
        self._samples += other._samples

    def to_bytes(self, dtype='float32'):
        """Create a byte string containing the audio content.

        :param dtype: Data type for export samples. Options: 'int16', 'int32',
                      'float32', 'float64'. Default is 'float32'.
        :type dtype: str
        :return: Byte string containing audio content.
        :rtype: str
        """
        samples = self._convert_samples_from_float32(self._samples, dtype)
        return samples.tostring()

    def to(self, dtype='int16'):
        """Create a `dtype` audio content.

        :param dtype: Data type for export samples. Options: 'int16', 'int32',
                      'float32', 'float64'. Default is 'float32'.
        :type dtype: str
        :return: np.ndarray containing `dtype` audio content.
        :rtype: str
        """
        samples = self._convert_samples_from_float32(self._samples, dtype)
        return samples

    def gain_db(self, gain):
        """Apply gain in decibels to samples.

        Note that this is an in-place transformation.

        :param gain: Gain in decibels to apply to samples.
        :type gain: float|1darray
        """
        self._samples *= 10.**(gain / 20.)

    def change_speed(self, speed_rate):
        """Change the audio speed by linear interpolation.

        Note that this is an in-place transformation.

        :param speed_rate: Rate of speed change:
                           speed_rate > 1.0, speed up the audio;
                           speed_rate = 1.0, unchanged;
                           speed_rate < 1.0, slow down the audio;
                           speed_rate <= 0.0, not allowed, raise ValueError.
        :type speed_rate: float
        :raises ValueError: If speed_rate <= 0.0.
        """
        if speed_rate == 1.0:
            return
        if speed_rate <= 0:
            raise ValueError("speed_rate should be greater than zero.")

        # numpy
        # old_length = self._samples.shape[0]
        # new_length = int(old_length / speed_rate)
        # old_indices = np.arange(old_length)
        # new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        # self._samples = np.interp(new_indices, old_indices, self._samples)

        # sox, slow
        try:
            import soxbindings as sox
        except ImportError:
            try:
                from paddlespeech.s2t.utils import dynamic_pip_install
                package = "sox"
                dynamic_pip_install.install(package)
                package = "soxbindings"
                dynamic_pip_install.install(package)
                import soxbindings as sox
            except Exception:
                raise RuntimeError(
                    "Can not install soxbindings on your system.")

        tfm = sox.Transformer()
        tfm.set_globals(multithread=False)
        tfm.speed(speed_rate)
        self._samples = tfm.build_array(
            input_array=self._samples,
            sample_rate_in=self._sample_rate).squeeze(-1).astype(
                np.float32).copy()

    def normalize(self, target_db=-20, max_gain_db=300.0):
        """Normalize audio to be of the desired RMS value in decibels.

        Note that this is an in-place transformation.

        :param target_db: Target RMS value in decibels. This value should be
                          less than 0.0 as 0.0 is full-scale audio.
        :type target_db: float
        :param max_gain_db: Max amount of gain in dB that can be applied for
                            normalization. This is to prevent nans when
                            attempting to normalize a signal consisting of
                            all zeros.
        :type max_gain_db: float
        :raises ValueError: If the required gain to normalize the segment to
                            the target_db value exceeds max_gain_db.
        """
        gain = target_db - self.rms_db
        if gain > max_gain_db:
            raise ValueError(
                "Unable to normalize segment to %f dB because the "
                "the probable gain have exceeds max_gain_db (%f dB)" %
                (target_db, max_gain_db))
        self.gain_db(min(max_gain_db, target_db - self.rms_db))

    def normalize_online_bayesian(self,
                                  target_db,
                                  prior_db,
                                  prior_samples,
                                  startup_delay=0.0):
        """Normalize audio using a production-compatible online/causal
        algorithm. This uses an exponential likelihood and gamma prior to
        make online estimates of the RMS even when there are very few samples.

        Note that this is an in-place transformation.

        :param target_db: Target RMS value in decibels.
        :type target_bd: float
        :param prior_db: Prior RMS estimate in decibels.
        :type prior_db: float
        :param prior_samples: Prior strength in number of samples.
        :type prior_samples: float
        :param startup_delay: Default 0.0s. If provided, this function will
                              accrue statistics for the first startup_delay
                              seconds before applying online normalization.
        :type startup_delay: float
        """
        # Estimate total RMS online.
        startup_sample_idx = min(self.num_samples - 1,
                                 int(self.sample_rate * startup_delay))
        prior_mean_squared = 10.**(prior_db / 10.)
        prior_sum_of_squares = prior_mean_squared * prior_samples
        cumsum_of_squares = np.cumsum(self.samples**2)
        sample_count = np.arange(self.num_samples) + 1
        if startup_sample_idx > 0:
            cumsum_of_squares[:startup_sample_idx] = \
                cumsum_of_squares[startup_sample_idx]
            sample_count[:startup_sample_idx] = \
                sample_count[startup_sample_idx]
        mean_squared_estimate = ((cumsum_of_squares + prior_sum_of_squares) /
                                 (sample_count + prior_samples))
        rms_estimate_db = 10 * np.log10(mean_squared_estimate)
        # Compute required time-varying gain.
        gain_db = target_db - rms_estimate_db
        self.gain_db(gain_db)

    def resample(self, target_sample_rate, filter='kaiser_best'):
        """Resample the audio to a target sample rate.

        Note that this is an in-place transformation.

        :param target_sample_rate: Target sample rate.
        :type target_sample_rate: int
        :param filter: The resampling filter to use one of {'kaiser_best',
                       'kaiser_fast'}.
        :type filter: str
        """
        self._samples = resampy.resample(self.samples,
                                         self.sample_rate,
                                         target_sample_rate,
                                         filter=filter)
        self._sample_rate = target_sample_rate

    def pad_silence(self, duration, sides='both'):
        """Pad this audio sample with a period of silence.

        Note that this is an in-place transformation.

        :param duration: Length of silence in seconds to pad.
        :type duration: float
        :param sides: Position for padding:
                     'beginning' - adds silence in the beginning;
                     'end' - adds silence in the end;
                     'both' - adds silence in both the beginning and the end.
        :type sides: str
        :raises ValueError: If sides is not supported.
        """
        if duration == 0.0:
            return self
        cls = type(self)
        silence = self.make_silence(duration, self._sample_rate)
        if sides == "beginning":
            padded = cls.concatenate(silence, self)
        elif sides == "end":
            padded = cls.concatenate(self, silence)
        elif sides == "both":
            padded = cls.concatenate(silence, self, silence)
        else:
            raise ValueError("Unknown value for the sides %s" % sides)
        self._samples = padded._samples

    def shift(self, shift_ms):
        """Shift the audio in time. If `shift_ms` is positive, shift with time
        advance; if negative, shift with time delay. Silence are padded to
        keep the duration unchanged.

        Note that this is an in-place transformation.

        :param shift_ms: Shift time in millseconds. If positive, shift with
                         time advance; if negative; shift with time delay.
        :type shift_ms: float
        :raises ValueError: If shift_ms is longer than audio duration.
        """
        if abs(shift_ms) / 1000.0 > self.duration:
            raise ValueError("Absolute value of shift_ms should be smaller "
                             "than audio duration.")
        shift_samples = int(shift_ms * self._sample_rate / 1000)
        if shift_samples > 0:
            # time advance
            self._samples[:-shift_samples] = self._samples[shift_samples:]
            self._samples[-shift_samples:] = 0
        elif shift_samples < 0:
            # time delay
            self._samples[-shift_samples:] = self._samples[:shift_samples]
            self._samples[:-shift_samples] = 0

    def subsegment(self, start_sec=None, end_sec=None):
        """Cut the AudioSegment between given boundaries.

        Note that this is an in-place transformation.

        :param start_sec: Beginning of subsegment in seconds.
        :type start_sec: float
        :param end_sec: End of subsegment in seconds.
        :type end_sec: float
        :raise ValueError: If start_sec or end_sec is incorrectly set, e.g. out
                           of bounds in time.
        """
        start_sec = 0.0 if start_sec is None else start_sec
        end_sec = self.duration if end_sec is None else end_sec
        if start_sec < 0.0:
            start_sec = self.duration + start_sec
        if end_sec < 0.0:
            end_sec = self.duration + end_sec
        if start_sec < 0.0:
            raise ValueError("The slice start position (%f s) is out of "
                             "bounds." % start_sec)
        if end_sec < 0.0:
            raise ValueError("The slice end position (%f s) is out of bounds." %
                             end_sec)
        if start_sec > end_sec:
            raise ValueError("The slice start position (%f s) is later than "
                             "the end position (%f s)." % (start_sec, end_sec))
        if end_sec > self.duration:
            raise ValueError("The slice end position (%f s) is out of bounds "
                             "(> %f s)" % (end_sec, self.duration))
        start_sample = int(round(start_sec * self._sample_rate))
        end_sample = int(round(end_sec * self._sample_rate))
        self._samples = self._samples[start_sample:end_sample]

    def random_subsegment(self, subsegment_length, rng=None):
        """Cut the specified length of the audiosegment randomly.

        Note that this is an in-place transformation.

        :param subsegment_length: Subsegment length in seconds.
        :type subsegment_length: float
        :param rng: Random number generator state.
        :type rng: random.Random
        :raises ValueError: If the length of subsegment is greater than
                            the origineal segemnt.
        """
        rng = random.Random() if rng is None else rng
        if subsegment_length > self.duration:
            raise ValueError("Length of subsegment must not be greater "
                             "than original segment.")
        start_time = rng.uniform(0.0, self.duration - subsegment_length)
        self.subsegment(start_time, start_time + subsegment_length)

    def convolve(self, impulse_segment, allow_resample=False):
        """Convolve this audio segment with the given impulse segment.

        Note that this is an in-place transformation.

        :param impulse_segment: Impulse response segments.
        :type impulse_segment: AudioSegment
        :param allow_resample: Indicates whether resampling is allowed when
                               the impulse_segment has a different sample
                               rate from this signal.
        :type allow_resample: bool
        :raises ValueError: If the sample rate is not match between two
                            audio segments when resample is not allowed.
        """
        if allow_resample and self.sample_rate != impulse_segment.sample_rate:
            impulse_segment.resample(self.sample_rate)
        if self.sample_rate != impulse_segment.sample_rate:
            raise ValueError("Impulse segment's sample rate (%d Hz) is not "
                             "equal to base signal sample rate (%d Hz)." %
                             (impulse_segment.sample_rate, self.sample_rate))
        samples = signal.fftconvolve(self.samples, impulse_segment.samples,
                                     "full")
        self._samples = samples

    def convolve_and_normalize(self, impulse_segment, allow_resample=False):
        """Convolve and normalize the resulting audio segment so that it
        has the same average power as the input signal.

        Note that this is an in-place transformation.

        :param impulse_segment: Impulse response segments.
        :type impulse_segment: AudioSegment
        :param allow_resample: Indicates whether resampling is allowed when
                               the impulse_segment has a different sample
                               rate from this signal.
        :type allow_resample: bool
        """
        target_db = self.rms_db
        self.convolve(impulse_segment, allow_resample=allow_resample)
        self.normalize(target_db)

    def add_noise(self,
                  noise,
                  snr_dB,
                  allow_downsampling=False,
                  max_gain_db=300.0,
                  rng=None):
        """Add the given noise segment at a specific signal-to-noise ratio.
        If the noise segment is longer than this segment, a random subsegment
        of matching length is sampled from it and used instead.

        Note that this is an in-place transformation.

        :param noise: Noise signal to add.
        :type noise: AudioSegment
        :param snr_dB: Signal-to-Noise Ratio, in decibels.
        :type snr_dB: float
        :param allow_downsampling: Whether to allow the noise signal to be
                                   downsampled to match the base signal sample
                                   rate.
        :type allow_downsampling: bool
        :param max_gain_db: Maximum amount of gain to apply to noise signal
                            before adding it in. This is to prevent attempting
                            to apply infinite gain to a zero signal.
        :type max_gain_db: float
        :param rng: Random number generator state.
        :type rng: None|random.Random
        :raises ValueError: If the sample rate does not match between the two
                            audio segments when downsampling is not allowed, or
                            if the duration of noise segments is shorter than
                            original audio segments.
        """
        rng = random.Random() if rng is None else rng
        if allow_downsampling and noise.sample_rate > self.sample_rate:
            noise = noise.resample(self.sample_rate)
        if noise.sample_rate != self.sample_rate:
            raise ValueError("Noise sample rate (%d Hz) is not equal to base "
                             "signal sample rate (%d Hz)." %
                             (noise.sample_rate, self.sample_rate))
        if noise.duration < self.duration:
            raise ValueError("Noise signal (%f sec) must be at least as long as"
                             " base signal (%f sec)." %
                             (noise.duration, self.duration))
        noise_gain_db = min(self.rms_db - noise.rms_db - snr_dB, max_gain_db)
        noise_new = copy.deepcopy(noise)
        noise_new.random_subsegment(self.duration, rng=rng)
        noise_new.gain_db(noise_gain_db)
        self.superimpose(noise_new)

    @property
    def samples(self):
        """Return audio samples.

        :return: Audio samples.
        :rtype: ndarray
        """
        return self._samples.copy()

    @property
    def sample_rate(self):
        """Return audio sample rate.

        :return: Audio sample rate.
        :rtype: int
        """
        return self._sample_rate

    @property
    def num_samples(self):
        """Return number of samples.

        :return: Number of samples.
        :rtype: int
        """
        return self._samples.shape[0]

    @property
    def duration(self):
        """Return audio duration.

        :return: Audio duration in seconds.
        :rtype: float
        """
        return self._samples.shape[0] / float(self._sample_rate)

    @property
    def rms_db(self):
        """Return root mean square energy of the audio in decibels.

        :return: Root mean square energy in decibels.
        :rtype: float
        """
        # square root => multiply by 10 instead of 20 for dBs
        mean_square = np.mean(self._samples**2)
        return 10 * np.log10(mean_square)

    def _convert_samples_to_float32(self, samples):
        """Convert sample type to float32.

        Audio sample type is usually integer or float-point.
        Integers will be scaled to [-1, 1] in float32.
        """
        return convert_samples_to_float32(samples)

    def _convert_samples_from_float32(self, samples, dtype):
        """Convert sample type from float32 to dtype.

        Audio sample type is usually integer or float-point. For integer
        type, float32 will be rescaled from [-1, 1] to the maximum range
        supported by the integer type.

        This is for writing a audio file.
        """
        return convert_samples_from_float32(samples, dtype)
