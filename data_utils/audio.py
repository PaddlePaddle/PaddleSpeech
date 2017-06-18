"""Contains the audio segment class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import io
import soundfile


class AudioSegment(object):
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
    def from_file(cls, file):
        """Create audio segment from audio file.
        
        :param filepath: Filepath or file object to audio file.
        :type filepath: basestring|file
        :return: Audio segment instance.
        :rtype: AudioSegment
        """
        samples, sample_rate = soundfile.read(file, dtype='float32')
        return cls(samples, sample_rate)

    @classmethod
    def from_bytes(cls, bytes):
        """Create audio segment from a byte string containing audio samples.
        
        :param bytes: Byte string containing audio samples.
        :type bytes: str
        :return: Audio segment instance.
        :rtype: AudioSegment
        """
        samples, sample_rate = soundfile.read(
            io.BytesIO(bytes), dtype='float32')
        return cls(samples, sample_rate)

    def to_wav_file(self, filepath, dtype='float32'):
        """Save audio segment to disk as wav file.
        
        :param filepath: WAV filepath or file object to save the
                         audio segment.
        :type filepath: basestring|file
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
        soundfile.write(
            filepath,
            samples,
            self._sample_rate,
            format='WAV',
            subtype=subtype_map[dtype])

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

    def apply_gain(self, gain):
        """Apply gain in decibels to samples.

        Note that this is an in-place transformation.
        
        :param gain: Gain in decibels to apply to samples. 
        :type gain: float
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
        if speed_rate <= 0:
            raise ValueError("speed_rate should be greater than zero.")
        old_length = self._samples.shape[0]
        new_length = int(old_length / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        self._samples = np.interp(new_indices, old_indices, self._samples)

    def normalize(self, target_sample_rate):
        raise NotImplementedError()

    def resample(self, target_sample_rate):
        raise NotImplementedError()

    def pad_silence(self, duration, sides='both'):
        raise NotImplementedError()

    def subsegment(self, start_sec=None, end_sec=None):
        raise NotImplementedError()

    def convolve(self, filter, allow_resample=False):
        raise NotImplementedError()

    def convolve_and_normalize(self, filter, allow_resample=False):
        raise NotImplementedError()

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
        return self._samples.shape(0)

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
        float32_samples = samples.astype('float32')
        if samples.dtype in np.sctypes['int']:
            bits = np.iinfo(samples.dtype).bits
            float32_samples *= (1. / 2**(bits - 1))
        elif samples.dtype in np.sctypes['float']:
            pass
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return float32_samples

    def _convert_samples_from_float32(self, samples, dtype):
        """Convert sample type from float32 to dtype.
        
        Audio sample type is usually integer or float-point. For integer
        type, float32 will be rescaled from [-1, 1] to the maximum range
        supported by the integer type.
        
        This is for writing a audio file.
        """
        dtype = np.dtype(dtype)
        output_samples = samples.copy()
        if dtype in np.sctypes['int']:
            bits = np.iinfo(dtype).bits
            output_samples *= (2**(bits - 1) / 1.)
            min_val = np.iinfo(dtype).min
            max_val = np.iinfo(dtype).max
            output_samples[output_samples > max_val] = max_val
            output_samples[output_samples < min_val] = min_val
        elif samples.dtype in np.sctypes['float']:
            min_val = np.finfo(dtype).min
            max_val = np.finfo(dtype).max
            output_samples[output_samples > max_val] = max_val
            output_samples[output_samples < min_val] = min_val
        else:
            raise TypeError("Unsupported sample type: %s." % samples.dtype)
        return output_samples.astype(dtype)
