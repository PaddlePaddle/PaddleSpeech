import numpy as np
import io
import soundfile


class AudioSegment(object):
    """Monaural audio segment abstraction.
    """

    def __init__(self, samples, sample_rate):
        if not samples.dtype == np.float32:
            raise ValueError("Sample data type of [%s] is not supported.")
        self._samples = samples
        self._sample_rate = sample_rate
        if self._samples.ndim >= 2:
            self._samples = np.mean(self._samples, 1)

    @classmethod
    def from_file(cls, filepath):
        samples, sample_rate = soundfile.read(filepath, dtype='float32')
        return cls(samples, sample_rate)

    @classmethod
    def from_bytes(cls, bytes):
        samples, sample_rate = soundfile.read(
            io.BytesIO(bytes), dtype='float32')
        return cls(samples, sample_rate)

    def apply_gain(self, gain):
        self.samples *= 10.**(gain / 20.)

    def resample(self, target_sample_rate):
        raise NotImplementedError()

    def change_speed(self, rate):
        raise NotImplementedError()

    @property
    def samples(self):
        return self._samples.copy()

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def duration(self):
        return self._samples.shape[0] / float(self._sample_rate)


class SpeechSegment(AudioSegment):
    def __init__(self, samples, sample_rate, transcript):
        AudioSegment.__init__(self, samples, sample_rate)
        self._transcript = transcript

    @classmethod
    def from_file(cls, filepath, transcript):
        audio = AudioSegment.from_file(filepath)
        return cls(audio.samples, audio.sample_rate, transcript)

    @classmethod
    def from_bytes(cls, bytes, transcript):
        audio = AudioSegment.from_bytes(bytes)
        return cls(audio.samples, audio.sample_rate, transcript)

    @property
    def transcript(self):
        return self._transcript
