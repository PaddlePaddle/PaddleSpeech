import itertools
import platform
import unittest
if platform.system() == "Windows":
    import warnings
    warnings.warn("sox io not support in Windows, please skip test.")
    exit()

from parameterized import parameterized
import numpy as np
from paddleaudio.backends import sox_io_backend

from common_utils import (
    get_wav_data,
    load_wav,
    save_wav, )

#code is from:https://github.com/pytorch/audio/blob/main/torchaudio/test/torchaudio_unittest/backend/sox_io/load_test.py


class TestLoad(unittest.TestCase):
    def assert_wav(self, dtype, sample_rate, num_channels, normalize, duration):
        """`sox_io_backend.load` can load wav format correctly.

        Wav data loaded with sox_io backend should match those with scipy
        """
        path = 'testdata/reference.wav'
        data = get_wav_data(
            dtype,
            num_channels,
            normalize=normalize,
            num_frames=duration * sample_rate)
        save_wav(path, data, sample_rate)
        expected = load_wav(path, normalize=normalize)[0]
        data, sr = sox_io_backend.load(path, normalize=normalize)
        assert sr == sample_rate
        np.testing.assert_array_almost_equal(data, expected, decimal=4)

    @parameterized.expand(
        list(
            itertools.product(
                [
                    "float64",
                    "float32",
                    "int32",
                ],
                [8000, 16000],
                [1, 2],
                [False, True], )), )
    def test_wav(self, dtype, sample_rate, num_channels, normalize):
        """`sox_io_backend.load` can load wav format correctly."""
        self.assert_wav(dtype, sample_rate, num_channels, normalize, duration=1)


if __name__ == '__main__':
    unittest.main()
