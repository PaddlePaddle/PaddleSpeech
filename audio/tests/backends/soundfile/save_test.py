import io
import unittest
from unittest.mock import patch

import numpy as np
import paddle
import soundfile
from common import fetch_wav_subtype
from common import parameterize
from common import skipIfFormatNotSupported
from paddleaudio.backends import soundfile_backend

from common_utils import get_wav_data
from common_utils import load_wav
from common_utils import nested_params
from common_utils import TempDirMixin


class MockedSaveTest(unittest.TestCase):
    @nested_params(
        ["float32", "int32"],
        [8000, 16000],
        [1, 2],
        [False, True],
        [
            (None, None),
            ("PCM_U", None),
            ("PCM_U", 8),
            ("PCM_S", None),
            ("PCM_S", 16),
            ("PCM_S", 32),
            ("PCM_F", None),
            ("PCM_F", 32),
            ("PCM_F", 64),
            ("ULAW", None),
            ("ULAW", 8),
            ("ALAW", None),
            ("ALAW", 8),
        ], )
    @patch("soundfile.write")
    def test_wav(self, dtype, sample_rate, num_channels, channels_first,
                 enc_params, mocked_write):
        """soundfile_backend.save passes correct subtype to soundfile.write when WAV"""
        filepath = "foo.wav"
        input_tensor = get_wav_data(
            dtype,
            num_channels,
            num_frames=3 * sample_rate,
            normalize=dtype == "float32",
            channels_first=channels_first, )
        input_tensor = paddle.transpose(input_tensor, [1, 0])

        encoding, bits_per_sample = enc_params
        soundfile_backend.save(
            filepath,
            input_tensor,
            sample_rate,
            channels_first=channels_first,
            encoding=encoding,
            bits_per_sample=bits_per_sample, )

        # on +Py3.8 call_args.kwargs is more descreptive
        args = mocked_write.call_args[1]
        assert args["file"] == filepath
        assert args["samplerate"] == sample_rate
        assert args["subtype"] == fetch_wav_subtype(dtype, encoding,
                                                    bits_per_sample)
        assert args["format"] is None
        tensor_result = paddle.transpose(
            input_tensor, [1, 0]) if channels_first else input_tensor
        #self.assertEqual(args["data"], tensor_result.numpy())
        np.testing.assert_array_almost_equal(args["data"].numpy(),
                                             tensor_result.numpy())

    @patch("soundfile.write")
    def assert_non_wav(
            self,
            fmt,
            dtype,
            sample_rate,
            num_channels,
            channels_first,
            mocked_write,
            encoding=None,
            bits_per_sample=None, ):
        """soundfile_backend.save passes correct subtype and format to soundfile.write when SPHERE"""
        filepath = f"foo.{fmt}"
        input_tensor = get_wav_data(
            dtype,
            num_channels,
            num_frames=3 * sample_rate,
            normalize=False,
            channels_first=channels_first, )
        input_tensor = paddle.transpose(input_tensor, [1, 0])

        expected_data = paddle.transpose(
            input_tensor, [1, 0]) if channels_first else input_tensor

        soundfile_backend.save(
            filepath,
            input_tensor,
            sample_rate,
            channels_first,
            encoding=encoding,
            bits_per_sample=bits_per_sample, )

        # on +Py3.8 call_args.kwargs is more descreptive
        args = mocked_write.call_args[1]
        assert args["file"] == filepath
        assert args["samplerate"] == sample_rate
        if fmt in ["sph", "nist", "nis"]:
            assert args["format"] == "NIST"
        else:
            assert args["format"] is None
        np.testing.assert_array_almost_equal(args["data"].numpy(),
                                             expected_data.numpy())
        #self.assertEqual(args["data"], expected_data)

    @nested_params(
        ["sph", "nist", "nis"],
        ["int32"],
        [8000, 16000],
        [1, 2],
        [False, True],
        [
            ("PCM_S", 8),
            ("PCM_S", 16),
            ("PCM_S", 24),
            ("PCM_S", 32),
            ("ULAW", 8),
            ("ALAW", 8),
            ("ALAW", 16),
            ("ALAW", 24),
            ("ALAW", 32),
        ], )
    def test_sph(self, fmt, dtype, sample_rate, num_channels, channels_first,
                 enc_params):
        """soundfile_backend.save passes default format and subtype (None-s) to
        soundfile.write when not WAV"""
        encoding, bits_per_sample = enc_params
        self.assert_non_wav(
            fmt,
            dtype,
            sample_rate,
            num_channels,
            channels_first,
            encoding=encoding,
            bits_per_sample=bits_per_sample)

    @parameterize(
        ["int32"],
        [8000, 16000],
        [1, 2],
        [False, True],
        [8, 16, 24], )
    def test_flac(self, dtype, sample_rate, num_channels, channels_first,
                  bits_per_sample):
        """soundfile_backend.save passes default format and subtype (None-s) to
        soundfile.write when not WAV"""
        self.assert_non_wav(
            "flac",
            dtype,
            sample_rate,
            num_channels,
            channels_first,
            bits_per_sample=bits_per_sample)

    @parameterize(
        ["int32"],
        [8000, 16000],
        [1, 2],
        [False, True], )
    def test_ogg(self, dtype, sample_rate, num_channels, channels_first):
        """soundfile_backend.save passes default format and subtype (None-s) to
        soundfile.write when not WAV"""
        self.assert_non_wav("ogg", dtype, sample_rate, num_channels,
                            channels_first)


class SaveTestBase(TempDirMixin, unittest.TestCase):
    def assert_wav(self, dtype, sample_rate, num_channels, num_frames):
        """`soundfile_backend.save` can save wav format."""
        path = self.get_temp_path("data.wav")
        expected = get_wav_data(
            dtype, num_channels, num_frames=num_frames, normalize=False)
        soundfile_backend.save(path, expected, sample_rate)
        found, sr = load_wav(path, normalize=False)
        assert sample_rate == sr
        #self.assertEqual(found, expected)
        np.testing.assert_array_almost_equal(found.numpy(), expected.numpy())

    def _assert_non_wav(self, fmt, dtype, sample_rate, num_channels):
        """`soundfile_backend.save` can save non-wav format.

        Due to precision missmatch, and the lack of alternative way to decode the
        resulting files without using soundfile, only meta data are validated.
        """
        num_frames = sample_rate * 3
        path = self.get_temp_path(f"data.{fmt}")
        expected = get_wav_data(
            dtype, num_channels, num_frames=num_frames, normalize=False)
        soundfile_backend.save(path, expected, sample_rate)
        sinfo = soundfile.info(path)
        assert sinfo.format == fmt.upper()
        #assert sinfo.frames == num_frames this go wrong
        assert sinfo.channels == num_channels
        assert sinfo.samplerate == sample_rate

    def assert_flac(self, dtype, sample_rate, num_channels):
        """`soundfile_backend.save` can save flac format."""
        self._assert_non_wav("flac", dtype, sample_rate, num_channels)

    def assert_sphere(self, dtype, sample_rate, num_channels):
        """`soundfile_backend.save` can save sph format."""
        self._assert_non_wav("nist", dtype, sample_rate, num_channels)

    def assert_ogg(self, dtype, sample_rate, num_channels):
        """`soundfile_backend.save` can save ogg format.

        As we cannot inspect the OGG format (it's lossy), we only check the metadata.
        """
        self._assert_non_wav("ogg", dtype, sample_rate, num_channels)


class TestSave(SaveTestBase):
    @parameterize(
        ["float32", "int32"],
        [8000, 16000],
        [1, 2], )
    def test_wav(self, dtype, sample_rate, num_channels):
        """`soundfile_backend.save` can save wav format."""
        self.assert_wav(dtype, sample_rate, num_channels, num_frames=None)

    @parameterize(
        ["float32", "int32"],
        [4, 8, 16, 32], )
    def test_multiple_channels(self, dtype, num_channels):
        """`soundfile_backend.save` can save wav with more than 2 channels."""
        sample_rate = 8000
        self.assert_wav(dtype, sample_rate, num_channels, num_frames=None)

    @parameterize(
        ["int32"],
        [8000, 16000],
        [1, 2], )
    @skipIfFormatNotSupported("NIST")
    def test_sphere(self, dtype, sample_rate, num_channels):
        """`soundfile_backend.save` can save sph format."""
        self.assert_sphere(dtype, sample_rate, num_channels)

    @parameterize(
        [8000, 16000],
        [1, 2], )
    @skipIfFormatNotSupported("FLAC")
    def test_flac(self, sample_rate, num_channels):
        """`soundfile_backend.save` can save flac format."""
        self.assert_flac("float32", sample_rate, num_channels)

    @parameterize(
        [8000, 16000],
        [1, 2], )
    @skipIfFormatNotSupported("OGG")
    def test_ogg(self, sample_rate, num_channels):
        """`soundfile_backend.save` can save ogg/vorbis format."""
        self.assert_ogg("float32", sample_rate, num_channels)


class TestSaveParams(TempDirMixin, unittest.TestCase):
    """Test the correctness of optional parameters of `soundfile_backend.save`"""

    @parameterize([True, False])
    def test_channels_first(self, channels_first):
        """channels_first swaps axes"""
        path = self.get_temp_path("data.wav")
        data = get_wav_data("int32", 2, channels_first=channels_first)
        soundfile_backend.save(path, data, 8000, channels_first=channels_first)
        found = load_wav(path)[0]
        expected = data if channels_first else data.transpose([1, 0])
        #self.assertEqual(found, expected, atol=1e-4, rtol=1e-8)
        np.testing.assert_array_almost_equal(found.numpy(), expected.numpy())


class TestFileObject(TempDirMixin, unittest.TestCase):
    def _test_fileobj(self, ext):
        """Saving audio to file-like object works"""
        sample_rate = 16000
        path = self.get_temp_path(f"test.{ext}")

        subtype = "FLOAT" if ext == "wav" else None
        data = get_wav_data("float32", num_channels=2)
        soundfile.write(path, data.numpy().T, sample_rate, subtype=subtype)
        expected = soundfile.read(path, dtype="float32")[0]

        fileobj = io.BytesIO()
        soundfile_backend.save(fileobj, data, sample_rate, format=ext)
        fileobj.seek(0)
        found, sr = soundfile.read(fileobj, dtype="float32")

        assert sr == sample_rate
        #self.assertEqual(expected, found, atol=1e-4, rtol=1e-8)
        np.testing.assert_array_almost_equal(found, expected)

    def test_fileobj_wav(self):
        """Saving audio via file-like object works"""
        self._test_fileobj("wav")

    @skipIfFormatNotSupported("FLAC")
    def test_fileobj_flac(self):
        """Saving audio via file-like object works"""
        self._test_fileobj("flac")

    @skipIfFormatNotSupported("NIST")
    def test_fileobj_nist(self):
        """Saving audio via file-like object works"""
        self._test_fileobj("NIST")

    @skipIfFormatNotSupported("OGG")
    def test_fileobj_ogg(self):
        """Saving audio via file-like object works"""
        self._test_fileobj("OGG")


if __name__ == '__main__':
    unittest.main()
