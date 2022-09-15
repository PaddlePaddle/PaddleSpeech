#this code is from: https://github.com/pytorch/audio/blob/main/test/torchaudio_unittest/backend/soundfile/info_test.py

import tarfile
import warnings
import unittest
from unittest.mock import patch

import paddle
from paddleaudio._internal import module_utils as _mod_utils
from paddleaudio.backends import soundfile_backend
from tests.backends.common import get_bits_per_sample, get_encoding 
from tests.common_utils import (
    get_wav_data,
    nested_params,
    save_wav,
    TempDirMixin,
)

from common import parameterize, skipIfFormatNotSupported

import soundfile


class TestInfo(TempDirMixin, unittest.TestCase):
    @parameterize(
        ["float32", "int32"],
        [8000, 16000],
        [1, 2],
    )
    def test_wav(self, dtype, sample_rate, num_channels):
        """`soundfile_backend.info` can check wav file correctly"""
        duration = 1
        path = self.get_temp_path("data.wav")
        data = get_wav_data(dtype, num_channels, normalize=False, num_frames=duration * sample_rate)
        save_wav(path, data, sample_rate)
        info = soundfile_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == get_bits_per_sample("wav", dtype)
        assert info.encoding == get_encoding("wav", dtype)

    @parameterize([8000, 16000], [1, 2])
    @skipIfFormatNotSupported("FLAC")
    def test_flac(self, sample_rate, num_channels):
        """`soundfile_backend.info` can check flac file correctly"""
        duration = 1
        num_frames = sample_rate * duration
        #data = torch.randn(num_frames, num_channels).numpy()
        data = paddle.randn(shape=[num_frames, num_channels]).numpy()

        path = self.get_temp_path("data.flac")
        soundfile.write(path, data, sample_rate)

        info = soundfile_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == num_frames
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 16
        assert info.encoding == "FLAC"

    #@parameterize([8000, 16000], [1, 2])
    #@skipIfFormatNotSupported("OGG")
    #def test_ogg(self, sample_rate, num_channels):
        #"""`soundfile_backend.info` can check ogg file correctly"""
        #duration = 1
        #num_frames = sample_rate * duration
        ##data = torch.randn(num_frames, num_channels).numpy()
        #data = paddle.randn(shape=[num_frames, num_channels]).numpy()
        #print(len(data))
        #path = self.get_temp_path("data.ogg")
        #soundfile.write(path, data, sample_rate)

        #info = soundfile_backend.info(path)
        #print(info)
        #assert info.sample_rate == sample_rate
        #print("info")
        #print(info.num_frames)
        #print("jiji")
        #print(sample_rate*duration)
        ##assert info.num_frames == sample_rate * duration
        #assert info.num_channels == num_channels
        #assert info.bits_per_sample == 0
        #assert info.encoding == "VORBIS"

    @nested_params(
        [8000, 16000],
        [1, 2],
        [("PCM_24", 24), ("PCM_32", 32)],
    )
    @skipIfFormatNotSupported("NIST")
    def test_sphere(self, sample_rate, num_channels, subtype_and_bit_depth):
        """`soundfile_backend.info` can check sph file correctly"""
        duration = 1
        num_frames = sample_rate * duration
        #data = torch.randn(num_frames, num_channels).numpy()
        data = paddle.randn(shape=[num_frames, num_channels]).numpy()
        path = self.get_temp_path("data.nist")
        subtype, bits_per_sample = subtype_and_bit_depth
        soundfile.write(path, data, sample_rate, subtype=subtype)

        info = soundfile_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == bits_per_sample
        assert info.encoding == "PCM_S"

    def test_unknown_subtype_warning(self):
        """soundfile_backend.info issues a warning when the subtype is unknown

        This will happen if a new subtype is supported in SoundFile: the _SUBTYPE_TO_BITS_PER_SAMPLE
        dict should be updated.
        """

        def _mock_info_func(_):
            class MockSoundFileInfo:
                samplerate = 8000
                frames = 356
                channels = 2
                subtype = "UNSEEN_SUBTYPE"
                format = "UNKNOWN"

            return MockSoundFileInfo()

        with patch("soundfile.info", _mock_info_func):
            with warnings.catch_warnings(record=True) as w:
                info = soundfile_backend.info("foo")
                assert len(w) == 1
                assert "UNSEEN_SUBTYPE subtype is unknown to PaddleAudio" in str(w[-1].message)
                assert info.bits_per_sample == 0


class TestFileObject(TempDirMixin, unittest.TestCase):
    def _test_fileobj(self, ext, subtype, bits_per_sample):
        """Query audio via file-like object works"""
        duration = 2
        sample_rate = 16000
        num_channels = 2
        num_frames = sample_rate * duration
        path = self.get_temp_path(f"test.{ext}")

        #data = torch.randn(num_frames, num_channels).numpy()
        data = paddle.randn(shape=[num_frames, num_channels]).numpy()
        soundfile.write(path, data, sample_rate, subtype=subtype)

        with open(path, "rb") as fileobj:
            info = soundfile_backend.info(fileobj)
        assert info.sample_rate == sample_rate
        assert info.num_frames == num_frames
        assert info.num_channels == num_channels
        assert info.bits_per_sample == bits_per_sample
        assert info.encoding == "FLAC" if ext == "flac" else "PCM_S"

    def test_fileobj_wav(self):
        """Loading audio via file-like object works"""
        self._test_fileobj("wav", "PCM_16", 16)

    @skipIfFormatNotSupported("FLAC")
    def test_fileobj_flac(self):
        """Loading audio via file-like object works"""
        self._test_fileobj("flac", "PCM_16", 16)

    def _test_tarobj(self, ext, subtype, bits_per_sample):
        """Query compressed audio via file-like object works"""
        duration = 2
        sample_rate = 16000
        num_channels = 2
        num_frames = sample_rate * duration
        audio_file = f"test.{ext}"
        audio_path = self.get_temp_path(audio_file)
        archive_path = self.get_temp_path("archive.tar.gz")

        #data = torch.randn(num_frames, num_channels).numpy()
        data = paddle.randn(shape=[num_frames, num_channels]).numpy()
        soundfile.write(audio_path, data, sample_rate, subtype=subtype)

        with tarfile.TarFile(archive_path, "w") as tarobj:
            tarobj.add(audio_path, arcname=audio_file)
        with tarfile.TarFile(archive_path, "r") as tarobj:
            fileobj = tarobj.extractfile(audio_file)
            info = soundfile_backend.info(fileobj)
        assert info.sample_rate == sample_rate
        assert info.num_frames == num_frames
        assert info.num_channels == num_channels
        assert info.bits_per_sample == bits_per_sample
        assert info.encoding == "FLAC" if ext == "flac" else "PCM_S"

    def test_tarobj_wav(self):
        """Query compressed audio via file-like object works"""
        self._test_tarobj("wav", "PCM_16", 16)

    @skipIfFormatNotSupported("FLAC")
    def test_tarobj_flac(self):
        """Query compressed audio via file-like object works"""
        self._test_tarobj("flac", "PCM_16", 16)

if __name__ == '__main__':
    unittest.main()
