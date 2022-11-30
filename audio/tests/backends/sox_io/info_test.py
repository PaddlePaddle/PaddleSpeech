import unittest
import itertools
import tarfile
from contextlib import contextmanager

import numpy as np
import paddle
import os
import io

import platform
if platform.system() == "Windows":
    import warnings
    warnings.warn("sox io not support in Windows, please skip test.")
    exit()

from parameterized import parameterized
from common import get_bits_per_sample, get_encoding 


from paddleaudio.backends import sox_io_backend

from common_utils import (
    get_wav_data,
    load_wav,
    save_wav,
    TempDirMixin,
    sox_utils,
)

#code is from:https://github.com/pytorch/audio/blob/main/torchaudio/test/torchaudio_unittest/backend/sox_io/info_test.py

class TestInfo(TempDirMixin, unittest.TestCase):
    @parameterized.expand(
        list(
            itertools.product(
                ["float32", "int32",],
                [8000, 16000],
                [1, 2],
            )
        ),
    )
    def test_wav(self, dtype, sample_rate, num_channels):
        """`sox_io_backend.info` can check wav file correctly"""
        duration = 1
        path = self.get_temp_path("data.wav")
        data = get_wav_data(dtype, num_channels, normalize=False, num_frames=duration * sample_rate)
        save_wav(path, data, sample_rate)
        info = sox_io_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == sox_utils.get_bit_depth(dtype)
        assert info.encoding == get_encoding("wav", dtype)

    @parameterized.expand(
        list(
            itertools.product(
                ["float32", "int32"],
                [8000, 16000],
                [4, 8, 16, 32],
            )
        ),
    )
    def test_wav_multiple_channels(self, dtype, sample_rate, num_channels):
        """`sox_io_backend.info` can check wav file with channels more than 2 correctly"""
        duration = 1
        path = self.get_temp_path("data.wav")
        data = get_wav_data(dtype, num_channels, normalize=False, num_frames=duration * sample_rate)
        save_wav(path, data, sample_rate)
        info = sox_io_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == sox_utils.get_bit_depth(dtype)

    def test_ulaw(self):
        """`sox_io_backend.info` can check ulaw file correctly"""
        duration = 1
        num_channels = 1
        sample_rate = 8000
        path = self.get_temp_path("data.wav")
        sox_utils.gen_audio_file(
            path, sample_rate=sample_rate, num_channels=num_channels, bit_depth=8, encoding="u-law", duration=duration
        )
        info = sox_io_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 8
        assert info.encoding == "ULAW" 

    def test_alaw(self):
        """`sox_io_backend.info` can check alaw file correctly"""
        duration = 1
        num_channels = 1
        sample_rate = 8000
        path = self.get_temp_path("data.wav")
        sox_utils.gen_audio_file(
            path, sample_rate=sample_rate, num_channels=num_channels, bit_depth=8, encoding="a-law", duration=duration
        )
        info = sox_io_backend.info(path)
        assert info.sample_rate == sample_rate
        assert info.num_frames == sample_rate * duration
        assert info.num_channels == num_channels
        assert info.bits_per_sample == 8
        assert info.encoding == "ALAW"

#class TestInfoOpus(unittest.TestCase):
    #@parameterized.expand(
        #list(
            #itertools.product(
                #["96k"],
                #[1, 2],
                #[0, 5, 10],
            #)
        #),
    #)
    #def test_opus(self, bitrate, num_channels, compression_level):
        #"""`sox_io_backend.info` can check opus file correcty"""
        #path = data_utils.get_asset_path("io", f"{bitrate}_{compression_level}_{num_channels}ch.opus")
        #info = sox_io_backend.info(path)
        #assert info.sample_rate == 48000
        #assert info.num_frames == 32768
        #assert info.num_channels == num_channels
        #assert info.bits_per_sample == 0  # bit_per_sample is irrelevant for compressed formats
        #assert info.encoding == "OPUS"

class FileObjTestBase(TempDirMixin):
    def _gen_file(self, ext, dtype, sample_rate, num_channels, num_frames, *, comments=None):
        path = self.get_temp_path(f"test.{ext}")
        bit_depth = sox_utils.get_bit_depth(dtype)
        duration = num_frames / sample_rate
        comment_file = self._gen_comment_file(comments) if comments else None

        sox_utils.gen_audio_file(
            path,
            sample_rate,
            num_channels=num_channels,
            encoding=sox_utils.get_encoding(dtype),
            bit_depth=bit_depth,
            duration=duration,
            comment_file=comment_file,
        )
        return path

    def _gen_comment_file(self, comments):
        comment_path = self.get_temp_path("comment.txt")
        with open(comment_path, "w") as file_:
            file_.writelines(comments)
        return comment_path

class Unseekable:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, n):
        return self.fileobj.read(n)

class TestFileObject(FileObjTestBase, unittest.TestCase):
    def _query_fileobj(self, ext, dtype, sample_rate, num_channels, num_frames, *, comments=None):
        path = self._gen_file(ext, dtype, sample_rate, num_channels, num_frames, comments=comments)
        format_ = ext if ext in ["mp3"] else None
        with open(path, "rb") as fileobj:
            return sox_io_backend.info(fileobj, format_)

    def _query_bytesio(self, ext, dtype, sample_rate, num_channels, num_frames):
        path = self._gen_file(ext, dtype, sample_rate, num_channels, num_frames)
        format_ = ext if ext in ["mp3"] else None
        with open(path, "rb") as file_:
            fileobj = io.BytesIO(file_.read())
        return sox_io_backend.info(fileobj, format_)

    def _query_tarfile(self, ext, dtype, sample_rate, num_channels, num_frames):
        audio_path = self._gen_file(ext, dtype, sample_rate, num_channels, num_frames)
        audio_file = os.path.basename(audio_path)
        archive_path = self.get_temp_path("archive.tar.gz")
        with tarfile.TarFile(archive_path, "w") as tarobj:
            tarobj.add(audio_path, arcname=audio_file)
        format_ = ext if ext in ["mp3"] else None
        with tarfile.TarFile(archive_path, "r") as tarobj:
            fileobj = tarobj.extractfile(audio_file)
            return sox_io_backend.info(fileobj, format_)

    @contextmanager
    def _set_buffer_size(self, buffer_size):
        try:
            original_buffer_size = get_buffer_size()
            set_buffer_size(buffer_size)
            yield
        finally:
            set_buffer_size(original_buffer_size)

    @parameterized.expand(
        [
            ("wav", "float32"),
            ("wav", "int32"),
            ("wav", "int16"),
            ("wav", "uint8"),
        ]
    )
    def test_fileobj(self, ext, dtype):
        """Querying audio via file object works"""
        sample_rate = 16000
        num_frames = 3 * sample_rate
        num_channels = 2
        sinfo = self._query_fileobj(ext, dtype, sample_rate, num_channels, num_frames)

        bits_per_sample = get_bits_per_sample(ext, dtype)
        num_frames = 0 if ext in ["mp3", "vorbis"] else num_frames

        assert sinfo.sample_rate == sample_rate
        assert sinfo.num_channels == num_channels
        assert sinfo.num_frames == num_frames
        assert sinfo.bits_per_sample == bits_per_sample
        assert sinfo.encoding == get_encoding(ext, dtype)

    @parameterized.expand(
        [
            ("wav", "float32"),
            ("wav", "int32"),
            ("wav", "int16"),
            ("wav", "uint8"),
        ]
    )
    def test_bytesio(self, ext, dtype):
        """Querying audio via ByteIO object works for small data"""
        sample_rate = 16000
        num_frames = 3 * sample_rate
        num_channels = 2
        sinfo = self._query_bytesio(ext, dtype, sample_rate, num_channels, num_frames)

        bits_per_sample = get_bits_per_sample(ext, dtype)
        num_frames = 0 if ext in ["mp3", "vorbis"] else num_frames

        assert sinfo.sample_rate == sample_rate
        assert sinfo.num_channels == num_channels
        assert sinfo.num_frames == num_frames
        assert sinfo.bits_per_sample == bits_per_sample
        assert sinfo.encoding == get_encoding(ext, dtype)

    @parameterized.expand(
        [
            ("wav", "float32"),
            ("wav", "int32"),
            ("wav", "int16"),
            ("wav", "uint8"),
        ]
    )
    def test_bytesio_tiny(self, ext, dtype):
        """Querying audio via ByteIO object works for small data"""
        sample_rate = 8000
        num_frames = 4
        num_channels = 2
        sinfo = self._query_bytesio(ext, dtype, sample_rate, num_channels, num_frames)

        bits_per_sample = get_bits_per_sample(ext, dtype)
        num_frames = 0 if ext in ["mp3", "vorbis"] else num_frames

        assert sinfo.sample_rate == sample_rate
        assert sinfo.num_channels == num_channels
        assert sinfo.num_frames == num_frames
        assert sinfo.bits_per_sample == bits_per_sample
        assert sinfo.encoding == get_encoding(ext, dtype)

    @parameterized.expand(
        [
            ("wav", "float32"),
            ("wav", "int32"),
            ("wav", "int16"),
            ("wav", "uint8"),
            ("flac", "float32"),
            ("vorbis", "float32"),
            ("amb", "int16"),
        ]
    )
    def test_tarfile(self, ext, dtype):
        """Querying compressed audio via file-like object works"""
        sample_rate = 16000
        num_frames = 3.0 * sample_rate
        num_channels = 2
        sinfo = self._query_tarfile(ext, dtype, sample_rate, num_channels, num_frames)

        bits_per_sample = get_bits_per_sample(ext, dtype)
        num_frames = 0 if ext in ["vorbis"] else num_frames

        assert sinfo.sample_rate == sample_rate
        assert sinfo.num_channels == num_channels
        assert sinfo.num_frames == num_frames
        assert sinfo.bits_per_sample == bits_per_sample
        assert sinfo.encoding == get_encoding(ext, dtype)



if __name__ == '__main__':
    unittest.main()
