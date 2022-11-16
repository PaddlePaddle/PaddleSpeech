import io
import os
import unittest

import numpy as np
import paddle
from parameterized import parameterized
from paddleaudio.backends import sox_io_backend

from tests.unit.common_utils import (
    get_wav_data,
    load_wav,
    save_wav,
    nested_params,
    TempDirMixin,
    sox_utils
)

#code is from:https://github.com/pytorch/audio/blob/main/torchaudio/test/torchaudio_unittest/backend/sox_io/save_test.py

def _get_sox_encoding(encoding):
    encodings = {
        "PCM_F": "floating-point",
        "PCM_S": "signed-integer",
        "PCM_U": "unsigned-integer",
        "ULAW": "u-law",
        "ALAW": "a-law",
    }
    return encodings.get(encoding)

class TestSaveBase(TempDirMixin):
    def assert_save_consistency(
        self,
        format: str,
        *,
        compression: float = None,
        encoding: str = None,
        bits_per_sample: int = None,
        sample_rate: float = 8000,
        num_channels: int = 2,
        num_frames: float = 3 * 8000,
        src_dtype: str = "int32",
        test_mode: str = "path",
    ):
        """`save` function produces file that is comparable with `sox` command

        To compare that the file produced by `save` function agains the file produced by
        the equivalent `sox` command, we need to load both files.
        But there are many formats that cannot be opened with common Python modules (like
        SciPy).
        So we use `sox` command to prepare the original data and convert the saved files
        into a format that SciPy can read (PCM wav).
        The following diagram illustrates this process. The difference is 2.1. and 3.1.

        This assumes that
         - loading data with SciPy preserves the data well.
         - converting the resulting files into WAV format with `sox` preserve the data well.

                          x
                          | 1. Generate source wav file with SciPy
                          |
                          v
          -------------- wav ----------------
         |                                   |
         | 2.1. load with scipy              | 3.1. Convert to the target
         |   then save it into the target    |      format depth with sox
         |   format with paddleaudio          |
         v                                   v
        target format                       target format
         |                                   |
         | 2.2. Convert to wav with sox      | 3.2. Convert to wav with sox
         |                                   |
         v                                   v
        wav                                 wav
         |                                   |
         | 2.3. load with scipy              | 3.3. load with scipy
         |                                   |
         v                                   v
        tensor -------> compare <--------- tensor

        """
        cmp_encoding = "floating-point"
        cmp_bit_depth = 32

        src_path = self.get_temp_path("1.source.wav")
        tgt_path = self.get_temp_path(f"2.1.paddleaudio.{format}")
        tst_path = self.get_temp_path("2.2.result.wav")
        sox_path = self.get_temp_path(f"3.1.sox.{format}")
        ref_path = self.get_temp_path("3.2.ref.wav")

        # 1. Generate original wav
        data = get_wav_data(src_dtype, num_channels, normalize=False, num_frames=num_frames)
        save_wav(src_path, data, sample_rate)

        # 2.1. Convert the original wav to target format with paddleaudio
        data = load_wav(src_path, normalize=False)[0]
        if test_mode == "path":
            sox_io_backend.save(
                tgt_path, data, sample_rate, compression=compression, encoding=encoding, bits_per_sample=bits_per_sample
            )
        elif test_mode == "fileobj":
            with open(tgt_path, "bw") as file_:
                sox_io_backend.save(
                    file_,
                    data,
                    sample_rate,
                    format=format,
                    compression=compression,
                    encoding=encoding,
                    bits_per_sample=bits_per_sample,
                )
        elif test_mode == "bytesio":
            file_ = io.BytesIO()
            sox_io_backend.save(
                file_,
                data,
                sample_rate,
                format=format,
                compression=compression,
                encoding=encoding,
                bits_per_sample=bits_per_sample,
            )
            file_.seek(0)
            with open(tgt_path, "bw") as f:
                f.write(file_.read())
        else:
            raise ValueError(f"Unexpected test mode: {test_mode}")
        # 2.2. Convert the target format to wav with sox
        sox_utils.convert_audio_file(tgt_path, tst_path, encoding=cmp_encoding, bit_depth=cmp_bit_depth)
        # 2.3. Load with SciPy
        found = load_wav(tst_path, normalize=False)[0]

        # 3.1. Convert the original wav to target format with sox
        sox_encoding = _get_sox_encoding(encoding)
        sox_utils.convert_audio_file(
            src_path, sox_path, compression=compression, encoding=sox_encoding, bit_depth=bits_per_sample
        )
        # 3.2. Convert the target format to wav with sox
        sox_utils.convert_audio_file(sox_path, ref_path, encoding=cmp_encoding, bit_depth=cmp_bit_depth)
        # 3.3. Load with SciPy
        expected = load_wav(ref_path, normalize=False)[0]

        np.testing.assert_array_almost_equal(found, expected)

class TestSave(TestSaveBase, unittest.TestCase):
    @nested_params(
        ["path",],
        [
            ("PCM_U", 8),
            ("PCM_S", 16),
            ("PCM_S", 32),
            ("PCM_F", 32),
            ("PCM_F", 64),
            ("ULAW", 8),
            ("ALAW", 8),
        ],
    )
    def test_save_wav(self, test_mode, enc_params):
        encoding, bits_per_sample = enc_params
        self.assert_save_consistency("wav", encoding=encoding, bits_per_sample=bits_per_sample, test_mode=test_mode)

    @nested_params(
        ["path", ],
        [
            ("float32",),
            ("int32",),
        ],
    )
    def test_save_wav_dtype(self, test_mode, params):
        (dtype,) = params
        self.assert_save_consistency("wav", src_dtype=dtype, test_mode=test_mode)


if __name__ == '__main__':
    unittest.main()
