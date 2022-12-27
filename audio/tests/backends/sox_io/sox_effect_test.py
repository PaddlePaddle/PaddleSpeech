#code is from: https://github.com/pytorch/audio/blob/main/test/torchaudio_unittest/sox_effect/sox_effect_test.py
import io
import itertools
import platform
import tarfile
import unittest
from pathlib import Path

import numpy as np
if platform.system() == "Windows":
    import warnings
    warnings.warn("sox io not support in Windows, please skip test.")
    exit()

from parameterized import parameterized
from paddleaudio import sox_effects
from common_utils import (get_sinusoid, get_wav_data, load_wav, save_wav,
                          sox_utils, TempDirMixin, load_effects_params)


class TestSoxEffects(unittest.TestCase):
    def test_init(self):
        """Calling init_sox_effects multiple times does not crush"""
        for _ in range(3):
            sox_effects.init_sox_effects()


class TestSoxEffectsTensor(TempDirMixin, unittest.TestCase):
    """Test suite for `apply_effects_tensor` function"""

    @parameterized.expand(
        list(
            itertools.product(["float32", "int32"], [8000, 16000], [1, 2, 4, 8],
                              [True, False])), )
    def test_apply_no_effect(self, dtype, sample_rate, num_channels,
                             channels_first):
        """`apply_effects_tensor` without effects should return identical data as input"""
        original = get_wav_data(
            dtype, num_channels, channels_first=channels_first)
        expected = original.clone()

        found, output_sample_rate = sox_effects.apply_effects_tensor(
            expected, sample_rate, [], channels_first)

        assert (output_sample_rate == sample_rate)
        # SoxEffect should not alter the input Tensor object
        #self.assertEqual(original, expected)
        np.testing.assert_array_almost_equal(original.numpy(), expected.numpy())

        # SoxEffect should not return the same Tensor object
        assert expected is not found
        # Returned Tensor should equal to the input Tensor
        #self.assertEqual(expected, found)
        np.testing.assert_array_almost_equal(expected.numpy(), found.numpy())

    @parameterized.expand(
        load_effects_params("sox_effect_test_args.jsonl"),
        name_func=lambda f, i, p: f'{f.__name__}_{i}_{p.args[0]["effects"][0][0]}',
    )
    def test_apply_effects(self, args):
        """`apply_effects_tensor` should return identical data as sox command"""
        effects = args["effects"]
        num_channels = args.get("num_channels", 2)
        input_sr = args.get("input_sample_rate", 8000)
        output_sr = args.get("output_sample_rate")

        input_path = self.get_temp_path("input.wav")
        reference_path = self.get_temp_path("reference.wav")

        original = get_sinusoid(
            frequency=800,
            sample_rate=input_sr,
            n_channels=num_channels,
            dtype="float32")
        save_wav(input_path, original, input_sr)
        sox_utils.run_sox_effect(
            input_path, reference_path, effects, output_sample_rate=output_sr)

        expected, expected_sr = load_wav(reference_path)
        found, sr = sox_effects.apply_effects_tensor(original, input_sr,
                                                     effects)

        assert sr == expected_sr
        #self.assertEqual(expected, found)
        np.testing.assert_array_almost_equal(expected.numpy(), found.numpy())


class TestSoxEffectsFile(TempDirMixin, unittest.TestCase):
    """Test suite for `apply_effects_file` function"""

    @parameterized.expand(
        list(
            itertools.product(
                ["float32", "int32"],
                [8000, 16000],
                [1, 2, 4, 8],
                [False, True], )),
        #name_func=name_func,
    )
    def test_apply_no_effect(self, dtype, sample_rate, num_channels,
                             channels_first):
        """`apply_effects_file` without effects should return identical data as input"""
        path = self.get_temp_path("input.wav")
        expected = get_wav_data(
            dtype, num_channels, channels_first=channels_first)
        save_wav(path, expected, sample_rate, channels_first=channels_first)

        found, output_sample_rate = sox_effects.apply_effects_file(
            path, [], normalize=False, channels_first=channels_first)

        assert output_sample_rate == sample_rate
        #self.assertEqual(expected, found)
        np.testing.assert_array_almost_equal(expected.numpy(), found.numpy())

    @parameterized.expand(
        load_effects_params("sox_effect_test_args.jsonl"),
        #name_func=lambda f, i, p: f'{f.__name__}_{i}_{p.args[0]["effects"][0][0]}',
    )
    def test_apply_effects_str(self, args):
        """`apply_effects_file` should return identical data as sox command"""
        dtype = "int32"
        channels_first = True
        effects = args["effects"]
        num_channels = args.get("num_channels", 2)
        input_sr = args.get("input_sample_rate", 8000)
        output_sr = args.get("output_sample_rate")

        input_path = self.get_temp_path("input.wav")
        reference_path = self.get_temp_path("reference.wav")
        data = get_wav_data(dtype, num_channels, channels_first=channels_first)
        save_wav(input_path, data, input_sr, channels_first=channels_first)
        sox_utils.run_sox_effect(
            input_path, reference_path, effects, output_sample_rate=output_sr)

        expected, expected_sr = load_wav(reference_path)
        found, sr = sox_effects.apply_effects_file(
            input_path, effects, normalize=False, channels_first=channels_first)

        assert sr == expected_sr
        #self.assertEqual(found, expected)
        np.testing.assert_array_almost_equal(expected.numpy(), found.numpy())

    def test_apply_effects_path(self):
        """`apply_effects_file` should return identical data as sox command when file path is given as a Path Object"""
        dtype = "int32"
        channels_first = True
        effects = [["hilbert"]]
        num_channels = 2
        input_sr = 8000
        output_sr = 8000

        input_path = self.get_temp_path("input.wav")
        reference_path = self.get_temp_path("reference.wav")
        data = get_wav_data(dtype, num_channels, channels_first=channels_first)
        save_wav(input_path, data, input_sr, channels_first=channels_first)
        sox_utils.run_sox_effect(
            input_path, reference_path, effects, output_sample_rate=output_sr)

        expected, expected_sr = load_wav(reference_path)
        found, sr = sox_effects.apply_effects_file(
            Path(input_path),
            effects,
            normalize=False,
            channels_first=channels_first)

        assert sr == expected_sr
        #self.assertEqual(found, expected)
        np.testing.assert_array_almost_equal(expected.numpy(), found.numpy())


class TestFileFormats(TempDirMixin, unittest.TestCase):
    """`apply_effects_file` gives the same result as sox on various file formats"""

    @parameterized.expand(
        list(itertools.product(
            ["float32", "int32"],
            [8000, 16000],
            [1, 2], )),
        #name_func=lambda f, _, p: f'{f.__name__}_{"_".join(str(arg) for arg in p.args)}',
    )
    def test_wav(self, dtype, sample_rate, num_channels):
        """`apply_effects_file` works on various wav format"""
        channels_first = True
        effects = [["band", "300", "10"]]

        input_path = self.get_temp_path("input.wav")
        reference_path = self.get_temp_path("reference.wav")
        data = get_wav_data(dtype, num_channels, channels_first=channels_first)
        save_wav(input_path, data, sample_rate, channels_first=channels_first)
        sox_utils.run_sox_effect(input_path, reference_path, effects)

        expected, expected_sr = load_wav(reference_path)
        found, sr = sox_effects.apply_effects_file(
            input_path, effects, normalize=False, channels_first=channels_first)

        assert sr == expected_sr
        #self.assertEqual(found, expected)
        np.testing.assert_array_almost_equal(found.numpy(), expected.numpy())

    #not support now
    #@parameterized.expand(
    #list(
    #itertools.product(
    #[8000, 16000],
    #[1, 2],
    #)
    #),
    ##name_func=lambda f, _, p: f'{f.__name__}_{"_".join(str(arg) for arg in p.args)}',
    #)
    #def test_flac(self, sample_rate, num_channels):
    #"""`apply_effects_file` works on various flac format"""
    #channels_first = True
    #effects = [["band", "300", "10"]]

    #input_path = self.get_temp_path("input.flac")
    #reference_path = self.get_temp_path("reference.wav")
    #sox_utils.gen_audio_file(input_path, sample_rate, num_channels)
    #sox_utils.run_sox_effect(input_path, reference_path, effects, output_bitdepth=32)

    #expected, expected_sr = load_wav(reference_path)
    #found, sr = sox_effects.apply_effects_file(input_path, effects, channels_first=channels_first)
    #save_wav(self.get_temp_path("result.wav"), found, sr, channels_first=channels_first)

    #assert sr == expected_sr
    ##self.assertEqual(found, expected)
    #np.testing.assert_array_almost_equal(found.numpy(), expected.numpy())

    #@parameterized.expand(
    #list(
    #itertools.product(
    #[8000, 16000],
    #[1, 2],
    #)
    #),
    ##name_func=lambda f, _, p: f'{f.__name__}_{"_".join(str(arg) for arg in p.args)}',
    #)
    #def test_vorbis(self, sample_rate, num_channels):
    #"""`apply_effects_file` works on various vorbis format"""
    #channels_first = True
    #effects = [["band", "300", "10"]]

    #input_path = self.get_temp_path("input.vorbis")
    #reference_path = self.get_temp_path("reference.wav")
    #sox_utils.gen_audio_file(input_path, sample_rate, num_channels)
    #sox_utils.run_sox_effect(input_path, reference_path, effects, output_bitdepth=32)

    #expected, expected_sr = load_wav(reference_path)
    #found, sr = sox_effects.apply_effects_file(input_path, effects, channels_first=channels_first)
    #save_wav(self.get_temp_path("result.wav"), found, sr, channels_first=channels_first)

    #assert sr == expected_sr
    ##self.assertEqual(found, expected)
    #np.testing.assert_array_almost_equal(found.numpy(), expected.numpy())


    #@skipIfNoExec("sox")
    #@skipIfNoSox
class TestFileObject(TempDirMixin, unittest.TestCase):
    @parameterized.expand([
        ("wav", None),
    ])
    def test_fileobj(self, ext, compression):
        """Applying effects via file object works"""
        sample_rate = 16000
        channels_first = True
        effects = [["band", "300", "10"]]
        input_path = self.get_temp_path(f"input.{ext}")
        reference_path = self.get_temp_path("reference.wav")

        #sox_utils.gen_audio_file(input_path, sample_rate, num_channels=2, compression=compression)
        data = get_wav_data("int32", 2, channels_first=channels_first)
        save_wav(input_path, data, sample_rate, channels_first=channels_first)

        sox_utils.run_sox_effect(
            input_path, reference_path, effects, output_bitdepth=32)
        expected, expected_sr = load_wav(reference_path)

        with open(input_path, "rb") as fileobj:
            found, sr = sox_effects.apply_effects_file(
                fileobj, effects, channels_first=channels_first)
        save_wav(
            self.get_temp_path("result.wav"),
            found,
            sr,
            channels_first=channels_first)
        assert sr == expected_sr
        #self.assertEqual(found, expected)
        np.testing.assert_array_almost_equal(found.numpy(), expected.numpy())

    @parameterized.expand([
        ("wav", None),
    ])
    def test_bytesio(self, ext, compression):
        """Applying effects via BytesIO object works"""
        sample_rate = 16000
        channels_first = True
        effects = [["band", "300", "10"]]
        input_path = self.get_temp_path(f"input.{ext}")
        reference_path = self.get_temp_path("reference.wav")

        #sox_utils.gen_audio_file(input_path, sample_rate, num_channels=2, compression=compression)
        data = get_wav_data("int32", 2, channels_first=channels_first)
        save_wav(input_path, data, sample_rate, channels_first=channels_first)
        sox_utils.run_sox_effect(
            input_path, reference_path, effects, output_bitdepth=32)
        expected, expected_sr = load_wav(reference_path)

        with open(input_path, "rb") as file_:
            fileobj = io.BytesIO(file_.read())
        found, sr = sox_effects.apply_effects_file(
            fileobj, effects, channels_first=channels_first)
        save_wav(
            self.get_temp_path("result.wav"),
            found,
            sr,
            channels_first=channels_first)
        assert sr == expected_sr
        #self.assertEqual(found, expected)
        print("found")
        print(found)
        print("expected")
        print(expected)
        np.testing.assert_array_almost_equal(found.numpy(), expected.numpy())

    @parameterized.expand([
        ("wav", None),
    ])
    def test_tarfile(self, ext, compression):
        """Applying effects to compressed audio via file-like file works"""
        sample_rate = 16000
        channels_first = True
        effects = [["band", "300", "10"]]
        audio_file = f"input.{ext}"

        input_path = self.get_temp_path(audio_file)
        reference_path = self.get_temp_path("reference.wav")
        archive_path = self.get_temp_path("archive.tar.gz")
        data = get_wav_data("int32", 2, channels_first=channels_first)
        save_wav(input_path, data, sample_rate, channels_first=channels_first)

        #       sox_utils.gen_audio_file(input_path, sample_rate, num_channels=2, compression=compression)
        sox_utils.run_sox_effect(
            input_path, reference_path, effects, output_bitdepth=32)

        expected, expected_sr = load_wav(reference_path)

        with tarfile.TarFile(archive_path, "w") as tarobj:
            tarobj.add(input_path, arcname=audio_file)
        with tarfile.TarFile(archive_path, "r") as tarobj:
            fileobj = tarobj.extractfile(audio_file)
            found, sr = sox_effects.apply_effects_file(
                fileobj, effects, channels_first=channels_first)
        save_wav(
            self.get_temp_path("result.wav"),
            found,
            sr,
            channels_first=channels_first)
        assert sr == expected_sr
        #self.assertEqual(found, expected)
        np.testing.assert_array_almost_equal(found.numpy(), expected.numpy())


if __name__ == '__main__':
    unittest.main()
