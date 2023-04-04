# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle


def test_add_noise(tmpdir, device):
    paddle.device.set_device(device)
    from paddlespeech.vector.io.augment import AddNoise

    test_waveform = paddle.sin(paddle.arange(16000.0,
                                             dtype="float32")).unsqueeze(0)
    test_noise = paddle.cos(paddle.arange(16000.0,
                                          dtype="float32")).unsqueeze(0)
    wav_lens = paddle.ones([1], dtype="float32")

    # Edge cases
    no_noise = AddNoise(mix_prob=0.0)
    assert no_noise(test_waveform, wav_lens).allclose(test_waveform)


def test_speed_perturb(device):
    paddle.device.set_device(device)
    from paddlespeech.vector.io.augment import SpeedPerturb

    test_waveform = paddle.sin(paddle.arange(16000.0,
                                             dtype="float32")).unsqueeze(0)

    # Edge cases
    no_perturb = SpeedPerturb(16000, perturb_prob=0.0)
    assert no_perturb(test_waveform).allclose(test_waveform)
    no_perturb = SpeedPerturb(16000, speeds=[100])
    assert no_perturb(test_waveform).allclose(test_waveform)

    # # Half speed
    half_speed = SpeedPerturb(16000, speeds=[50])
    assert half_speed(test_waveform).allclose(test_waveform[:, ::2], atol=3e-1)


def test_babble(device):
    paddle.device.set_device(device)
    from paddlespeech.vector.io.augment import AddBabble

    test_waveform = paddle.stack((
        paddle.sin(paddle.arange(16000.0, dtype="float32")),
        paddle.cos(paddle.arange(16000.0, dtype="float32")),
    ))
    lengths = paddle.ones([2])

    # Edge cases
    no_babble = AddBabble(mix_prob=0.0)
    assert no_babble(test_waveform, lengths).allclose(test_waveform)
    no_babble = AddBabble(speaker_count=1, snr_low=1000, snr_high=1000)
    assert no_babble(test_waveform, lengths).allclose(test_waveform)

    # One babbler just averages the two speakers
    babble = AddBabble(speaker_count=1).to(device)
    expected = (test_waveform + test_waveform.roll(1, 0)) / 2
    assert babble(test_waveform, lengths).allclose(expected, atol=1e-4)


def test_drop_freq(device):
    paddle.device.set_device(device)
    from paddlespeech.vector.io.augment import DropFreq

    test_waveform = paddle.sin(paddle.arange(16000.0,
                                             dtype="float32")).unsqueeze(0)

    # Edge cases
    no_drop = DropFreq(drop_prob=0.0)
    assert no_drop(test_waveform).allclose(test_waveform)
    no_drop = DropFreq(drop_count_low=0, drop_count_high=0)
    assert no_drop(test_waveform).allclose(test_waveform)

    # Check case where frequency range *does not* include signal frequency
    drop_diff_freq = DropFreq(drop_freq_low=0.5, drop_freq_high=0.9)
    assert drop_diff_freq(test_waveform).allclose(test_waveform, atol=1e-1)

    # Check case where frequency range *does* include signal frequency
    drop_same_freq = DropFreq(drop_freq_low=0.28, drop_freq_high=0.28)
    assert drop_same_freq(test_waveform).allclose(paddle.zeros([1, 16000]),
                                                  atol=4e-1)


def test_drop_chunk(device):
    paddle.device.set_device(device)
    from paddlespeech.vector.io.augment import DropChunk

    test_waveform = paddle.sin(paddle.arange(16000.0,
                                             dtype="float32")).unsqueeze(0)
    lengths = paddle.ones([1])

    # Edge cases
    no_drop = DropChunk(drop_prob=0.0)
    assert no_drop(test_waveform, lengths).allclose(test_waveform)
    no_drop = DropChunk(drop_length_low=0, drop_length_high=0)
    assert no_drop(test_waveform, lengths).allclose(test_waveform)
    no_drop = DropChunk(drop_count_low=0, drop_count_high=0)
    assert no_drop(test_waveform, lengths).allclose(test_waveform)
    no_drop = DropChunk(drop_start=0, drop_end=0)
    assert no_drop(test_waveform, lengths).allclose(test_waveform)

    # Specify all parameters to ensure it is deterministic
    dropper = DropChunk(
        drop_length_low=100,
        drop_length_high=100,
        drop_count_low=1,
        drop_count_high=1,
        drop_start=100,
        drop_end=200,
        noise_factor=0.0,
    )
    expected_waveform = test_waveform.clone()
    expected_waveform[:, 100:200] = 0.0

    assert dropper(test_waveform, lengths).allclose(expected_waveform)

    # Make sure amplitude is similar before and after
    dropper = DropChunk(noise_factor=1.0)
    drop_amplitude = dropper(test_waveform, lengths).abs().mean()
    orig_amplitude = test_waveform.abs().mean()
    assert drop_amplitude.allclose(orig_amplitude, atol=1e-2)
