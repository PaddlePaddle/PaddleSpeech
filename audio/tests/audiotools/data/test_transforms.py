# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/tests/data/test_transforms.py)
import inspect
import sys
import warnings
from pathlib import Path

import numpy as np
import paddle
import pytest

from audio import audiotools
from audio.audiotools import AudioSignal
from audio.audiotools import util
from audio.audiotools.data import transforms as tfm
from audio.audiotools.data.datasets import AudioDataset
from paddlespeech.vector.training.seeding import seed_everything

non_deterministic_transforms = ["TimeNoise", "FrequencyNoise"]
transforms_to_test = []
for x in dir(tfm):
    if hasattr(getattr(tfm, x), "transform"):
        if x not in [
                "Compose",
                "Choose",
                "Repeat",
                "RepeatUpTo",
                # The above 4 transforms are currently excluded from testing at 1e-4 precision due to potential accuracy issues
                "BackgroundNoise",
                "Equalizer",
                "FrequencyNoise",
                "RoomImpulseResponse"
        ]:
            transforms_to_test.append(x)


def _compare_transform(transform_name, signal):
    regression_data = Path(f"regression/transforms/{transform_name}.wav")
    regression_data.parent.mkdir(exist_ok=True, parents=True)

    if regression_data.exists():
        regression_signal = AudioSignal(regression_data)

        assert paddle.allclose(
            signal.audio_data, regression_signal.audio_data, atol=1e-4)
    else:
        signal.write(regression_data)


@pytest.mark.parametrize("transform_name", transforms_to_test)
def test_transform(transform_name):
    seed = 0
    seed_everything(seed)
    transform_cls = getattr(tfm, transform_name)

    kwargs = {}
    if transform_name == "BackgroundNoise":
        kwargs["sources"] = ["./audio/noises.csv"]
    if transform_name == "RoomImpulseResponse":
        kwargs["sources"] = ["./audio/irs.csv"]
    if transform_name == "CrossTalk":
        kwargs["sources"] = ["./audio/spk.csv"]

    audio_path = "./audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)
    signal.metadata["loudness"] = AudioSignal(
        audio_path).ffmpeg_loudness().item()
    transform = transform_cls(prob=1.0, **kwargs)

    kwargs = transform.instantiate(seed, signal)
    for k in kwargs[transform_name]:
        assert k in transform.keys

    output = transform(signal, **kwargs)
    assert isinstance(output, AudioSignal)

    _compare_transform(transform_name, output)

    if transform_name in non_deterministic_transforms:
        return

    # Test that if you make a batch of signals and call it,
    # the first item in the batch is still the same as above.
    batch_size = 4
    signal = AudioSignal(audio_path, offset=10, duration=2)
    signal_batch = AudioSignal.batch(
        [signal.clone() for _ in range(batch_size)])
    signal_batch.metadata["loudness"] = AudioSignal(
        audio_path).ffmpeg_loudness().item()

    states = [seed + idx for idx in list(range(batch_size))]
    kwargs = transform.batch_instantiate(states, signal_batch)
    batch_output = transform(signal_batch, **kwargs)

    assert batch_output[0] == output

    ## Test that you can apply transform with the same args twice.
    signal = AudioSignal(audio_path, offset=10, duration=2)
    signal.metadata["loudness"] = AudioSignal(
        audio_path).ffmpeg_loudness().item()
    kwargs = transform.instantiate(seed, signal)
    output_a = transform(signal.clone(), **kwargs)
    output_b = transform(signal.clone(), **kwargs)

    assert output_a == output_b


def test_compose_basic():
    seed = 0

    audio_path = "./audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)
    transform = tfm.Compose(
        [
            tfm.RoomImpulseResponse(sources=["./audio/irs.csv"]),
            tfm.BackgroundNoise(sources=["./audio/noises.csv"]),
        ], )

    kwargs = transform.instantiate(seed, signal)
    output = transform(signal, **kwargs)

    # Due to precision issues with RoomImpulseResponse and BackgroundNoise used in the Compose test,
    # we only perform logical testing for Compose and skip precision testing of the final output
    # _compare_transform("Compose", output)

    assert isinstance(transform[0], tfm.RoomImpulseResponse)
    assert isinstance(transform[1], tfm.BackgroundNoise)
    assert len(transform) == 2

    # Make sure __iter__ works
    for _tfm in transform:
        pass


class MulTransform(tfm.BaseTransform):
    def __init__(self, num, name=None):
        self.num = num
        super().__init__(name=name, keys=["num"])

    def _transform(self, signal, num):

        if not num.dim():
            num = num.unsqueeze(axis=0)

        signal.audio_data = signal.audio_data * num[:, None, None]
        return signal

    def _instantiate(self, state):
        return {"num": self.num}


def test_compose_with_duplicate_transforms():
    muls = [0.5, 0.25, 0.125]
    transform = tfm.Compose([MulTransform(x) for x in muls])
    full_mul = np.prod(muls)

    kwargs = transform.instantiate(0)
    audio_path = "./audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)

    output = transform(signal.clone(), **kwargs)
    expected_output = signal.audio_data * full_mul

    assert paddle.allclose(output.audio_data, expected_output)


def test_nested_compose():
    muls = [0.5, 0.25, 0.125]
    transform = tfm.Compose([
        MulTransform(muls[0]),
        tfm.Compose(
            [MulTransform(muls[1]), tfm.Compose([MulTransform(muls[2])])]),
    ])
    full_mul = np.prod(muls)

    kwargs = transform.instantiate(0)
    audio_path = "./audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)

    output = transform(signal.clone(), **kwargs)
    expected_output = signal.audio_data * full_mul

    assert paddle.allclose(output.audio_data, expected_output)


def test_compose_filtering():
    muls = [0.5, 0.25, 0.125]
    transform = tfm.Compose([MulTransform(x, name=str(x)) for x in muls])

    kwargs = transform.instantiate(0)
    audio_path = "./audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)

    for s in range(len(muls)):
        for _ in range(10):
            _muls = np.random.choice(muls, size=s, replace=False).tolist()
            full_mul = np.prod(_muls)
            with transform.filter(*[str(x) for x in _muls]):
                output = transform(signal.clone(), **kwargs)

            expected_output = signal.audio_data * full_mul
            assert paddle.allclose(output.audio_data, expected_output)


def test_sequential_compose():
    muls = [0.5, 0.25, 0.125]
    transform = tfm.Compose([
        tfm.Compose([MulTransform(muls[0])]),
        tfm.Compose([MulTransform(muls[1]), MulTransform(muls[2])]),
    ])
    full_mul = np.prod(muls)

    kwargs = transform.instantiate(0)
    audio_path = "./audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)

    output = transform(signal.clone(), **kwargs)
    expected_output = signal.audio_data * full_mul

    assert paddle.allclose(output.audio_data, expected_output)


def test_choose_basic():
    seed = 0
    audio_path = "./audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)
    transform = tfm.Choose([
        tfm.RoomImpulseResponse(sources=["./audio/irs.csv"]),
        tfm.BackgroundNoise(sources=["./audio/noises.csv"]),
    ])

    kwargs = transform.instantiate(seed, signal)
    output = transform(signal.clone(), **kwargs)

    # Due to precision issues with RoomImpulseResponse and BackgroundNoise used in the Choose test,
    # we only perform logical testing for Choose and skip precision testing of the final output
    # _compare_transform("Choose", output)

    transform = tfm.Choose([
        MulTransform(0.0),
        MulTransform(2.0),
    ])
    targets = [signal.clone() * 0.0, signal.clone() * 2.0]

    for seed in range(10):
        kwargs = transform.instantiate(seed, signal)
        output = transform(signal.clone(), **kwargs)

        assert any([output == target for target in targets])

    # Test that if you make a batch of signals and call it,
    # the first item in the batch is still the same as above.
    batch_size = 4
    signal = AudioSignal(audio_path, offset=10, duration=2)
    signal_batch = AudioSignal.batch(
        [signal.clone() for _ in range(batch_size)])

    states = [seed + idx for idx in list(range(batch_size))]
    kwargs = transform.batch_instantiate(states, signal_batch)
    batch_output = transform(signal_batch, **kwargs)

    for nb in range(batch_size):
        assert batch_output[nb] in targets


def test_choose_weighted():
    seed = 0
    audio_path = "./audio/spk/f10_script4_produced.wav"
    transform = tfm.Choose(
        [
            MulTransform(0.0),
            MulTransform(2.0),
        ],
        weights=[0.0, 1.0], )

    # Test that if you make a batch of signals and call it,
    # the first item in the batch is still the same as above.
    batch_size = 4
    signal = AudioSignal(audio_path, offset=10, duration=2)
    signal_batch = AudioSignal.batch(
        [signal.clone() for _ in range(batch_size)])

    targets = [signal.clone() * 0.0, signal.clone() * 2.0]

    states = [seed + idx for idx in list(range(batch_size))]
    kwargs = transform.batch_instantiate(states, signal_batch)
    batch_output = transform(signal_batch, **kwargs)

    for nb in range(batch_size):
        assert batch_output[nb] == targets[1]


def test_choose_with_compose():
    audio_path = "./audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)

    transform = tfm.Choose([
        tfm.Compose([MulTransform(0.0)]),
        tfm.Compose([MulTransform(2.0)]),
    ])

    targets = [signal.clone() * 0.0, signal.clone() * 2.0]

    for seed in range(10):
        kwargs = transform.instantiate(seed, signal)
        output = transform(signal, **kwargs)

        assert output in targets


def test_repeat():
    seed = 0
    audio_path = "./audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)

    kwargs = {}
    kwargs["transform"] = tfm.Compose(
        tfm.FrequencyMask(),
        tfm.TimeMask(), )
    kwargs["n_repeat"] = 5

    transform = tfm.Repeat(**kwargs)
    kwargs = transform.instantiate(seed, signal)
    output = transform(signal.clone(), **kwargs)

    _compare_transform("Repeat", output)

    kwargs = {}
    kwargs["transform"] = tfm.Compose(
        tfm.FrequencyMask(),
        tfm.TimeMask(), )
    kwargs["max_repeat"] = 10

    transform = tfm.RepeatUpTo(**kwargs)
    kwargs = transform.instantiate(seed, signal)
    output = transform(signal.clone(), **kwargs)

    _compare_transform("RepeatUpTo", output)

    # Make sure repeat does what it says
    transform = tfm.Repeat(MulTransform(0.5), n_repeat=3)
    kwargs = transform.instantiate(seed, signal)
    signal = AudioSignal(paddle.randn([1, 1, 100]).clip(1e-5), 44100)
    output = transform(signal.clone(), **kwargs)

    scale = (output.audio_data / signal.audio_data).mean()
    assert scale == (0.5**3)


class DummyData(paddle.io.Dataset):
    def __init__(self, audio_path):
        super().__init__()

        self.audio_path = audio_path
        self.length = 100
        self.transform = tfm.Silence(prob=0.5)

    def __getitem__(self, idx):
        state = util.random_state(idx)
        signal = AudioSignal.salient_excerpt(
            self.audio_path, state=state, duration=1.0).resample(44100)

        item = self.transform.instantiate(state, signal=signal)
        item["signal"] = signal

        return item

    def __len__(self):
        return self.length


def test_masking():
    dataset = DummyData("./audio/spk/f10_script4_produced.wav")
    dataloader = paddle.io.DataLoader(
        dataset,
        batch_size=16,
        num_workers=0,
        collate_fn=util.collate, )
    for batch in dataloader:
        signal = batch.pop("signal")
        original = signal.clone()

        signal = dataset.transform(signal, **batch)
        original = dataset.transform(original, **batch)
        mask = batch["Silence"]["mask"]

        zeros_ = paddle.zeros_like(signal[mask].audio_data)
        original_ = original[~mask].audio_data

        assert paddle.allclose(signal[mask].audio_data, zeros_)
        assert paddle.allclose(original[~mask].audio_data, original_)


def test_nested_masking():
    transform = tfm.Compose(
        [
            tfm.VolumeNorm(prob=0.5),
            tfm.Silence(prob=0.9),
        ],
        prob=0.9, )

    loader = audiotools.data.datasets.AudioLoader(sources=["./audio/spk.csv"])
    dataset = audiotools.data.datasets.AudioDataset(
        loader,
        44100,
        n_examples=100,
        transform=transform, )
    dataloader = paddle.io.DataLoader(
        dataset, num_workers=0, batch_size=10, collate_fn=dataset.collate)

    for batch in dataloader:
        batch = util.prepare_batch(batch, device="cpu")
        signal = batch["signal"]
        kwargs = batch["transform_args"]
        with paddle.no_grad():
            output = dataset.transform(signal, **kwargs)


def test_smoothing_edge_case():
    transform = tfm.Smoothing()
    zeros = paddle.zeros([1, 1, 44100])
    signal = AudioSignal(zeros, 44100)
    kwargs = transform.instantiate(0, signal)
    output = transform(signal, **kwargs)

    assert paddle.allclose(output.audio_data, zeros)


def test_global_volume_norm():
    signal = AudioSignal.wave(440, 1, 44100, 1)

    # signal with -inf loudness should be unchanged
    signal.metadata["loudness"] = float("-inf")

    transform = tfm.GlobalVolumeNorm(db=("const", -100))
    kwargs = transform.instantiate(0, signal)

    output = transform(signal.clone(), **kwargs)
    assert paddle.allclose(output.samples, signal.samples)

    # signal without a loudness key should be unchanged
    signal.metadata.pop("loudness")
    kwargs = transform.instantiate(0, signal)
    output = transform(signal.clone(), **kwargs)
    assert paddle.allclose(output.samples, signal.samples)

    # signal with the actual loudness should be normalized
    signal.metadata["loudness"] = signal.ffmpeg_loudness()
    kwargs = transform.instantiate(0, signal)
    output = transform(signal.clone(), **kwargs)
    assert not paddle.allclose(output.samples, signal.samples)
