# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/tests/core/test_util.py)
import os
import random
import sys
import tempfile

import numpy as np
import paddle
import pytest

from audio.audiotools import util
from audio.audiotools.core.audio_signal import AudioSignal
from paddlespeech.vector.training.seeding import seed_everything


def test_check_random_state():
    # seed is None
    rng_type = type(np.random.RandomState(10))
    rng = util.random_state(None)
    assert type(rng) == rng_type

    # seed is int
    rng = util.random_state(10)
    assert type(rng) == rng_type

    # seed is RandomState
    rng_test = np.random.RandomState(10)
    rng = util.random_state(rng_test)
    assert type(rng) == rng_type

    # seed is none of the above : error
    pytest.raises(ValueError, util.random_state, "random")


def test_seed():
    seed_everything(0)
    paddle_result_a = paddle.randn([1])
    np_result_a = np.random.randn(1)
    py_result_a = random.random()

    seed_everything(0)
    paddle_result_b = paddle.randn([1])
    np_result_b = np.random.randn(1)
    py_result_b = random.random()

    assert paddle_result_a == paddle_result_b
    assert np_result_a == np_result_b
    assert py_result_a == py_result_b


def test_hz_to_bin():
    hz = paddle.to_tensor(np.array([100, 200, 300]), dtype="float32")
    sr = 1000
    n_fft = 2048

    bins = util.hz_to_bin(hz, n_fft, sr)

    assert (((bins / n_fft) * sr) - hz).abs().max() < 1


def test_find_audio():
    wav_files = util.find_audio("tests/", ["wav"])
    for a in wav_files:
        assert "wav" in str(a)

    audio_files = util.find_audio("tests/", ["flac"])
    assert not audio_files

    # Make sure it works with single audio files
    audio_files = util.find_audio("./audio/spk//f10_script4_produced.wav")

    # Make sure it works with globs
    audio_files = util.find_audio("tests/**/*.wav")
    assert len(audio_files) == len(wav_files)


def test_chdir():
    with tempfile.TemporaryDirectory(suffix="tmp") as d:
        with util.chdir(d):
            assert os.path.samefile(d, os.path.realpath("."))


def test_prepare_batch():
    batch = {"tensor": paddle.randn([1]), "non_tensor": np.random.randn(1)}
    util.prepare_batch(batch)

    batch = paddle.randn([1])
    util.prepare_batch(batch)

    batch = [paddle.randn([1]), np.random.randn(1)]
    util.prepare_batch(batch)


def test_sample_dist():
    state = util.random_state(0)
    v1 = state.uniform(0.0, 1.0)
    v2 = util.sample_from_dist(("uniform", 0.0, 1.0), 0)
    assert v1 == v2

    assert util.sample_from_dist(("const", 1.0)) == 1.0

    dist_tuple = ("choice", [8, 16, 32])
    assert util.sample_from_dist(dist_tuple) in [8, 16, 32]


def test_collate():
    batch_size = 16

    def _one_item():
        return {
            "signal": AudioSignal(paddle.randn([1, 1, 44100]), 44100),
            "tensor": paddle.randn([1]),
            "string": "Testing",
            "dict": {
                "nested_signal":
                AudioSignal(paddle.randn([1, 1, 44100]), 44100),
            },
        }

    items = [_one_item() for _ in range(batch_size)]
    collated = util.collate(items)

    assert collated["signal"].batch_size == batch_size
    assert collated["tensor"].shape[0] == batch_size
    assert len(collated["string"]) == batch_size
    assert collated["dict"]["nested_signal"].batch_size == batch_size

    # test collate with splitting (evenly)
    batch_size = 16
    n_splits = 4

    items = [_one_item() for _ in range(batch_size)]
    collated = util.collate(items, n_splits=n_splits)

    for x in collated:
        assert x["signal"].batch_size == batch_size // n_splits
        assert x["tensor"].shape[0] == batch_size // n_splits
        assert len(x["string"]) == batch_size // n_splits
        assert x["dict"]["nested_signal"].batch_size == batch_size // n_splits

    # test collate with splitting (unevenly)
    batch_size = 15
    n_splits = 4

    items = [_one_item() for _ in range(batch_size)]
    collated = util.collate(items, n_splits=n_splits)

    tlen = [4, 4, 4, 3]

    for x, t in zip(collated, tlen):
        assert x["signal"].batch_size == t
        assert x["tensor"].shape[0] == t
        assert len(x["string"]) == t
        assert x["dict"]["nested_signal"].batch_size == t
