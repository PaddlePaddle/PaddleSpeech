# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/tests/core/test_loudness.py)
import sys

import numpy as np
import pyloudnorm
import soundfile as sf

from paddlespeech.audiotools import AudioSignal
from paddlespeech.audiotools import datasets
from paddlespeech.audiotools import Meter
from paddlespeech.audiotools import transforms

ATOL = 1e-1


def test_loudness_against_pyln():
    audio_path = "./audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=5, duration=10)
    signal_loudness = signal.loudness()

    meter = pyloudnorm.Meter(
        signal.sample_rate, filter_class="K-weighting", block_size=0.4)
    py_loudness = meter.integrated_loudness(signal.numpy()[0].T)
    assert np.allclose(signal_loudness, py_loudness)


def test_loudness_short():
    audio_path = "./audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=0.25)
    signal_loudness = signal.loudness()


def test_batch_loudness():
    np.random.seed(0)
    array = np.random.randn(16, 2, 16000)
    array /= np.abs(array).max()

    gains = np.random.rand(array.shape[0])[:, None, None]
    array = array * gains

    meter = pyloudnorm.Meter(16000)
    py_loudness = [
        meter.integrated_loudness(array[i].T) for i in range(array.shape[0])
    ]

    meter = Meter(16000)
    meter.filter_class
    at_loudness_iso = [
        meter.integrated_loudness(array[i].T).item()
        for i in range(array.shape[0])
    ]

    assert np.allclose(py_loudness, at_loudness_iso, atol=1e-1)

    signal = AudioSignal(array, sample_rate=16000)
    at_loudness_batch = signal.loudness()
    assert np.allclose(py_loudness, at_loudness_batch, atol=1e-1)


# Tests below are copied from pyloudnorm
def test_integrated_loudness():
    data, rate = sf.read("./audio/loudness/sine_1000.wav")
    meter = Meter(rate)
    loudness = meter(data)

    targetLoudness = -3.0523438444331137
    assert np.allclose(loudness, targetLoudness)


def test_rel_gate_test():
    data, rate = sf.read("./audio/loudness/1770-2_Comp_RelGateTest.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -10.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_abs_gate_test():
    data, rate = sf.read("./audio/loudness/1770-2_Comp_AbsGateTest.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -69.5
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_24LKFS_25Hz_2ch():
    data, rate = sf.read("./audio/loudness/1770-2_Comp_24LKFS_25Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_24LKFS_100Hz_2ch():
    data, rate = sf.read("./audio/loudness/1770-2_Comp_24LKFS_100Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_24LKFS_500Hz_2ch():
    data, rate = sf.read("./audio/loudness/1770-2_Comp_24LKFS_500Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_24LKFS_1000Hz_2ch():
    data, rate = sf.read("./audio/loudness/1770-2_Comp_24LKFS_1000Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_24LKFS_2000Hz_2ch():
    data, rate = sf.read("./audio/loudness/1770-2_Comp_24LKFS_2000Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_24LKFS_10000Hz_2ch():
    data, rate = sf.read("./audio/loudness/1770-2_Comp_24LKFS_10000Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_23LKFS_25Hz_2ch():
    data, rate = sf.read("./audio/loudness/1770-2_Comp_23LKFS_25Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_23LKFS_100Hz_2ch():
    data, rate = sf.read("./audio/loudness/1770-2_Comp_23LKFS_100Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_23LKFS_500Hz_2ch():
    data, rate = sf.read("./audio/loudness/1770-2_Comp_23LKFS_500Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_23LKFS_1000Hz_2ch():
    data, rate = sf.read("./audio/loudness/1770-2_Comp_23LKFS_1000Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_23LKFS_2000Hz_2ch():
    data, rate = sf.read("./audio/loudness/1770-2_Comp_23LKFS_2000Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_23LKFS_10000Hz_2ch():
    data, rate = sf.read("./audio/loudness/1770-2_Comp_23LKFS_10000Hz_2ch.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_18LKFS_frequency_sweep():
    data, rate = sf.read(
        "./audio/loudness/1770-2_Comp_18LKFS_FrequencySweep.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -18.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_conf_stereo_vinL_R_23LKFS():
    data, rate = sf.read(
        "./audio/loudness/1770-2_Conf_Stereo_VinL+R-23LKFS.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_conf_monovoice_music_24LKFS():
    data, rate = sf.read(
        "./audio/loudness/1770-2_Conf_Mono_Voice+Music-24LKFS.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def conf_monovoice_music_24LKFS():
    data, rate = sf.read(
        "./audio/loudness/1770-2_Conf_Mono_Voice+Music-24LKFS.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -24.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_conf_monovoice_music_23LKFS():
    data, rate = sf.read(
        "./audio/loudness/1770-2_Conf_Mono_Voice+Music-23LKFS.wav")
    meter = Meter(rate)
    loudness = meter.integrated_loudness(data)

    targetLoudness = -23.0
    assert np.allclose(loudness, targetLoudness, atol=ATOL)


def test_fir_accuracy():
    transform = transforms.Compose(
        transforms.ClippingDistortion(prob=0.5),
        transforms.LowPass(prob=0.5),
        transforms.HighPass(prob=0.5),
        transforms.Equalizer(prob=0.5),
        prob=0.5, )
    loader = datasets.AudioLoader(sources=["./audio/spk.csv"])
    dataset = datasets.AudioDataset(
        loader,
        44100,
        10,
        5.0,
        transform=transform, )

    for i in range(20):
        item = dataset[i]
        kwargs = item["transform_args"]
        signal = item["signal"]
        signal = transform(signal, **kwargs)

        signal._loudness = None
        iir_db = signal.clone().loudness()
        fir_db = signal.clone().loudness(use_fir=True)

        assert np.allclose(iir_db, fir_db, atol=1e-2)
