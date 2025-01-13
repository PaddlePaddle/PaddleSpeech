# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/tests/core/test_audio_signal.py)
import pathlib
import sys
import tempfile

import librosa
import numpy as np
import paddle
import pytest
import rich

from audio import audiotools
from audio.audiotools import AudioSignal
from audio.audiotools import util


def test_io():
    audio_path = "./audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(pathlib.Path(audio_path))

    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        signal.write(f.name)
        signal_from_file = AudioSignal(f.name)

    mp3_signal = AudioSignal(audio_path.replace("wav", "mp3"))
    print(mp3_signal)

    assert signal == signal_from_file
    print(signal)
    print(signal.markdown())

    mp3_signal = AudioSignal.excerpt(
        audio_path.replace("wav", "mp3"), offset=5, duration=5)
    assert mp3_signal.signal_duration == 5.0
    assert mp3_signal.duration == 5.0
    assert mp3_signal.length == mp3_signal.signal_length

    rich.print(signal)

    array = np.random.randn(2, 16000)
    signal = AudioSignal(array, sample_rate=16000)
    assert np.allclose(signal.numpy(), array)

    signal = AudioSignal(array, 44100)
    assert signal.sample_rate == 44100
    signal.shape

    with pytest.raises(ValueError):
        signal = AudioSignal(5, sample_rate=16000)

    signal = AudioSignal(audio_path, offset=10, duration=10)
    assert np.allclose(signal.signal_duration, 10.0)
    assert np.allclose(signal.duration, 10.0)

    signal = AudioSignal.excerpt(audio_path, offset=5, duration=5)
    assert signal.signal_duration == 5.0
    assert signal.duration == 5.0

    assert "offset" in signal.metadata
    assert "duration" in signal.metadata

    signal = AudioSignal(paddle.randn([1000]), 44100)
    assert signal.audio_data.ndim == 3
    assert paddle.all(signal.samples == signal.audio_data)

    audio_path = "./audio/spk/f10_script4_produced.wav"
    assert AudioSignal(audio_path).hash() == AudioSignal(audio_path).hash()
    assert AudioSignal(audio_path).hash() != AudioSignal(audio_path).normalize(
        -20).hash()

    with pytest.raises(RuntimeError):
        AudioSignal(audio_path, offset=100000, duration=3)


def test_copy_and_clone():
    audio_path = "./audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path)
    signal.stft()
    signal.loudness()

    copied = signal.copy()
    deep_copied = signal.deepcopy()
    cloned = signal.clone()

    for a in ["audio_data", "stft_data", "_loudness"]:
        a1 = getattr(signal, a)
        a2 = getattr(cloned, a)
        a3 = getattr(copied, a)
        a4 = getattr(deep_copied, a)

        assert id(a1) != id(a2)
        assert id(a1) == id(a3)
        assert id(a1) != id(a4)

        assert np.allclose(a1, a2)
        assert np.allclose(a1, a3)
        assert np.allclose(a1, a4)

    for a in ["path_to_file", "metadata"]:
        a1 = getattr(signal, a)
        a2 = getattr(cloned, a)
        a3 = getattr(copied, a)
        a4 = getattr(deep_copied, a)

        assert id(a1) == id(a2) if isinstance(a1, str) else id(a1) != id(a2)
        assert id(a1) == id(a3)
        assert id(a1) == id(a4) if isinstance(a1, str) else id(a1) != id(a2)

    # for clone, id should differ if path is list, and should differ always for metadata
    # if path is string, id should remain same...

    assert signal.original_signal_length == copied.original_signal_length
    assert signal.original_signal_length == deep_copied.original_signal_length
    assert signal.original_signal_length == cloned.original_signal_length

    signal = signal.detach()


@pytest.mark.parametrize("loudness_cutoff", [-np.inf, -160, -80, -40, -20])
def test_salient_excerpt(loudness_cutoff):
    MAP = {-np.inf: 0.0, -160: 0.0, -80: 0.001, -40: 0.01, -20: 0.1}
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sr = 44100
        signal = AudioSignal(paddle.zeros([sr * 60]), sr)

        signal[..., sr * 20:sr * 21] = MAP[loudness_cutoff] * paddle.randn(
            [44100])

        signal.write(f.name)
        signal = AudioSignal.salient_excerpt(
            f.name, loudness_cutoff=loudness_cutoff, duration=1, num_tries=None)

        assert "offset" in signal.metadata
        assert "duration" in signal.metadata
        assert signal.loudness() >= loudness_cutoff

        signal = AudioSignal.salient_excerpt(
            f.name, loudness_cutoff=np.inf, duration=1, num_tries=10)
        signal = AudioSignal.salient_excerpt(
            f.name,
            loudness_cutoff=None,
            duration=1, )


def test_arithmetic():
    def _make_signals():
        array = np.random.randn(2, 16000)
        sig1 = AudioSignal(array, sample_rate=16000)

        array = np.random.randn(2, 16000)
        sig2 = AudioSignal(array, sample_rate=16000)
        return sig1, sig2

    # Addition (with a copy)
    sig1, sig2 = _make_signals()
    sig3 = sig1 + sig2
    assert paddle.allclose(sig3.audio_data, sig1.audio_data + sig2.audio_data)

    # Addition (rmul)
    sig1, _ = _make_signals()
    sig3 = 5.0 + sig1
    assert paddle.allclose(sig3.audio_data, sig1.audio_data + 5.0)

    # In place addition
    sig3, sig2 = _make_signals()
    sig1 = sig3.deepcopy()
    sig3 += sig2
    assert paddle.allclose(sig3.audio_data, sig1.audio_data + sig2.audio_data)

    # Subtraction (with a copy)
    sig1, sig2 = _make_signals()
    sig3 = sig1 - sig2
    assert paddle.allclose(sig3.audio_data, sig1.audio_data - sig2.audio_data)

    # In place subtraction
    sig3, sig2 = _make_signals()
    sig1 = sig3.deepcopy()
    sig3 -= sig2
    assert paddle.allclose(sig3.audio_data, sig1.audio_data - sig2.audio_data)

    # Multiplication (element-wise)
    sig1, sig2 = _make_signals()
    sig3 = sig1 * sig2
    assert paddle.allclose(sig3.audio_data, sig1.audio_data * sig2.audio_data)

    # Multiplication (gain)
    sig1, _ = _make_signals()
    sig3 = sig1 * 5.0
    assert paddle.allclose(sig3.audio_data, sig1.audio_data * 5.0)

    # Multiplication (rmul)
    sig1, _ = _make_signals()
    sig3 = 5.0 * sig1
    assert paddle.allclose(sig3.audio_data, sig1.audio_data * 5.0)

    # Multiplication (in-place)
    sig3, sig2 = _make_signals()
    sig1 = sig3.deepcopy()
    sig3 *= sig2
    assert paddle.allclose(sig3.audio_data, sig1.audio_data * sig2.audio_data)


def test_equality():
    array = np.random.randn(2, 16000)
    sig1 = AudioSignal(array, sample_rate=16000)
    sig2 = AudioSignal(array, sample_rate=16000)

    assert sig1 == sig2

    array = np.random.randn(2, 16000)
    sig3 = AudioSignal(array, sample_rate=16000)

    assert sig1 != sig3

    assert not np.allclose(sig1.numpy(), sig3.numpy())


def test_indexing():
    array = np.random.randn(4, 2, 16000)
    sig1 = AudioSignal(array, sample_rate=16000)

    assert np.allclose(sig1[0].audio_data, array[0])
    assert np.allclose(sig1[0, :, 8000].audio_data, array[0, :, 8000])

    # Test with the associated STFT data.
    array = np.random.randn(4, 2, 16000)
    sig1 = AudioSignal(array, sample_rate=16000)
    sig1.loudness()
    sig1.stft()

    indexed = sig1[0]

    assert np.allclose(indexed.audio_data, array[0])
    assert np.allclose(indexed.stft_data, sig1.stft_data[0])
    assert np.allclose(indexed._loudness, sig1._loudness[0])

    indexed = sig1[0:2]

    assert np.allclose(indexed.audio_data, array[0:2])
    assert np.allclose(indexed.stft_data, sig1.stft_data[0:2])
    assert np.allclose(indexed._loudness, sig1._loudness[0:2])

    # Test using a boolean tensor to index batch
    mask = paddle.to_tensor([True, False, True, False])
    indexed = sig1[mask]

    assert np.allclose(indexed.audio_data, sig1.audio_data[mask])
    # assert np.allclose(indexed.stft_data, sig1.stft_data[mask])
    assert np.allclose(indexed.stft_data,
                       util.bool_index_compat(sig1.stft_data, mask))
    assert np.allclose(indexed._loudness, sig1._loudness[mask])

    # Set parts of signal using tensor
    other_array = paddle.to_tensor(np.random.randn(4, 2, 16000))
    sig1 = AudioSignal(array, sample_rate=16000)
    sig1[0, :, 6000:8000] = other_array[0, :, 6000:8000]

    assert np.allclose(sig1[0, :, 6000:8000].audio_data,
                       other_array[0, :, 6000:8000])

    # Set parts of signal using AudioSignal
    sig2 = AudioSignal(other_array, sample_rate=16000)

    sig1 = AudioSignal(array, sample_rate=16000)
    sig1[0, :, 6000:8000] = sig2[0, :, 6000:8000]

    assert np.allclose(sig1[0, :, 6000:8000].audio_data,
                       sig2[0, :, 6000:8000].audio_data)

    # Check that loudnesses and stft_data get set as well, if only the batch
    # dim is indexed.
    sig2 = AudioSignal(other_array, sample_rate=16000)
    sig2.stft()
    sig2.loudness()

    sig1 = AudioSignal(array, sample_rate=16000)
    sig1.stft()
    sig1.loudness()

    # Test using a boolean tensor to index batch
    mask = paddle.to_tensor([True, False, True, False])
    sig1[mask] = sig2[mask]

    for k in ["stft_data", "audio_data", "_loudness"]:
        a1 = getattr(sig1, k)
        a2 = getattr(sig2, k)

        # assert np.allclose(a1[mask], a2[mask])
        assert np.allclose(
            util.bool_index_compat(a1, mask), util.bool_index_compat(a2, mask))


def test_zeros():
    x = AudioSignal.zeros(0.5, 44100)
    assert x.signal_duration == 0.5
    assert x.duration == 0.5
    assert x.sample_rate == 44100


@pytest.mark.parametrize("shape",
                         ["sine", "square", "sawtooth", "triangle", "beep"])
def test_waves(shape: str):
    # error case
    if shape == "beep":
        with pytest.raises(ValueError):
            AudioSignal.wave(440, 0.5, 44100, shape=shape)

        return

    x = AudioSignal.wave(440, 0.5, 44100, shape=shape)
    assert x.duration == 0.5
    assert x.sample_rate == 44100

    # test the default shape arg
    x = AudioSignal.wave(440, 0.5, 44100)
    assert x.duration == 0.5
    assert x.sample_rate == 44100


def test_zero_pad():
    array = np.random.randn(4, 2, 16000)
    sig1 = AudioSignal(array, sample_rate=16000)

    sig1.zero_pad(100, 100)
    zeros = paddle.zeros([4, 2, 100], dtype="float64")
    assert paddle.allclose(sig1.audio_data[..., :100], zeros)
    assert paddle.allclose(sig1.audio_data[..., -100:], zeros)


def test_zero_pad_to():
    array = np.random.randn(4, 2, 16000)
    sig1 = AudioSignal(array, sample_rate=16000)

    sig1.zero_pad_to(16100)
    zeros = paddle.zeros([4, 2, 100], dtype="float64")
    assert paddle.allclose(sig1.audio_data[..., -100:], zeros)
    assert sig1.signal_length == 16100

    sig1 = AudioSignal(array, sample_rate=16000)
    sig1.zero_pad_to(15000)
    assert sig1.signal_length == 16000

    sig1 = AudioSignal(array, sample_rate=16000)
    sig1.zero_pad_to(16100, mode="before")
    zeros = paddle.zeros([4, 2, 100], dtype="float64")
    assert paddle.allclose(sig1.audio_data[..., :100], zeros)
    assert sig1.signal_length == 16100

    sig1 = AudioSignal(array, sample_rate=16000)
    sig1.zero_pad_to(15000, mode="before")
    assert sig1.signal_length == 16000


def test_truncate():
    array = np.random.randn(4, 2, 16000)
    sig1 = AudioSignal(array, sample_rate=16000)

    sig1.truncate_samples(100)
    assert sig1.signal_length == 100
    assert np.allclose(sig1.audio_data, array[..., :100])


def test_trim():
    array = np.random.randn(4, 2, 16000)
    sig1 = AudioSignal(array, sample_rate=16000)

    sig1.trim(100, 100)
    assert sig1.signal_length == 16000 - 200
    assert np.allclose(sig1.audio_data, array[..., 100:-100])

    array = np.random.randn(4, 2, 16000)
    sig1 = AudioSignal(array, sample_rate=16000)
    sig1.trim(0, 0)
    assert np.allclose(sig1.audio_data, array)


def test_to_from_ops():
    audio_path = "./audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path)
    signal.stft()
    signal.loudness()
    signal = signal.to("cpu")

    assert str(signal.audio_data.place) == "Place(cpu)"
    assert isinstance(signal.numpy(), np.ndarray)

    signal.cpu()
    # signal.cuda()
    signal.float()


def test_device():
    audio_path = "./audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path)
    signal.to("cpu")

    assert str(signal.device) == "Place(cpu)"


@pytest.mark.parametrize("window_length", [2048, 512])
@pytest.mark.parametrize("hop_length", [512, 128])
@pytest.mark.parametrize("window_type", ["sqrt_hann", "hann", None])
def test_stft(window_length, hop_length, window_type):
    if hop_length >= window_length:
        hop_length = window_length // 2
    audio_path = "./audio/spk/f10_script4_produced.wav"
    stft_params = audiotools.STFTParams(
        window_length=window_length,
        hop_length=hop_length,
        window_type=window_type)
    for _stft_params in [None, stft_params]:
        signal = AudioSignal(audio_path, duration=10, stft_params=_stft_params)
        with pytest.raises(RuntimeError):
            signal.istft()

        stft_data = signal.stft()

        # assert paddle.allclose(signal.stft_data, stft_data)
        assert np.allclose(signal.stft_data.cpu().numpy(),
                           stft_data.cpu().numpy())
        copied_signal = signal.deepcopy()
        copied_signal.stft()
        copied_signal = copied_signal.istft()

        assert copied_signal == signal

        mag = signal.magnitude
        phase = signal.phase

        recon_stft = mag * util.exp_compat(1j * phase)
        # assert paddle.allclose(recon_stft, signal.stft_data)
        assert np.allclose(recon_stft.cpu().numpy(),
                           signal.stft_data.cpu().numpy())

        signal.stft_data = None
        mag = signal.magnitude
        signal.stft_data = None
        phase = signal.phase

        recon_stft = mag * util.exp_compat(1j * phase)
        # assert paddle.allclose(recon_stft, signal.stft_data)
        assert np.allclose(recon_stft.cpu().numpy(),
                           signal.stft_data.cpu().numpy())

        # Test with match_stride=True, ignoring the beginning and end.
        s = signal.stft_params
        if s.hop_length == s.window_length // 4:
            og_signal = signal.clone()
            stft_data = signal.stft(match_stride=True)
            recon_data = signal.istft(match_stride=True)
            discard = window_length * 2

            right_pad, _ = signal.compute_stft_padding(
                s.window_length, s.hop_length, match_stride=True)
            length = signal.signal_length + right_pad
            assert stft_data.shape[-1] == length // s.hop_length

            assert paddle.allclose(
                recon_data.audio_data[..., discard:-discard],
                og_signal.audio_data[..., discard:-discard],
                atol=1e-6, )


def test_log_magnitude():
    audio_path = "./audio/spk/f10_script4_produced.wav"
    for _ in range(10):
        signal = AudioSignal.excerpt(audio_path, duration=5.0)
        magnitude = signal.magnitude.numpy()[0, 0]
        librosa_log_mag = librosa.amplitude_to_db(magnitude)
        log_mag = signal.log_magnitude().numpy()[0, 0]

        # print(abs((log_mag - librosa_log_mag)).max())
        assert np.allclose(log_mag, librosa_log_mag, atol=10e-7)


@pytest.mark.parametrize("n_mels", [40, 80, 128])
@pytest.mark.parametrize("window_length", [2048, 512])
@pytest.mark.parametrize("hop_length", [512, 128])
@pytest.mark.parametrize("window_type", ["sqrt_hann", "hann", None])
def test_mel_spectrogram(n_mels, window_length, hop_length, window_type):
    if hop_length >= window_length:
        hop_length = window_length // 2
    audio_path = "./audio/spk/f10_script4_produced.wav"
    stft_params = audiotools.STFTParams(
        window_length=window_length,
        hop_length=hop_length,
        window_type=window_type)
    for _stft_params in [None, stft_params]:
        signal = AudioSignal(audio_path, duration=10, stft_params=_stft_params)
        mel_spec = signal.mel_spectrogram(n_mels=n_mels)
        assert mel_spec.shape[2] == n_mels


@pytest.mark.parametrize("n_mfcc", [20, 40])
@pytest.mark.parametrize("n_mels", [40, 80, 128])
@pytest.mark.parametrize("window_length", [2048, 512])
@pytest.mark.parametrize("hop_length", [512, 128])
def test_mfcc(n_mfcc, n_mels, window_length, hop_length):
    if hop_length >= window_length:
        hop_length = window_length // 2
    audio_path = "./audio/spk/f10_script4_produced.wav"
    stft_params = audiotools.STFTParams(
        window_length=window_length, hop_length=hop_length)
    for _stft_params in [None, stft_params]:
        signal = AudioSignal(audio_path, duration=10, stft_params=_stft_params)
        mfcc = signal.mfcc(n_mfcc=n_mfcc, n_mels=n_mels)
        assert mfcc.shape[2] == n_mfcc


def test_to_mono():
    array = np.random.randn(4, 2, 16000)
    sr = 16000

    signal = AudioSignal(array, sample_rate=sr)
    assert signal.num_channels == 2

    signal = signal.to_mono()
    assert signal.num_channels == 1


def test_float():
    array = np.random.randn(4, 1, 16000).astype("float64")
    sr = 1600
    signal = AudioSignal(array, sample_rate=sr)

    signal = signal.float()
    assert signal.audio_data.dtype == paddle.float32


@pytest.mark.parametrize("sample_rate", [8000, 16000, 22050, 44100, 48000])
def test_resample(sample_rate):
    array = np.random.randn(4, 2, 16000)
    sr = 16000

    signal = AudioSignal(array, sample_rate=sr)

    signal = signal.resample(sample_rate)
    assert signal.sample_rate == sample_rate
    assert signal.signal_length == sample_rate


def test_batching():
    signals = []
    batch_size = 16

    # All same length, same sample rate.
    for _ in range(batch_size):
        array = np.random.randn(2, 16000)
        signal = AudioSignal(array, sample_rate=16000)
        signals.append(signal)

    batched_signal = AudioSignal.batch(signals)
    assert batched_signal.batch_size == batch_size

    signals = []
    # All different lengths, same sample rate, pad signals
    for _ in range(batch_size):
        L = np.random.randint(8000, 32000)
        array = np.random.randn(2, L)
        signal = AudioSignal(array, sample_rate=16000)
        signals.append(signal)

    with pytest.raises(RuntimeError):
        batched_signal = AudioSignal.batch(signals)

    signal_lengths = [x.signal_length for x in signals]
    max_length = max(signal_lengths)
    batched_signal = AudioSignal.batch(signals, pad_signals=True)

    assert batched_signal.signal_length == max_length
    assert batched_signal.batch_size == batch_size

    signals = []
    # All different lengths, same sample rate, truncate signals
    for _ in range(batch_size):
        L = np.random.randint(8000, 32000)
        array = np.random.randn(2, L)
        signal = AudioSignal(array, sample_rate=16000)
        signals.append(signal)

    with pytest.raises(RuntimeError):
        batched_signal = AudioSignal.batch(signals)

    signal_lengths = [x.signal_length for x in signals]
    min_length = min(signal_lengths)
    batched_signal = AudioSignal.batch(signals, truncate_signals=True)

    assert batched_signal.signal_length == min_length
    assert batched_signal.batch_size == batch_size

    signals = []
    # All different lengths, different sample rate, pad signals
    for _ in range(batch_size):
        L = np.random.randint(8000, 32000)
        sr = np.random.choice([8000, 16000, 32000])
        array = np.random.randn(2, L)
        signal = AudioSignal(array, sample_rate=int(sr))
        signals.append(signal)

    with pytest.raises(RuntimeError):
        batched_signal = AudioSignal.batch(signals)

    signal_lengths = [x.signal_length for x in signals]
    max_length = max(signal_lengths)
    for i, x in enumerate(signals):
        x.path_to_file = i
    batched_signal = AudioSignal.batch(signals, resample=True, pad_signals=True)

    assert batched_signal.signal_length == max_length
    assert batched_signal.batch_size == batch_size
    assert batched_signal.path_to_file == list(range(len(signals)))
    assert batched_signal.path_to_input_file == batched_signal.path_to_file
