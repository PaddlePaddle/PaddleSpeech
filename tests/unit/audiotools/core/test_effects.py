# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/tests/core/test_effects.py)
import sys

import numpy as np
import paddle
import pytest

from paddlespeech.audiotools import AudioSignal


def test_normalize():
    audio_path = "./audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=10)
    signal = signal.normalize()
    assert np.allclose(signal.loudness(), -24, atol=1e-1)

    array = np.random.randn(1, 2, 32000)
    array = array / np.abs(array).max()

    signal = AudioSignal(array, sample_rate=16000)
    for db_incr in np.arange(10, 75, 5):
        db = -80 + db_incr
        signal = signal.normalize(db)
        loudness = signal.loudness()
        assert np.allclose(loudness, db, atol=1)  # TODO, atol=1e-1

    batch_size = 16
    db = -60 + paddle.linspace(10, 30, batch_size)

    array = np.random.randn(batch_size, 2, 32000)
    array = array / np.abs(array).max()
    signal = AudioSignal(array, sample_rate=16000)

    signal = signal.normalize(db)
    assert np.allclose(signal.loudness(), db, 1e-1)


def test_volume_change():
    audio_path = "./audio/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=10)

    boost = 3
    before_db = signal.loudness().clone()
    signal = signal.volume_change(boost)
    after_db = signal.loudness()
    assert np.allclose(before_db + boost, after_db)

    signal._loudness = None
    after_db = signal.loudness()
    assert np.allclose(before_db + boost, after_db, 1e-1)


def test_mix():
    audio_path = "./audio/spk/f10_script4_produced.wav"
    spk = AudioSignal(audio_path, offset=10, duration=10)

    audio_path = "./audio/nz/f5_script2_ipad_balcony1_room_tone.wav"
    nz = AudioSignal(audio_path, offset=10, duration=10)

    spk.deepcopy().mix(nz, snr=-10)
    snr = spk.loudness() - nz.loudness()
    assert np.allclose(snr, -10, atol=1)

    # Test in batch
    audio_path = "./audio/spk/f10_script4_produced.wav"
    spk = AudioSignal(audio_path, offset=10, duration=10)

    audio_path = "./audio/nz/f5_script2_ipad_balcony1_room_tone.wav"
    nz = AudioSignal(audio_path, offset=10, duration=10)

    batch_size = 4
    tgt_snr = paddle.linspace(-10, 10, batch_size)

    spk_batch = AudioSignal.batch([spk.deepcopy() for _ in range(batch_size)])
    nz_batch = AudioSignal.batch([nz.deepcopy() for _ in range(batch_size)])

    spk_batch.deepcopy().mix(nz_batch, snr=tgt_snr)
    snr = spk_batch.loudness() - nz_batch.loudness()
    assert np.allclose(snr, tgt_snr, atol=1)

    # Test with "EQing" the other signal
    db = 0 + 0 * paddle.rand([10])
    spk_batch.deepcopy().mix(nz_batch, snr=tgt_snr, other_eq=db)
    snr = spk_batch.loudness() - nz_batch.loudness()
    assert np.allclose(snr, tgt_snr, atol=1)


def test_convolve():
    np.random.seed(6)  # Found a failing seed
    audio_path = "./audio/spk/f10_script4_produced.wav"
    spk = AudioSignal(audio_path, offset=10, duration=10)

    impulse = np.zeros((1, 16000), dtype="float32")
    impulse[..., 0] = 1
    ir = AudioSignal(impulse, 16000)
    batch_size = 4

    spk_batch = AudioSignal.batch([spk.deepcopy() for _ in range(batch_size)])
    ir_batch = AudioSignal.batch(
        [
            ir.deepcopy().zero_pad(np.random.randint(1000), 0)
            for _ in range(batch_size)
        ],
        pad_signals=True, )

    convolved = spk_batch.deepcopy().convolve(ir_batch)
    assert convolved == spk_batch

    # Short duration
    audio_path = "./audio/spk/f10_script4_produced.wav"
    spk = AudioSignal(audio_path, offset=10, duration=0.1)

    impulse = np.zeros((1, 16000), dtype="float32")
    impulse[..., 0] = 1
    ir = AudioSignal(impulse, 16000)
    batch_size = 4

    spk_batch = AudioSignal.batch([spk.deepcopy() for _ in range(batch_size)])
    ir_batch = AudioSignal.batch(
        [
            ir.deepcopy().zero_pad(np.random.randint(1000), 0)
            for _ in range(batch_size)
        ],
        pad_signals=True, )

    convolved = spk_batch.deepcopy().convolve(ir_batch)
    assert convolved == spk_batch


def test_pipeline():
    # An actual IR, no batching
    audio_path = "./audio/spk/f10_script4_produced.wav"
    spk = AudioSignal(audio_path, offset=10, duration=5)

    audio_path = "./audio/ir/h179_Bar_1txts.wav"
    ir = AudioSignal(audio_path)
    spk.deepcopy().convolve(ir)

    audio_path = "./audio/nz/f5_script2_ipad_balcony1_room_tone.wav"
    nz = AudioSignal(audio_path, offset=10, duration=5)

    batch_size = 16
    tgt_snr = paddle.linspace(20, 30, batch_size)

    (spk @ ir).mix(nz, snr=tgt_snr)


@pytest.mark.parametrize("n_bands", [1, 2, 4, 8, 12, 16])
def test_mel_filterbank(n_bands):
    audio_path = "./audio/spk/f10_script4_produced.wav"
    spk = AudioSignal(audio_path, offset=10, duration=1)
    fbank = spk.deepcopy().mel_filterbank(n_bands)

    assert paddle.allclose(fbank.sum(-1), spk.audio_data, atol=1e-6)

    # Check if it works in batches.
    spk_batch = AudioSignal.batch([
        AudioSignal.excerpt("./audio/spk/f10_script4_produced.wav", duration=2)
        for _ in range(16)
    ])
    fbank = spk_batch.deepcopy().mel_filterbank(n_bands)
    summed = fbank.sum(-1)
    assert paddle.allclose(summed, spk_batch.audio_data, atol=1e-6)


@pytest.mark.parametrize("n_bands", [1, 2, 4, 8, 12, 16])
def test_equalizer(n_bands):
    audio_path = "./audio/spk/f10_script4_produced.wav"
    spk = AudioSignal(audio_path, offset=10, duration=10)

    db = -3 + 1 * paddle.rand([n_bands])
    spk.deepcopy().equalizer(db)

    db = -3 + 1 * np.random.rand(n_bands)
    spk.deepcopy().equalizer(db)

    audio_path = "./audio/ir/h179_Bar_1txts.wav"
    ir = AudioSignal(audio_path)
    db = -3 + 1 * paddle.rand([n_bands])

    spk.deepcopy().convolve(ir.equalizer(db))

    spk_batch = AudioSignal.batch([
        AudioSignal.excerpt("./audio/spk/f10_script4_produced.wav", duration=2)
        for _ in range(16)
    ])

    db = paddle.zeros([spk_batch.batch_size, n_bands])
    output = spk_batch.deepcopy().equalizer(db)

    assert output == spk_batch


def test_clip_distortion():
    audio_path = "./audio/spk/f10_script4_produced.wav"
    spk = AudioSignal(audio_path, offset=10, duration=2)
    clipped = spk.deepcopy().clip_distortion(0.05)

    spk_batch = AudioSignal.batch([
        AudioSignal.excerpt("./audio/spk/f10_script4_produced.wav", duration=2)
        for _ in range(16)
    ])
    percs = paddle.to_tensor(np.random.uniform(size=(16, ))).astype("float32")
    clipped_batch = spk_batch.deepcopy().clip_distortion(percs)

    assert clipped.audio_data.abs().max() < 1.0
    assert clipped_batch.audio_data.abs().max() < 1.0


@pytest.mark.parametrize("quant_ch", [2, 4, 8, 16, 32, 64, 128])
def test_quantization(quant_ch):
    audio_path = "./audio/spk/f10_script4_produced.wav"
    spk = AudioSignal(audio_path, offset=10, duration=2)

    quantized = spk.deepcopy().quantization(quant_ch)

    # Need to round audio_data off because torch ops with straight
    # through estimator are sometimes a bit off past 3 decimal places.
    found_quant_ch = len(np.unique(np.around(quantized.audio_data, decimals=3)))
    assert found_quant_ch <= quant_ch

    spk_batch = AudioSignal.batch([
        AudioSignal.excerpt("./audio/spk/f10_script4_produced.wav", duration=2)
        for _ in range(16)
    ])

    quant_ch = np.random.choice(
        [2, 4, 8, 16, 32, 64, 128], size=(16, ), replace=True)
    quantized = spk_batch.deepcopy().quantization(quant_ch)

    for i, q_ch in enumerate(quant_ch):
        found_quant_ch = len(
            np.unique(np.around(quantized.audio_data[i], decimals=3)))
        assert found_quant_ch <= q_ch


@pytest.mark.parametrize("quant_ch", [2, 4, 8, 16, 32, 64, 128])
def test_mulaw_quantization(quant_ch):
    audio_path = "./audio/spk/f10_script4_produced.wav"
    spk = AudioSignal(audio_path, offset=10, duration=2)

    quantized = spk.deepcopy().mulaw_quantization(quant_ch)

    # Need to round audio_data off because torch ops with straight
    # through estimator are sometimes a bit off past 3 decimal places.
    found_quant_ch = len(np.unique(np.around(quantized.audio_data, decimals=3)))
    assert found_quant_ch <= quant_ch

    spk_batch = AudioSignal.batch([
        AudioSignal.excerpt("./audio/spk/f10_script4_produced.wav", duration=2)
        for _ in range(16)
    ])

    quant_ch = np.random.choice(
        [2, 4, 8, 16, 32, 64, 128], size=(16, ), replace=True)
    quantized = spk_batch.deepcopy().mulaw_quantization(quant_ch)

    for i, q_ch in enumerate(quant_ch):
        found_quant_ch = len(
            np.unique(np.around(quantized.audio_data[i], decimals=3)))
        assert found_quant_ch <= q_ch


def test_impulse_response_augmentation():
    audio_path = "./audio/ir/h179_Bar_1txts.wav"
    batch_size = 16
    ir = AudioSignal(audio_path)
    ir_batch = AudioSignal.batch([ir for _ in range(batch_size)])
    early_response, late_field, window = ir_batch.decompose_ir()

    assert early_response.shape == late_field.shape
    assert late_field.shape == window.shape

    drr = ir_batch.measure_drr()

    alpha = AudioSignal.solve_alpha(early_response, late_field, window, drr)
    assert np.allclose(alpha, np.ones_like(alpha), 1e-5)

    target_drr = 5
    out = ir_batch.deepcopy().alter_drr(target_drr)
    drr = out.measure_drr()
    assert np.allclose(drr, np.ones_like(drr) * target_drr)

    target_drr = np.random.rand(batch_size).astype("float32") * 50
    altered_ir = ir_batch.deepcopy().alter_drr(target_drr)
    drr = altered_ir.measure_drr()
    assert np.allclose(drr.flatten(), target_drr.flatten())


def test_apply_ir():
    audio_path = "./audio/spk/f10_script4_produced.wav"
    ir_path = "./audio/ir/h179_Bar_1txts.wav"

    spk = AudioSignal(audio_path, offset=10, duration=2)
    ir = AudioSignal(ir_path)
    db = 0 + 0 * paddle.rand([10])
    output = spk.deepcopy().apply_ir(ir, drr=10, ir_eq=db)

    assert np.allclose(ir.measure_drr().flatten(), 10)

    output = spk.deepcopy().apply_ir(
        ir, drr=10, ir_eq=db, use_original_phase=True)


def test_ensure_max_of_audio():
    spk = AudioSignal(paddle.randn([1, 1, 44100]), 44100)

    max_vals = [1.0] + [np.random.rand() for _ in range(10)]
    for val in max_vals:
        after = spk.deepcopy().ensure_max_of_audio(val)
        assert after.audio_data.abs().max() <= val + 1e-3

    # Make sure it does nothing to a tiny signal
    spk = AudioSignal(paddle.rand([1, 1, 44100]), 44100)
    spk.audio_data = spk.audio_data * 0.5
    after = spk.deepcopy().ensure_max_of_audio()

    assert paddle.allclose(after.audio_data, spk.audio_data)
