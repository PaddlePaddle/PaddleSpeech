# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/tests/core/test_display.py)
import sys
from pathlib import Path

import numpy as np
from visualdl import LogWriter

from paddlespeech.audiotools import AudioSignal


def test_specshow():
    array = np.zeros((1, 16000))
    AudioSignal(array, sample_rate=16000).specshow()
    AudioSignal(array, sample_rate=16000).specshow(preemphasis=True)
    AudioSignal(
        array, sample_rate=16000).specshow(
            title="test", preemphasis=True)
    AudioSignal(
        array, sample_rate=16000).specshow(
            format=False, preemphasis=True)
    AudioSignal(
        array, sample_rate=16000).specshow(
            format=False, preemphasis=False, y_axis="mel")


def test_waveplot():
    array = np.zeros((1, 16000))
    AudioSignal(array, sample_rate=16000).waveplot()


def test_wavespec():
    array = np.zeros((1, 16000))
    AudioSignal(array, sample_rate=16000).wavespec()


def test_write_audio_to_tb():
    signal = AudioSignal("./audio/spk/f10_script4_produced.mp3", duration=5)

    Path("./scratch").mkdir(parents=True, exist_ok=True)
    writer = LogWriter("./scratch/")
    signal.write_audio_to_tb("tag", writer)


def test_save_image():
    signal = AudioSignal(
        "./audio/spk/f10_script4_produced.wav", duration=10, offset=10)
    Path("./scratch").mkdir(parents=True, exist_ok=True)
    signal.save_image("./scratch/image.png")
