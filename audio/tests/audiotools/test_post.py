# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/tests/test_post.py)
import sys
from pathlib import Path

from audio.audiotools import AudioSignal
from audio.audiotools import post
from audio.audiotools import transforms


def test_audio_table():
    tfm = transforms.LowPass()

    audio_dict = {}

    audio_dict["inputs"] = [
        AudioSignal.excerpt("./audio/spk/f10_script4_produced.wav", duration=5)
        for _ in range(3)
    ]
    audio_dict["outputs"] = []
    for i in range(3):
        x = audio_dict["inputs"][i]

        kwargs = tfm.instantiate()
        output = tfm(x.clone(), **kwargs)
        audio_dict["outputs"].append(output)

    post.audio_table(audio_dict)
