# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/tests/data/test_preprocess.py)
import sys
import tempfile
from pathlib import Path

import paddle

from paddlespeech.audiotools.core.util import find_audio
from paddlespeech.audiotools.core.util import read_sources
from paddlespeech.audiotools.data import preprocess


def test_create_csv():
    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        preprocess.create_csv(
            find_audio("././audio/spk", ext=["wav"]), f.name, loudness=True)


def test_create_csv_with_empty_rows():
    audio_files = find_audio("././audio/spk", ext=["wav"])
    audio_files.insert(0, "")
    audio_files.insert(2, "")

    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        preprocess.create_csv(audio_files, f.name, loudness=True)

        audio_files = read_sources([f.name], remove_empty=True)
        assert len(audio_files[0]) == 1
        audio_files = read_sources([f.name], remove_empty=False)
        assert len(audio_files[0]) == 3
