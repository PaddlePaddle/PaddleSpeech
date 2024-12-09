import sys
import tempfile
from pathlib import Path

import paddle
sys.path.append("/home/aistudio/PaddleSpeech/audio")
from audiotools.core.util import find_audio
from audiotools.core.util import read_sources
from audiotools.data import preprocess


def test_create_csv():
    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        preprocess.create_csv(
            find_audio("./tests/audiotools/audio/spk", ext=["wav"]),
            f.name,
            loudness=True)


def test_create_csv_with_empty_rows():
    audio_files = find_audio("./tests/audiotools/audio/spk", ext=["wav"])
    audio_files.insert(0, "")
    audio_files.insert(2, "")

    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        preprocess.create_csv(audio_files, f.name, loudness=True)

        audio_files = read_sources([f.name], remove_empty=True)
        assert len(audio_files[0]) == 1
        audio_files = read_sources([f.name], remove_empty=False)
        assert len(audio_files[0]) == 3
