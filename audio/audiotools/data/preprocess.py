# MIT License, Copyright (c) 2023-Present, Descript.
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Modified from audiotools(https://github.com/descriptinc/audiotools/blob/master/audiotools/data/preprocess.py)
import csv
import os
from pathlib import Path

from tqdm import tqdm

from ..core import AudioSignal


def create_csv(audio_files: list,
               output_csv: Path,
               loudness: bool=False,
               data_path: str=None):
    """Converts a folder of audio files to a CSV file. If ``loudness = True``,
    the output of this function will create a CSV file that looks something
    like:

    ..  csv-table::
        :header: path,loudness

        daps/produced/f1_script1_produced.wav,-16.299999237060547
        daps/produced/f1_script2_produced.wav,-16.600000381469727
        daps/produced/f1_script3_produced.wav,-17.299999237060547
        daps/produced/f1_script4_produced.wav,-16.100000381469727
        daps/produced/f1_script5_produced.wav,-16.700000762939453
        daps/produced/f3_script1_produced.wav,-16.5

    ..  note::
        The paths above are written relative to the ``data_path`` argument
        which defaults to the environment variable ``PATH_TO_DATA`` if
        it isn't passed to this function, and defaults to the empty string
        if that environment variable is not set.

    You can produce a CSV file from a directory of audio files via:

    >>> from audio import audiotools
    >>> directory = ...
    >>> audio_files = audiotools.util.find_audio(directory)
    >>> output_path = "train.csv"
    >>> audiotools.data.preprocess.create_csv(
    >>>     audio_files, output_csv, loudness=True
    >>> )

    Note that you can create empty rows in the CSV file by passing an empty
    string or None in the ``audio_files`` list. This is useful if you want to
    sync multiple CSV files in a multitrack setting. The loudness of these
    empty rows will be set to -inf.

    Parameters
    ----------
    audio_files : list
        List of audio files.
    output_csv : Path
        Output CSV, with each row containing the relative path of every file
        to ``data_path``, if specified (defaults to None).
    loudness : bool
        Compute loudness of entire file and store alongside path.
    """

    info = []
    pbar = tqdm(audio_files)
    for af in pbar:
        af = Path(af)
        pbar.set_description(f"Processing {af.name}")
        _info = {}
        if af.name == "":
            _info["path"] = ""
            if loudness:
                _info["loudness"] = -float("inf")
        else:
            _info["path"] = af.relative_to(
                data_path) if data_path is not None else af
            if loudness:
                _info["loudness"] = AudioSignal(af).ffmpeg_loudness().item()

        info.append(_info)

    with open(output_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(info[0].keys()))
        writer.writeheader()

        for item in info:
            writer.writerow(item)
