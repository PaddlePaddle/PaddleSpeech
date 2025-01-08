# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains code derived from [Descript]
# (https://github.com/descriptinc/audiotools),
# which is licensed under the MIT License:
#
# MIT License
#
# Copyright (c) 2023-Present, Descript
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import ffmpy
import numpy as np
import paddle


def r128stats(filepath: str, quiet: bool):
    """Takes a path to an audio file, returns a dict with the loudness
    stats computed by the ffmpeg ebur128 filter.

    Parameters
    ----------
    filepath : str
        Path to compute loudness stats on.
    quiet : bool
        Whether to show FFMPEG output during computation.

    Returns
    -------
    dict
        Dictionary containing loudness stats.
    """
    ffargs = [
        "ffmpeg",
        "-nostats",
        "-i",
        filepath,
        "-filter_complex",
        "ebur128",
        "-f",
        "null",
        "-",
    ]
    if quiet:
        ffargs += ["-hide_banner"]
    proc = subprocess.Popen(
        ffargs, stderr=subprocess.PIPE, universal_newlines=True)
    stats = proc.communicate()[1]
    summary_index = stats.rfind("Summary:")

    summary_list = stats[summary_index:].split()
    i_lufs = float(summary_list[summary_list.index("I:") + 1])
    i_thresh = float(summary_list[summary_list.index("I:") + 4])
    lra = float(summary_list[summary_list.index("LRA:") + 1])
    lra_thresh = float(summary_list[summary_list.index("LRA:") + 4])
    lra_low = float(summary_list[summary_list.index("low:") + 1])
    lra_high = float(summary_list[summary_list.index("high:") + 1])
    stats_dict = {
        "I": i_lufs,
        "I Threshold": i_thresh,
        "LRA": lra,
        "LRA Threshold": lra_thresh,
        "LRA Low": lra_low,
        "LRA High": lra_high,
    }

    return stats_dict


def ffprobe_offset_and_codec(path: str) -> Tuple[float, str]:
    """Given a path to a file, returns the start time offset and codec of
    the first audio stream.
    """
    ff = ffmpy.FFprobe(
        inputs={path: None},
        global_options="-show_entries format=start_time:stream=duration,start_time,codec_type,codec_name,start_pts,time_base -of json -v quiet",
    )
    streams = json.loads(ff.run(stdout=subprocess.PIPE)[0])["streams"]
    seconds_offset = 0.0
    codec = None

    # Get the offset and codec of the first audio stream we find
    # and return its start time, if it has one.
    for stream in streams:
        if stream["codec_type"] == "audio":
            seconds_offset = stream.get("start_time", 0.0)
            codec = stream.get("codec_name")
            break
    return float(seconds_offset), codec


class FFMPEGMixin:
    _loudness = None

    def ffmpeg_loudness(self, quiet: bool=True):
        """Computes loudness of audio file using FFMPEG.

        Parameters
        ----------
        quiet : bool, optional
            Whether to show FFMPEG output during computation,
            by default True

        Returns
        -------
        paddle.Tensor
            Loudness of every item in the batch, computed via
            FFMPEG.
        """
        loudness = []

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            for i in range(self.batch_size):
                self[i].write(f.name)
                loudness_stats = r128stats(f.name, quiet=quiet)
                loudness.append(loudness_stats["I"])

        self._loudness = paddle.to_tensor(np.array(loudness)).astype("float32")
        return self.loudness()
