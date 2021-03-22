# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""Contains data helper functions."""

import numpy as np
import math
import json
import codecs
import os
import tarfile
import time
from threading import Thread
from multiprocessing import Process, Manager, Value

from paddle.dataset.common import md5file


def read_manifest(manifest_path, max_duration=float('inf'), min_duration=0.0):
    """Load and parse manifest file.

    Instances with durations outside [min_duration, max_duration] will be
    filtered out.

    :param manifest_path: Manifest file to load and parse.
    :type manifest_path: str
    :param max_duration: Maximal duration in seconds for instance filter.
    :type max_duration: float
    :param min_duration: Minimal duration in seconds for instance filter.
    :type min_duration: float
    :return: Manifest parsing results. List of dict.
    :rtype: list
    :raises IOError: If failed to parse the manifest.
    """
    manifest = []
    for json_line in codecs.open(manifest_path, 'r', 'utf-8'):
        try:
            json_data = json.loads(json_line)
        except Exception as e:
            raise IOError("Error reading manifest: %s" % str(e))
        if (json_data["duration"] <= max_duration and
                json_data["duration"] >= min_duration):
            manifest.append(json_data)
    return manifest


def rms_to_db(rms: float):
    """Root Mean Square to dB.
    Args:
        rms ([float]): root mean square

    Returns:
        float: dB
    """
    return 20.0 * math.log10(max(1e-16, rms))


def rms_to_dbfs(rms: float):
    """Root Mean Square to dBFS.
    https://fireattack.wordpress.com/2017/02/06/replaygain-loudness-normalization-and-applications/
    Audio is mix of sine wave, so 1 amp sine wave's Full scale is 0.7071, equal to -3.0103dB.
   
    dB = dBFS + 3.0103
    dBFS = db - 3.0103
    e.g. 0 dB = -3.0103 dBFS

    Args:
        rms ([float]): root mean square

    Returns:
        float: dBFS
    """
    return rms_to_db(rms) - 3.0103


def max_dbfs(sample_data: np.ndarry):
    """Peak dBFS based on the maximum energy sample. 

    Args:
        sample_data ([np.ndarry]): float array, [-1, 1].

    Returns:
        float: dBFS 
    """
    # Peak dBFS based on the maximum energy sample. Will prevent overdrive if used for normalization.
    return rms_to_dbfs(max(abs(np.min(sample_data)), abs(np.max(sample_data))))


def mean_dbfs(sample_data):
    """Peak dBFS based on the RMS energy. 

    Args:
        sample_data ([np.ndarry]): float array, [-1, 1].

    Returns:
        float: dBFS 
    """
    return rms_to_dbfs(
        math.sqrt(np.mean(np.square(sample_data, dtype=np.float64))))


def gain_db_to_ratio(gain_db: float):
    """dB to ratio

    Args:
        gain_db (float): gain in dB

    Returns:
        float: scale in amp
    """
    return math.pow(10.0, gain_db / 20.0)


def normalize_audio(sample_data: np.ndarry, dbfs: float=-3.0103):
    """Nomalize audio to dBFS.
    
    Args:
        sample_data (np.ndarry): input wave samples, [-1, 1].
        dbfs (float, optional): target dBFS. Defaults to -3.0103.

    Returns:
        np.ndarry: normalized wave
    """
    return np.maximum(
        np.minimum(sample_data * gain_db_to_ratio(dbfs - max_dbfs(sample_data)),
                   1.0), -1.0)
