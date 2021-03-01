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
