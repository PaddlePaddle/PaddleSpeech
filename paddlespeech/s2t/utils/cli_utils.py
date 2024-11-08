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
# Modified from espnet(https://github.com/espnet/espnet)
import sys
from collections.abc import Sequence

import numpy

from paddlespeech.utils.argparse import strtobool as dist_strtobool


def strtobool(x):
    # paddlespeech.utils.argparse.strtobool returns integer, but it's confusing,
    return bool(dist_strtobool(x))


def get_commandline_args():
    extra_chars = [
        " ",
        ";",
        "&",
        "(",
        ")",
        "|",
        "^",
        "<",
        ">",
        "?",
        "*",
        "[",
        "]",
        "$",
        "`",
        '"',
        "\\",
        "!",
        "{",
        "}",
    ]

    # Escape the extra characters for shell
    argv = [
        arg.replace("'", "'\\''") if all(char not in arg
                                         for char in extra_chars) else
        "'" + arg.replace("'", "'\\''") + "'" for arg in sys.argv
    ]

    return sys.executable + " " + " ".join(argv)


def is_scipy_wav_style(value):
    # If Tuple[int, numpy.ndarray] or not
    return (isinstance(value, Sequence) and len(value) == 2 and
            isinstance(value[0], int) and isinstance(value[1], numpy.ndarray))


def assert_scipy_wav_style(value):
    assert is_scipy_wav_style(
        value), "Must be Tuple[int, numpy.ndarray], but got {}".format(
            type(value) if not isinstance(value, Sequence) else "{}[{}]".format(
                type(value), ", ".join(str(type(v)) for v in value)))
