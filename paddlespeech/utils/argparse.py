# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import hashlib
import os
import sys
from typing import Text

import distutils

__all__ = [
    "print_arguments", "add_arguments", "get_commandline_args", "strtobool"
]


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


def print_arguments(args, info=None):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    filename = ""
    if info:
        filename = info["__file__"]
    filename = os.path.basename(filename)
    print(f"----------- {filename} Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("-----------------------------------------------------------")


def strtobool(value):
    """Convert a string value to an integer boolean (1 for True, 0 for False).

    The function recognizes the following strings as True (case insensitive):
    - "yes"
    - "true"
    - "1"

    All other values are considered False.

    NOTE: After Python 3.10, the distutils module, particularly distutils.util, has been partially deprecated. To maintain compatibility with existing code, the strtobool function implemented here.
    """
    if isinstance(value, bool):
        return int(value)
    value = value.strip().lower()
    if value in ('yes', 'true', '1'):
        return 1
    else:
        return 0


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)
