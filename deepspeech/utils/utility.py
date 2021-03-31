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
"""Contains common utility functions."""

import math
import numpy as np
import distutils.util

__all__ = [
    'print_arguments', 'add_arguments', "log_add", "remove_duplicates_and_blank"
]


def print_arguments(args):
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
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


def log_add(args: List[int]) -> float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def remove_duplicates_and_blank(hyp: List[int], blank_id=0) -> List[int]:
    """ctc alignment to ctc label ids.

    "abaa-acee-" -> "abaace"

    Args:
        hyp (List[int]): hypotheses ids, (L)
        blank_id (int, optional): blank id. Defaults to 0.

    Returns:
        List[int]: remove dupicate ids, then remove blank id.
    """
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp
