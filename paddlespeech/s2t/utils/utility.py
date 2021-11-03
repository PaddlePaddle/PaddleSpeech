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
import distutils.util
import math
import os
import random
import sys
from contextlib import contextmanager
from pprint import pformat
from typing import List

import numpy as np
import paddle
import soundfile

from paddlespeech.s2t.utils.log import Log

logger = Log(__name__).getlog()

__all__ = [
    "all_version", "UpdateConfig", "seed_all", 'print_arguments',
    'add_arguments', "log_add"
]


def all_version():
    vers = {
        "python": sys.version,
        "paddle": paddle.__version__,
        "paddle_commit": paddle.version.commit,
        "soundfile": soundfile.__version__,
    }
    logger.info(f"Deps Module Version:{pformat(list(vers.items()))}")


@contextmanager
def UpdateConfig(config):
    """Update yacs config"""
    config.defrost()
    yield
    config.freeze()


def seed_all(seed: int=20210329):
    """freeze random generator seed."""
    np.random.seed(seed)
    random.seed(seed)
    paddle.seed(seed)


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
    print(f"----------- {filename} Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("-----------------------------------------------------------")


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
    """Stable log add

    Args:
        args (List[int]): log scores

    Returns:
        float: sum of log scores
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def get_subsample(config):
    """Subsample rate from config.

    Args:
        config (yacs.config.CfgNode): yaml config

    Returns:
        int: subsample rate.
    """
    input_layer = config["model"]["encoder_conf"]["input_layer"]
    assert input_layer in ["conv2d", "conv2d6", "conv2d8"]
    if input_layer == "conv2d":
        return 4
    elif input_layer == "conv2d6":
        return 6
    elif input_layer == "conv2d8":
        return 8
