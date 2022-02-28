#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright     2020    Zeng Xingui(zengxingui@baidu.com)
#
########################################################################

"""Helper functions to help with reproducibility of models. """

import os
import random

import numpy as np
import paddle

from sidt import _logger as log


def seed_everything(seed):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random and sets PYTHONHASHSEED environment variable.
    In addition, sets the env variable `SIDT_GLOBAL_SEED` which will be passed to
    spawned subprocesses (e.g. ddp_spawn backend).

    Args:
        seed: the integer value seed for global random state in SIDT(sid_nnet_training).
            If `None`, will read seed from `SIDT_GLOBAL_SEED` env variable
            or select it randomly.

    Returns:
        seed: the integer value seed

    Raises:
        TypeError and ValueError
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        if seed is None:
            seed = os.environ.get("SIDT_GLOBAL_SEED", _select_seed_randomly(min_seed_value, max_seed_value))
        seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    if (seed > max_seed_value) or (seed < min_seed_value):
        log.warning(
            f"{seed} is not in bounds, \
            numpy accepts from {min_seed_value} to {max_seed_value}"
        )
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["SIDT_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    return seed


def _select_seed_randomly(min_seed_value=0, max_seed_value=255):
    seed = random.randint(min_seed_value, max_seed_value)
    log.warning(f"No correct seed found, seed set to {seed}")
    return seed
