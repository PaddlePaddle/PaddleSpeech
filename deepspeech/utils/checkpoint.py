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

import os
import logging
import re
import json
from typing import Union

import paddle
from paddle import distributed as dist
from paddle.optimizer import Optimizer

from deepspeech.utils import mp_tools

logger = logging.getLogger(__name__)

__all__ = ["load_parameters", "save_parameters"]


def _load_latest_checkpoint(checkpoint_dir: str) -> int:
    """Get the iteration number corresponding to the latest saved checkpoint.
    Args:
        checkpoint_dir (str): the directory where checkpoint is saved.
    Returns:
        int: the latest iteration number. -1 for no checkpoint to load.
    """
    checkpoint_record = os.path.join(checkpoint_dir, "checkpoint")
    if not os.path.isfile(checkpoint_record):
        return -1

    # Fetch the latest checkpoint index.
    with open(checkpoint_record, "rt") as handle:
        latest_checkpoint = handle.readlines()[-1].strip()
        iteration = int(latest_checkpoint.split(":")[-1])
    return iteration


def _save_checkpoint(checkpoint_dir: str, iteration: int):
    """Save the iteration number of the latest model to be checkpointed.
    Args:
        checkpoint_dir (str): the directory where checkpoint is saved.
        iteration (int): the latest iteration number.
    Returns:
        None
    """
    checkpoint_record = os.path.join(checkpoint_dir, "checkpoint")
    # Update the latest checkpoint index.
    with open(checkpoint_record, "a+") as handle:
        handle.write("model_checkpoint_path:{}\n".format(iteration))


def load_parameters(model,
                    optimizer=None,
                    checkpoint_dir=None,
                    checkpoint_path=None):
    """Load a specific model checkpoint from disk. 
    Args:
        model (Layer): model to load parameters.
        optimizer (Optimizer, optional): optimizer to load states if needed.
            Defaults to None.
        checkpoint_dir (str, optional): the directory where checkpoint is saved.
        checkpoint_path (str, optional): if specified, load the checkpoint
            stored in the checkpoint_path(prefix) and the argument 'checkpoint_dir' will 
            be ignored. Defaults to None. 
    Returns:
        configs (dict): epoch or step, lr and other meta info should be saved.
    """
    configs = {}

    if checkpoint_path is not None:
        tag = os.path.basename(checkpoint_path).split(":")[-1]
    elif checkpoint_dir is not None:
        iteration = _load_latest_checkpoint(checkpoint_dir)
        if iteration == -1:
            return configs
        checkpoint_path = os.path.join(checkpoint_dir, "{}".format(iteration))
    else:
        raise ValueError(
            "At least one of 'checkpoint_dir' and 'checkpoint_path' should be specified!"
        )

    rank = dist.get_rank()

    params_path = checkpoint_path + ".pdparams"
    model_dict = paddle.load(params_path)
    model.set_state_dict(model_dict)
    logger.info("Rank {}: loaded model from {}".format(rank, params_path))

    optimizer_path = checkpoint_path + ".pdopt"
    if optimizer and os.path.isfile(optimizer_path):
        optimizer_dict = paddle.load(optimizer_path)
        optimizer.set_state_dict(optimizer_dict)
        logger.info("Rank {}: loaded optimizer state from {}".format(
            rank, optimizer_path))

    info_path = re.sub('.pdparams$', '.json', params_path)
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = json.load(fin)
    return configs


@mp_tools.rank_zero_only
def save_parameters(checkpoint_dir: str,
                    tag_or_iteration: Union[int, str],
                    model: paddle.nn.Layer,
                    optimizer: Optimizer=None,
                    infos: dict=None):
    """Checkpoint the latest trained model parameters.
    Args:
        checkpoint_dir (str): the directory where checkpoint is saved.
        tag_or_iteration (int or str): the latest iteration(step or epoch) number.
        model (Layer): model to be checkpointed.
        optimizer (Optimizer, optional): optimizer to be checkpointed.
            Defaults to None.
        infos (dict or None): any info you want to save.
    Returns:
        None
    """
    checkpoint_path = os.path.join(checkpoint_dir,
                                   "{}".format(tag_or_iteration))

    model_dict = model.state_dict()
    params_path = checkpoint_path + ".pdparams"
    paddle.save(model_dict, params_path)
    logger.info("Saved model to {}".format(params_path))

    if optimizer:
        opt_dict = optimizer.state_dict()
        optimizer_path = checkpoint_path + ".pdopt"
        paddle.save(opt_dict, optimizer_path)
        logger.info("Saved optimzier state to {}".format(optimizer_path))

    info_path = re.sub('.pdparams$', '.json', params_path)
    infos = {} if infos is None else infos
    with open(info_path, 'w') as fout:
        data = json.dumps(infos)
        fout.write(data)

    if isinstance(tag_or_iteration, int):
        _save_checkpoint(checkpoint_dir, tag_or_iteration)
