# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import distributed as dist

from paddlespeech.t2s.utils import mp_tools

__all__ = ["load_parameters", "save_parameters"]


def _load_latest_checkpoint(checkpoint_dir: str) -> int:
    """Get the iteration number corresponding to the latest saved checkpoint.

    Args:
        checkpoint_dir (str): the directory where checkpoint is saved.

    Returns:
        int: the latest iteration number.
    """
    checkpoint_record = os.path.join(checkpoint_dir, "checkpoint")
    if (not os.path.isfile(checkpoint_record)):
        return 0

    # Fetch the latest checkpoint index.
    with open(checkpoint_record, "rt") as handle:
        latest_checkpoint = handle.readline().split()[-1]
        iteration = int(latest_checkpoint.split("-")[-1])

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
    with open(checkpoint_record, "wt") as handle:
        handle.write("model_checkpoint_path: step-{}".format(iteration))


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
            stored in the checkpoint_path and the argument 'checkpoint_dir' will
            be ignored. Defaults to None.

    Returns:
        iteration (int): number of iterations that the loaded checkpoint has 
            been trained.
    """
    if checkpoint_path is not None:
        iteration = int(os.path.basename(checkpoint_path).split("-")[-1])
    elif checkpoint_dir is not None:
        iteration = _load_latest_checkpoint(checkpoint_dir)
        if iteration == 0:
            return iteration
        checkpoint_path = os.path.join(checkpoint_dir,
                                       "step-{}".format(iteration))
    else:
        raise ValueError(
            "At least one of 'checkpoint_dir' and 'checkpoint_path' should be specified!"
        )

    local_rank = dist.get_rank()

    params_path = checkpoint_path + ".pdparams"
    model_dict = paddle.load(params_path)
    model.set_state_dict(model_dict)
    print("[checkpoint] Rank {}: loaded model from {}".format(local_rank,
                                                              params_path))

    optimizer_path = checkpoint_path + ".pdopt"
    if optimizer and os.path.isfile(optimizer_path):
        optimizer_dict = paddle.load(optimizer_path)
        optimizer.set_state_dict(optimizer_dict)
        print("[checkpoint] Rank {}: loaded optimizer state from {}".format(
            local_rank, optimizer_path))

    return iteration


@mp_tools.rank_zero_only
def save_parameters(checkpoint_dir, iteration, model, optimizer=None):
    """Checkpoint the latest trained model parameters.

    Args:
        checkpoint_dir (str): the directory where checkpoint is saved.
        iteration (int): the latest iteration number.
        model (Layer): model to be checkpointed.
        optimizer (Optimizer, optional): optimizer to be checkpointed.
            Defaults to None.

    Returns:
        None
    """
    checkpoint_path = os.path.join(checkpoint_dir, "step-{}".format(iteration))

    model_dict = model.state_dict()
    params_path = checkpoint_path + ".pdparams"
    paddle.save(model_dict, params_path)
    print("[checkpoint] Saved model to {}".format(params_path))

    if optimizer:
        opt_dict = optimizer.state_dict()
        optimizer_path = checkpoint_path + ".pdopt"
        paddle.save(opt_dict, optimizer_path)
        print("[checkpoint] Saved optimzier state to {}".format(optimizer_path))

    _save_checkpoint(checkpoint_dir, iteration)
