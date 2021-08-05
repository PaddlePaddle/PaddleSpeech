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
from typing import Any
from typing import Dict
from typing import Text

from paddle.optimizer import Optimizer
from paddle.regularizer import L2Decay

from deepspeech.training.gradclip import ClipGradByGlobalNormWithLog
from deepspeech.utils.dynamic_import import dynamic_import
from deepspeech.utils.dynamic_import import filter_valid_args
from deepspeech.utils.log import Log

__all__ = ["OptimizerFactory"]

logger = Log(__name__).getlog()

OPTIMIZER_DICT = {
    "sgd": "paddle.optimizer:SGD",
    "momentum": "paddle.optimizer:Momentum",
    "adadelta": "paddle.optimizer:Adadelta",
    "adam": "paddle.optimizer:Adam",
    "adamw": "paddle.optimizer:AdamW",
}


def register_optimizer(cls):
    """Register optimizer."""
    alias = cls.__name__.lower()
    OPTIMIZER_DICT[cls.__name__.lower()] = cls.__module__ + ":" + cls.__name__
    return cls


def dynamic_import_optimizer(module):
    """Import Optimizer class dynamically.

    Args:
        module (str): module_name:class_name or alias in `OPTIMIZER_DICT`

    Returns:
        type: Optimizer class

    """
    module_class = dynamic_import(module, OPTIMIZER_DICT)
    assert issubclass(module_class,
                      Optimizer), f"{module} does not implement Optimizer"
    return module_class


class OptimizerFactory():
    @classmethod
    def from_args(cls, name: str, args: Dict[Text, Any]):
        assert "parameters" in args, "parameters not in args."
        assert "learning_rate" in args, "learning_rate not in args."

        grad_clip = ClipGradByGlobalNormWithLog(
            args['grad_clip']) if "grad_clip" in args else None
        weight_decay = L2Decay(
            args['weight_decay']) if "weight_decay" in args else None
        module_class = dynamic_import_optimizer(name.lower())

        if weight_decay:
            logger.info(f'WeightDecay: {weight_decay}')
        if grad_clip:
            logger.info(f'GradClip: {grad_clip}')
        logger.info(
            f"Optimizer: {module_class.__name__} {args['learning_rate']}")

        args.update({"grad_clip": grad_clip, "weight_decay": weight_decay})

        args = filter_valid_args(args)
        return module_class(**args)
