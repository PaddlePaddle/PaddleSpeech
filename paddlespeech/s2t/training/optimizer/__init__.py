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
from typing import Any
from typing import Dict
from typing import Text

import paddle
from paddle.optimizer import Optimizer
from paddle.regularizer import L2Decay

from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.s2t.utils.dynamic_import import instance_class
from paddlespeech.s2t.utils.log import Log

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


@register_optimizer
class Noam(paddle.optimizer.Adam):
    """Seem to: espnet/nets/pytorch_backend/transformer/optimizer.py """

    def __init__(self,
                 learning_rate=0,
                 beta1=0.9,
                 beta2=0.98,
                 epsilon=1e-9,
                 parameters=None,
                 weight_decay=None,
                 grad_clip=None,
                 lazy_mode=False,
                 multi_precision=False,
                 name=None):
        super().__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            lazy_mode=lazy_mode,
            multi_precision=multi_precision,
            name=name)

    def __repr__(self):
        echo = f"<{self.__class__.__module__}.{self.__class__.__name__} object at {hex(id(self))}> "
        echo += f"learning_rate: {self._learning_rate}, "
        echo += f"(beta1: {self._beta1} beta2: {self._beta2}), "
        echo += f"epsilon: {self._epsilon}"


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

        grad_clip = paddle.nn.ClipGradByGlobalNorm(
            args['grad_clip']) if "grad_clip" in args else None
        weight_decay = L2Decay(
            args['weight_decay']) if "weight_decay" in args else None
        if weight_decay:
            logger.info(f'<WeightDecay - {weight_decay}>')
        if grad_clip:
            logger.info(f'<GradClip - {grad_clip}>')

        module_class = dynamic_import_optimizer(name.lower())
        args.update({"grad_clip": grad_clip, "weight_decay": weight_decay})
        opt = instance_class(module_class, args)
        if "__repr__" in vars(opt):
            logger.info(f"{opt}")
        else:
            logger.info(
                f"<Optimizer {module_class.__module__}.{module_class.__name__}> LR: {args['learning_rate']}"
            )
        return opt
