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
from typing import Union

from paddle.optimizer.lr import LRScheduler
from typeguard import check_argument_types

from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.s2t.utils.dynamic_import import instance_class
from paddlespeech.s2t.utils.log import Log

__all__ = ["WarmupLR", "LRSchedulerFactory"]

logger = Log(__name__).getlog()

SCHEDULER_DICT = {
    "noam": "paddle.optimizer.lr:NoamDecay",
    "expdecaylr": "paddle.optimizer.lr:ExponentialDecay",
    "piecewisedecay": "paddle.optimizer.lr:PiecewiseDecay",
}


def register_scheduler(cls):
    """Register scheduler."""
    alias = cls.__name__.lower()
    SCHEDULER_DICT[cls.__name__.lower()] = cls.__module__ + ":" + cls.__name__
    return cls


@register_scheduler
class WarmupLR(LRScheduler):
    """The WarmupLR scheduler
    This scheduler is almost same as NoamLR Scheduler except for following
    difference:
    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    Note that the maximum lr equals to optimizer.lr in this scheduler.
    """

    def __init__(self,
                 warmup_steps: Union[int, float]=25000,
                 learning_rate=1.0,
                 last_epoch=-1,
                 verbose=False,
                 **kwargs):
        assert check_argument_types()
        self.warmup_steps = warmup_steps
        super().__init__(learning_rate, last_epoch, verbose)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps}, lr={self.base_lr}, last_epoch={self.last_epoch})"

    def get_lr(self):
        # self.last_epoch start from zero
        step_num = self.last_epoch + 1
        return self.base_lr * self.warmup_steps**0.5 * min(
            step_num**-0.5, step_num * self.warmup_steps**-1.5)

    def set_step(self, step: int=None):
        '''
        It will update the learning rate in optimizer according to current ``epoch`` .
        The new learning rate will take effect on next ``optimizer.step`` .

        Args:
            step (int, None): specify current epoch. Default: None. Auto-increment from last_epoch=-1.
        Returns:
            None
        '''
        self.step(epoch=step)


@register_scheduler
class ConstantLR(LRScheduler):
    """
    Args:
        learning_rate (float): The initial learning rate. It is a python float number.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .

    Returns:
        ``ConstantLR`` instance to schedule learning rate.
    """

    def __init__(self, learning_rate, last_epoch=-1, verbose=False):
        super().__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return self.base_lr

@register_scheduler
class NewBobScheduler(LRScheduler):
    """
    Args:
        learning_rate (float): The initial learning rate. It is a python float number.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False`` .

    Returns:
        ``ConstantLR`` instance to schedule learning rate.
    """
    def __init__(
        self,
        learning_rate,
        annealing_factor=0.5,
        improvement_threshold=0.0025,
        patient=0,
    ):
        self.hyperparam_value = learning_rate
        self.annealing_factor = annealing_factor
        self.improvement_threshold = improvement_threshold
        self.patient = patient
        self.metric_values = []
        self.current_patient = self.patient

    def __call__(self, metric_value):
        """Returns the current and new value for the hyperparameter.

        Arguments
        ---------
        metric_value : int
            A number for determining whether to change the hyperparameter value.
        """
        old_value = new_value = self.hyperparam_value
        if len(self.metric_values) > 0:
            prev_metric = self.metric_values[-1]
            # Update value if improvement too small and patience is 0
            if prev_metric == 0:  # Prevent division by zero
                improvement = 0
            else:
                improvement = (prev_metric - metric_value) / prev_metric
            if improvement < self.improvement_threshold:
                if self.current_patient == 0:
                    new_value *= self.annealing_factor
                    self.current_patient = self.patient
                else:
                    self.current_patient -= 1
        # Store relevant info
        self.metric_values.append(metric_value)
        self.hyperparam_value = new_value

        return old_value, new_value


def dynamic_import_scheduler(module):
    """Import Scheduler class dynamically.

    Args:
        module (str): module_name:class_name or alias in `SCHEDULER_DICT`

    Returns:
        type: Scheduler class

    """
    module_class = dynamic_import(module, SCHEDULER_DICT)
    assert issubclass(module_class,
                      LRScheduler), f"{module} does not implement LRScheduler"
    return module_class


class LRSchedulerFactory():
    @classmethod
    def from_args(cls, name: str, args: Dict[Text, Any]):
        module_class = dynamic_import_scheduler(name.lower())
        return instance_class(module_class, args)
