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
import paddle
from paddle import nn

optim_classes = dict(
    adadelta=paddle.optimizer.Adadelta,
    adagrad=paddle.optimizer.Adagrad,
    adam=paddle.optimizer.Adam,
    adamax=paddle.optimizer.Adamax,
    adamw=paddle.optimizer.AdamW,
    lamb=paddle.optimizer.Lamb,
    momentum=paddle.optimizer.Momentum,
    rmsprop=paddle.optimizer.RMSProp,
    sgd=paddle.optimizer.SGD, )


def build_optimizers(model: nn.Layer,
                     optim='adadelta',
                     max_grad_norm=None,
                     learning_rate=0.01) -> paddle.optimizer:
    optim_class = optim_classes.get(optim)
    if optim_class is None:
        raise ValueError(f"must be one of {list(optim_classes)}: {optim}")
    else:
        grad_clip = None
        if max_grad_norm:
            grad_clip = paddle.nn.ClipGradByGlobalNorm(max_grad_norm)
        optim = optim_class(
            parameters=model.parameters(),
            learning_rate=learning_rate,
            grad_clip=grad_clip)

    optimizers = optim
    return optimizers
