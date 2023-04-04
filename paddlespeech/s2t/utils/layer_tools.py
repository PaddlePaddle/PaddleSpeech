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
import numpy as np
from paddle import nn

__all__ = [
    "summary", "gradient_norm", "freeze", "unfreeze", "print_grads",
    "print_params"
]


def summary(layer: nn.Layer, print_func=print):
    if print_func is None:
        return
    num_params = num_elements = 0
    for name, param in layer.state_dict().items():
        if print_func:
            print_func("{} | {} | {}".format(name, param.shape,
                                             np.prod(param.shape)))
        num_elements += np.prod(param.shape)
        num_params += 1
    if print_func:
        num_elements = num_elements / 1024**2
        print_func(
            f"Total parameters: {num_params}, {num_elements:.2f}M elements.")


def print_grads(model, print_func=print):
    if print_func is None:
        return
    for n, p in model.named_parameters():
        msg = f"param grad: {n}: shape: {p.shape} grad: {p.grad}"
        print_func(msg)


def print_params(model, print_func=print):
    if print_func is None:
        return
    total = 0.0
    num_params = 0.0
    for n, p in model.named_parameters():
        msg = f"{n} | {p.shape} | {np.prod(p.shape)} | {not p.stop_gradient}"
        total += np.prod(p.shape)
        num_params += 1
        if print_func:
            print_func(msg)
    if print_func:
        total = total / 1024**2
        print_func(f"Total parameters: {num_params}, {total:.2f}M elements.")


def gradient_norm(layer: nn.Layer):
    grad_norm_dict = {}
    for name, param in layer.state_dict().items():
        if param.trainable:
            grad = param.gradient()  # return numpy.ndarray
            grad_norm_dict[name] = np.linalg.norm(grad) / grad.size
    return grad_norm_dict


def recursively_remove_weight_norm(layer: nn.Layer):
    for layer in layer.sublayers():
        try:
            nn.utils.remove_weight_norm(layer)
        except ValueError as e:
            # ther is not weight norm hoom in this layer
            pass


def freeze(layer: nn.Layer):
    for param in layer.parameters():
        param.trainable = False


def unfreeze(layer: nn.Layer):
    for param in layer.parameters():
        param.trainable = True
