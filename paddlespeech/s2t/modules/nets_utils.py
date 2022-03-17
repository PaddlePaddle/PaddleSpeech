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
from paddle import nn
from typeguard import check_argument_types

def initialize(model: nn.Layer, init: str):
    """Initialize weights of a neural network module.

    Parameters are initialized using the given method or distribution.

    Custom initialization routines can be implemented into submodules

    Args:
        model (nn.Layer): Target.
        init (str): Method of initialization.
    """
    assert check_argument_types()

    if init == "xavier_uniform":
        nn.initializer.set_global_initializer(nn.initializer.XavierUniform(),
                                              nn.initializer.Constant())
    elif init == "xavier_normal":
        nn.initializer.set_global_initializer(nn.initializer.XavierNormal(),
                                              nn.initializer.Constant())
    elif init == "kaiming_uniform":
        nn.initializer.set_global_initializer(nn.initializer.KaimingUniform(),
                                              nn.initializer.KaimingUniform())
    elif init == "kaiming_normal":
        nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(),
                                              nn.initializer.Constant())
    else:
        raise ValueError("Unknown initialization: " + init)
