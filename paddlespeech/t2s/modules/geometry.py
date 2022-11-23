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
import numpy as np
import paddle


def shuffle_dim(x, axis, perm=None):
    """Permute input tensor along aixs given the permutation or randomly.
    
    Args:
        x (Tensor): 
            The input tensor.
        axis (int): 
            The axis to shuffle.
        perm (List[int], ndarray, optional): 
            The order to reorder the tensor along the ``axis``-th dimension.
            It is a permutation of ``[0, d)``, where d is the size of the
            ``axis``-th dimension of the input tensor. If not provided,
            a random permutation is used. Defaults to None.

    Returns:
        Tensor: The shuffled tensor, which has the same shape as x does.
    """
    size = x.shape[axis]
    if perm is not None and len(perm) != size:
        raise ValueError("length of permutation should equals the input "
                         "tensor's axis-th dimension's size")
    if perm is not None:
        perm = np.array(perm)
    else:
        perm = np.random.permutation(size)

    perm = paddle.to_tensor(perm)
    out = paddle.gather(x, perm, axis)
    return out
