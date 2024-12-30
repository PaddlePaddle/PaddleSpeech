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
from paddle.framework import core

__all__ = ["convert_dtype_to_np_dtype_"]


def convert_dtype_to_np_dtype_(dtype):
    """
    Convert paddle's data type to corrsponding numpy data type.

    Args:
        dtype(np.dtype): 
            the data type in paddle.

    Returns:
        type: the data type in numpy.

    """
    if dtype is core.VarDesc.VarType.FP32:
        return np.float32
    elif dtype is core.VarDesc.VarType.FP64:
        return np.float64
    elif dtype is core.VarDesc.VarType.FP16:
        return np.float16
    elif dtype is core.VarDesc.VarType.BOOL:
        return np.bool_
    elif dtype is core.VarDesc.VarType.INT32:
        return np.int32
    elif dtype is core.VarDesc.VarType.INT64:
        return np.int64
    elif dtype is core.VarDesc.VarType.INT16:
        return np.int16
    elif dtype is core.VarDesc.VarType.INT8:
        return np.int8
    elif dtype is core.VarDesc.VarType.UINT8:
        return np.uint8
    elif dtype is core.VarDesc.VarType.BF16:
        return np.uint16
    else:
        raise ValueError("Not supported dtype %s" % dtype)
