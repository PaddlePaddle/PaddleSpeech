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
from contextlib import contextmanager

import paddle
from paddle.framework import core
from paddle.framework import CUDAPlace


def synchronize():
    """Trigger cuda synchronization for better timing."""
    place = paddle.fluid.framework._current_expected_place()
    if isinstance(place, CUDAPlace):
        paddle.fluid.core._cuda_synchronize(place)


@contextmanager
def nvtx_span(name):
    try:
        core.nvprof_nvtx_push(name)
        yield
    finally:
        core.nvprof_nvtx_pop()
