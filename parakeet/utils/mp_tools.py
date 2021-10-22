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
from functools import wraps

from paddle import distributed as dist

__all__ = ["rank_zero_only"]


def rank_zero_only(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if dist.get_rank() != 0:
            return
        result = func(*args, **kwargs)
        return result

    return wrapper
