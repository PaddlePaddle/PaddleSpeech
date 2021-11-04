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
import inspect


def check_kwargs(func, kwargs, name=None):
    """check kwargs are valid for func

    If kwargs are invalid, raise TypeError as same as python default
    :param function func: function to be validated
    :param dict kwargs: keyword arguments for func
    :param str name: name used in TypeError (default is func name)
    """
    try:
        params = inspect.signature(func).parameters
    except ValueError:
        return
    if name is None:
        name = func.__name__
    for k in kwargs.keys():
        if k not in params:
            raise TypeError(
                f"{name}() got an unexpected keyword argument '{k}'")
