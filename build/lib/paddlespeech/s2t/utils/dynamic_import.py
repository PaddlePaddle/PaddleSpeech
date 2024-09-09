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
import importlib
import inspect
from typing import Any
from typing import Dict
from typing import List
from typing import Text

from paddlespeech.s2t.utils.log import Log
from paddlespeech.s2t.utils.tensor_utils import has_tensor

logger = Log(__name__).getlog()

__all__ = ["dynamic_import", "instance_class"]


def dynamic_import(import_path, alias=dict()):
    """dynamic import module and class

    :param str import_path: syntax 'module_name:class_name'
        e.g., 'paddlespeech.s2t.models.u2:U2Model'
    :param dict alias: shortcut for registered class
    :return: imported class
    """
    if import_path not in alias and ":" not in import_path:
        raise ValueError(
            "import_path should be one of {} or "
            'include ":", e.g. "paddlespeech.s2t.models.u2:U2Model" : '
            "{}".format(set(alias), import_path))
    if ":" not in import_path:
        import_path = alias[import_path]

    module_name, objname = import_path.split(":")
    m = importlib.import_module(module_name)
    return getattr(m, objname)


def filter_valid_args(args: Dict[Text, Any], valid_keys: List[Text]):
    # filter by `valid_keys` and filter `val` is not None
    new_args = {
        key: val
        for key, val in args.items() if (key in valid_keys and val is not None)
    }
    return new_args


def filter_out_tensor(args: Dict[Text, Any]):
    return {key: val for key, val in args.items() if not has_tensor(val)}


def instance_class(module_class, args: Dict[Text, Any]):
    valid_keys = inspect.signature(module_class).parameters.keys()
    new_args = filter_valid_args(args, valid_keys)
    logger.info(
        f"Instance: {module_class.__name__} {filter_out_tensor(new_args)}.")
    return module_class(**new_args)
