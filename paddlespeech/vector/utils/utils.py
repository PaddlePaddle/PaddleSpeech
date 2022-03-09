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

"""
utilities
"""
import os
import sys
import paddle
import numpy as np

from paddlespeech.vector import _logger as log


def exit_if_not_exist(in_path):
    """
    Check the existence of a file or directory, if not exit, exit the program.

    Args:
        in_path: input dicrector
    """
    if not is_exist(in_path):
        sys.exit(-1)


def is_exist(in_path):
    """
    Check the existence of a file or directory

    Args:
        in_path: input dicrector

    Returns:
        True or False
    """
    if not os.path.exists(in_path):
        log.error("No such file or directory: %s" % (in_path))
        return False

    return True


def get_latest_file(target_dir):
    """
    Get the latest file in target directory

    Args:
        target_dir: target directory

    Returns:
        latest_file: a string or None
    """
    items = os.listdir(target_dir)
    items.sort(key=lambda fn: os.path.getmtime(os.path.join(target_dir, fn)) \
               if not os.path.isdir(os.path.join(target_dir, fn)) else 0)
    latest_file = None if not items else os.path.join(target_dir, items[-1])
    return latest_file


def avg_models(models):
    """
    merge multiple models
    """
    checkpoint_dict = paddle.load(models[0])
    final_state_dict = checkpoint_dict

    if len(models) > 1:
        for model in models[1:]:
            checkpoint_dict = paddle.load(model)
            for k, v in checkpoint_dict.items():
                final_state_dict[k] += v
        for k in final_state_dict.keys():
            final_state_dict[k] /= float(len(models))
            if np.any(np.isnan(final_state_dict[k])):
                print("Nan in %s" % (k))

    return final_state_dict

def Q_from_tokens(token_num):
    """
    get prior model, data from uniform, would support others(guassian) in future
    """
    freq = [1] * token_num
    Q = paddle.to_tensor(freq, dtype = 'float64')
    return Q / Q.sum()


def read_map_file(map_file, key_func=None, value_func=None, values_func=None):
    """ Read map file. First colume is key, the rest columes are values.

    Args:
        map_file: map file
        key_func: convert function for key
        value_func: convert function for each value
        values_func: convert function for values

    Returns:
        dict: key 2 value
        dict: value 2 key
    """
    if not is_exist(map_file):
        sys.exit(0)

    key2val = {}
    val2key = {}
    with open(map_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items = line.split()
            assert len(items) >= 2
            key = items[0] if not key_func else key_func(items[0])
            values = items[1:] if not value_func else [value_func(item) for item in items[1:]]
            if values_func:
                values = values_func(values)
            key2val[key] = values
            for value in values:
                val2key[value] = key

    return key2val, val2key
