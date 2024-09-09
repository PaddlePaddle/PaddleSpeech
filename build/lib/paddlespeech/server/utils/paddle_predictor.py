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
import os
from typing import List
from typing import Optional

import paddle
from paddle.inference import Config
from paddle.inference import create_predictor


def init_predictor(model_dir: Optional[os.PathLike]=None,
                   model_file: Optional[os.PathLike]=None,
                   params_file: Optional[os.PathLike]=None,
                   predictor_conf: dict=None):
    """Create predictor with Paddle inference

    Args:
        model_dir (Optional[os.PathLike], optional): The path of the static model saved in the model layer. Defaults to None.
        model_file (Optional[os.PathLike], optional): *.pdmodel file path. Defaults to None.
        params_file (Optional[os.PathLike], optional): *.pdiparams file path.. Defaults to None.
        predictor_conf (dict, optional): The configuration parameters of predictor. Defaults to None.

    Returns:
        predictor (PaddleInferPredictor): created predictor
    """
    if model_dir is not None:
        assert os.path.isdir(model_dir), 'Please check model dir.'
        config = Config(args.model_dir)
    else:
        assert os.path.isfile(model_file) and os.path.isfile(
            params_file), 'Please check model and parameter files.'
        config = Config(model_file, params_file)

    # set device
    if predictor_conf["device"]:
        device = predictor_conf["device"]
    else:
        device = paddle.get_device()
    if "gpu" in device:
        gpu_id = device.split(":")[-1]
        config.enable_use_gpu(1000, int(gpu_id))

    # IR optim
    if predictor_conf["switch_ir_optim"]:
        config.switch_ir_optim()

    # glog
    if not predictor_conf["glog_info"]:
        config.disable_glog_info()

    # config summary
    if predictor_conf["summary"]:
        print(config.summary())

    # memory optim
    config.enable_memory_optim()

    predictor = create_predictor(config)
    return predictor


def run_model(predictor, input: List) -> List:
    """ run predictor

    Args:
        predictor: paddle inference predictor
        input (list): The input of predictor

    Returns:
        list: result list
    """
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        input_handle = predictor.get_input_handle(name)
        input_handle.copy_from_cpu(input[i])
    # do the inference
    predictor.run()
    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_handle = predictor.get_output_handle(name)
        output_data = output_handle.copy_to_cpu()
        results.append(output_data)

    return results
